"""冻结 Residual48 的可选精确推理后端；训练和默认 PyTorch 路径不受影响。"""
from collections import OrderedDict
from functools import lru_cache
import math
from pathlib import Path
import torch


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent
    return load(name='mycartgs_frozen_inference_v4',
        sources=[str(root/'frozen_inference_bindings.cpp'), str(root/'frozen_inference_kernels.cu')],
        extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--fmad=false'], verbose=False)


class FrozenResidualInference:
    """同形状 GEMM + CUDA 打包；Graph 按实际 K 缓存，禁止 padding 改变 GEMM。"""
    def __init__(self, model, gaussians, graph=False):
        spatial, anchors = model.spatial_compensation, model.anchors
        self.joint_schema = int(getattr(model, 'joint_view_conditioning_schema', 0))
        width = model.local_emd[0].in_features
        if (width not in (11, 19) or (width == 19 and self.joint_schema not in (1, 2, 3, 4)) or
                (width == 11 and self.joint_schema) or anchors.features.shape[1] != 0 or
                hasattr(model, 'selector_set_encoder') or hasattr(model, 'selector_context_residual') or
                hasattr(model, 'selector_spatial_projection') or hasattr(model,'selector_view_residual') or hasattr(anchors, 'ratio_knots') or
                spatial.ratio_feature_width != 48 or not spatial.offset_only or spatial.bounded_xyz or
                hasattr(spatial, 'ratio_gate') or hasattr(spatial, 'geometry_head_schema') or
                getattr(anchors, 'compensation_residual_mode', 0) != 1 or
                getattr(model, 'reliability_gate_config', {}).get('enabled', False)):
            raise ValueError('CUDA frozen inference requires plain11/joint19 Selector and unbounded Residual48 four Heads')
        if bool(self.joint_schema) != hasattr(spatial, 'view_projection'):
            raise ValueError('Joint Selector/Compensation view schemas must match')
        if gaussians.get_xyz.device.type != 'cuda' or gaussians.get_xyz.dtype != torch.float32:
            raise ValueError('CUDA frozen inference requires CUDA FP32 Master')
        if torch.backends.cuda.matmul.allow_tf32:
            raise ValueError('Exact FP32 inference requires matmul.allow_tf32=False')
        self.model, self.pc, self.spatial, self.anchors = model, gaussians, spatial, anchors
        self.ext, self.graph = extension(), bool(graph)
        self.fused_view = True
        self.dynamic_fusion = True
        self.quaternion_fusion = True
        self.narrow_sort = True
        self.summary_fusion = True
        self.fused_gemm_relu = True
        self.parallel_heads = False
        self.head_resources = None
        self.decode_fusion = (hasattr(spatial, 'softplus_scale_schema') or
                              hasattr(spatial, 'physical_scale_additive_schema'))
        self.graphs = OrderedDict()
        self.shape_visits = OrderedDict()
        self.graph_bytes = 0
        self.display_workspace = None
        self.max_graphs, self.max_graph_bytes = 6, 512*1024**2
        self.selector_parameters = [model.local_emd[0].weight, model.local_emd[0].bias,
            model.local_emd[2].weight, model.local_emd[2].bias, model.soft_net.weight, model.soft_net.bias]
        self.compensation_parameters = [spatial.encode.weight, spatial.encode.bias,
            spatial.body.weight, spatial.body.bias, spatial.ratio_projection.weight]
        for name in ('xyz', 'scale', 'rotation', 'other'):
            head = spatial.heads[name]
            self.compensation_parameters += [head[0].weight, head[0].bias, head[2].weight, head[2].bias]
        self.view_weight = spatial.view_projection.weight if self.joint_schema else None
        self.tables = [getattr(anchors, 'compensation_features_'+str(j))
                       for j in range(len(anchors.compensation_values))]
        if not 2 <= len(self.tables) <= 6 or any(table.shape != (len(anchors.centers), 48) for table in self.tables):
            raise ValueError('CUDA frozen inference requires 2-6 Anchor48 tables')
        self.rebuild()

    @torch.no_grad()
    def rebuild(self):
        pc, a, s = self.pc, self.anchors, self.spatial
        self.owners = a.rows.contiguous()
        self.reference = a.offset_scales[self.owners].contiguous()
        offset = (pc.get_xyz-a.centers[self.owners])/self.reference
        self.offset = ((torch.asinh(offset)-s.mean[3:6])/s.stddev[3:6]).clamp(-15, 15).contiguous()
        self.scale, self.rotation, self.opacity = pc.get_scaling, pc.get_rotation, pc.get_opacity
        self.empty = pc.get_xyz.new_empty(0)
        self.base_master = [value.contiguous() for value in (pc.get_xyz, self.opacity, self.scale,
            self.rotation, pc.get_features_dc, pc.get_features_rest)]
        self.compensated_master = list(self.base_master);self.compensated_master[1] = pc._opacity.contiguous()
        self.selector_static = torch.cat((offset, self.scale, pc.get_features_dc.squeeze(1), self.opacity), 1)
        from selector import slot_eligibility
        self.eligible = slot_eligibility(self.model, self.opacity)
        self.eligible_counts = torch.bincount(self.owners[self.eligible], minlength=len(a.centers))
        n, device = len(self.owners), self.owners.device
        self.candidate_storage = torch.empty_like(self.owners)
        self.candidate_counts = torch.empty(3, dtype=torch.int64, device=device)
        self.candidate_workspace = torch.empty(0, dtype=torch.uint8, device=device)
        self.sort_workspace = [torch.empty(n, dtype=dtype, device=device)
                               for dtype in (torch.float32, torch.float32, torch.int64, torch.int64)]
        self.sort_workspace.append(torch.empty(32*n+4096, dtype=torch.uint8, device=device))
        if self.joint_schema:
            centers = a.centers
            self.view_centers4 = torch.cat((centers, torch.ones_like(centers[:, :1])), 1)
            signs = centers.new_tensor([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                                       [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
            corners = centers[:,None,:]+a.offset_scales[:,None,:]*.5*signs
            self.view_corners4 = torch.cat((corners, torch.ones_like(corners[:,:,:1])), 2)
        self.graphs.clear()
        self.shape_visits.clear()
        self.graph_bytes = 0
        self.display_workspace = None

    @torch.no_grad()
    def display_bytes(self, image, reference=None):
        """原生Viewer的有界FP32 staging；一次读回，字节保持原协议。"""
        size = image.numel()*(2 if reference is not None else 1)
        if size*4 > 64*1024**2 or (reference is not None and reference.shape != image.shape):
            return None
        key = (image.device, tuple(image.shape), reference is not None)
        if self.display_workspace is None or self.display_workspace['key'] != key:
            self.display_workspace = dict(key=key, device=torch.empty(size,device=image.device,dtype=torch.float32),
                host=torch.empty(size,pin_memory=True,dtype=torch.float32), event=torch.cuda.Event())
        workspace = self.display_workspace
        self.ext.display_pack(image.contiguous(), self.empty if reference is None else reference.contiguous(), workspace['device'])
        workspace['host'].copy_(workspace['device'], non_blocking=True)
        workspace['event'].record();workspace['event'].synchronize()
        values = workspace['host'].numpy().astype('<f4',copy=False)
        boundary = image.numel()
        return values[:boundary].tobytes(), values[boundary:].tobytes()

    def controls(self, ratio):
        value = float(ratio)
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError('Inference ratio must be in (0,1]')
        knots = self.anchors.compensation_values
        weights = [1.] + [-1.]*(len(knots)-1)
        for j in range(1, len(knots)):
            if value <= knots[j-1]+1e-8:break
            weights[j] = min(1., (value-knots[j-1])/(knots[j]-knots[j-1]))
        lo, hi = self.spatial.ratio_encoding_values
        t = 2*(math.log(value)-math.log(lo))/(math.log(hi)-math.log(lo))-1
        return self.pc.get_xyz.new_tensor(weights+[t, math.sin(math.pi*t/2), math.cos(math.pi*t/2)])

    @torch.no_grad()
    def frustum(self, projection, occupied):
        return self.ext.frustum(self.anchors.centers, self.anchors.offset_scales, projection, occupied)

    @torch.no_grad()
    def scores(self, logits, noise, rows):
        return self.ext.selection_scores(logits, noise, rows)

    @torch.no_grad()
    def candidates(self, visible):
        _, counts, self.candidate_workspace = self.ext.visible_candidates(visible, self.owners,
            self.eligible_counts, self.candidate_storage, self.candidate_counts, self.candidate_workspace)
        n, visible_count, available = counts.cpu().tolist()
        return self.candidate_storage[:n], visible_count, available

    @torch.no_grad()
    def select(self, scores, eligible, k):
        return self.ext.exact_select(scores.contiguous(), eligible, k, self.sort_workspace, self.narrow_sort)

    @torch.no_grad()
    def selector_input(self, rows, ratio, code=None, local=False):
        if self.joint_schema:
            if local and self.joint_schema in (2,3) and (code is None or code.shape != (len(self.anchors.centers),7)):
                raise ValueError('Local Selector requires current Anchor view7')
            return self.ext.selector_view_pack(self.selector_static, rows, self.owners,
                self.pc.get_xyz.new_tensor([ratio]), self.empty if code is None else code,
                bool(local), self.joint_schema in (2, 3))
        return self.ext.selector_pack(self.selector_static, rows,
                                      self.pc.get_xyz.new_tensor([ratio]))

    @torch.no_grad()
    def logits(self, rows, ratio, code=None, local=False):
        return self.ext.selector_logits(self.selector_input(rows, ratio, code, local), self.selector_parameters, self.fused_gemm_relu)

    @torch.no_grad()
    def view_code(self, camera):
        if not self.fused_view:
            return self.model.joint_view_code(camera)
        if self.joint_schema == 1:
            return self.anchors.centers.new_zeros((len(self.anchors.centers), 7))
        vector = camera.camera_center.reshape(1,3)-self.anchors.centers
        # Preserve the original ATen sum/sqrt and all four GEMM shapes/order.
        direction = vector/vector.square().sum(1,keepdim=True).sqrt().clamp_min(1e-6)
        return self.ext.view_code_pack(direction, self.view_centers4@camera.world_view_transform,
            self.view_centers4@camera.full_proj_transform, self.view_corners4@camera.full_proj_transform,
            self.view_corners4@camera.world_view_transform, self.model.selector_view_scene_scale,
            float(camera.image_width)/camera.image_height, camera.znear)

    @torch.no_grad()
    def compensation_input(self, rows, controls):
        return self.ext.compensation_pack(self.offset, rows, self.owners, self.tables, controls[:len(self.tables)])

    def _head_resources(self):
        if self.head_resources is None:
            streams = [torch.cuda.Stream(device=self.pc.get_xyz.device) for _ in range(4)]
            events = [torch.cuda.Event() for _ in range(5)]
            for event in events:event.record()
            self.head_resources = (streams, events, tuple(s.cuda_stream for s in streams),
                                   tuple(e.cuda_event for e in events))
        return self.head_resources[2:]

    @torch.no_grad()
    def residual(self, rows, controls, code=None, local=False):
        if local and self.joint_schema in (2,4) and (code is None or code.shape != (len(self.anchors.centers),7)):
            raise ValueError('Local Compensation requires current Anchor view7')
        condition = (self.ext.condition_pack(rows, self.owners, self.empty if code is None else code,
                     self.joint_schema in (2, 4)) if local and self.joint_schema else self.empty)
        streams, events = self._head_resources() if self.parallel_heads and not self.graph else ((), ())
        return self.ext.compensation_network(self.compensation_input(rows, controls),
            controls[len(self.tables):].view(1, 3), self.reference[rows], self.compensation_parameters,
            condition, self.empty if self.view_weight is None else self.view_weight,
            self.fused_gemm_relu, streams, events)

    def _compute(self, rows, controls, compensation, code=None, local=False):
        pc = self.pc
        if not compensation:
            if self.decode_fusion:
                return self.ext.publish_selected(self.base_master, rows, self.empty, self.empty, True)
            return [value[rows] for value in (pc.get_xyz, self.opacity, self.scale,
                                              self.rotation, pc.get_features_dc, pc.get_features_rest)]
        from transform_field import _normalize_quaternion, _multiply_quaternion
        residual = self.residual(rows, controls, code, local)
        if self.quaternion_fusion:
            rotation = self.ext.quaternion_decode(residual, rows, self.rotation)
        else:
            identity = torch.zeros_like(residual[:, 6:10]);identity[:, 0] = 1
            rotation = _normalize_quaternion(_multiply_quaternion(
                _normalize_quaternion(identity+residual[:, 6:10]), self.rotation[rows]))
            rotation = pc.rotation_activation(rotation)
        if self.decode_fusion:
            if not (hasattr(self.spatial, 'softplus_scale_schema') or hasattr(self.spatial, 'physical_scale_additive_schema')):
                raise ValueError('Fused decode requires Softplus or physical-additive Scale')
            return self.ext.publish_selected(self.compensated_master, rows, residual, rotation,
                                             hasattr(self.spatial, 'softplus_scale_schema'))
        xyz = pc.get_xyz[rows]+residual[:, :3]
        scale = self.spatial.decode_scale(pc._scaling[rows], residual[:, 3:6]).log().exp()
        opacity = (pc._opacity[rows]+residual[:, 10:11]).sigmoid()
        dc = pc.get_features_dc[rows]+residual[:, 11:14].unsqueeze(1)
        return [value.contiguous() for value in (xyz, opacity, scale, rotation, dc, pc.get_features_rest[rows])]

    def _capture(self, rows, controls, compensation, code=None, local=False):
        before = torch.cuda.memory_allocated(rows.device)
        reserved = torch.cuda.memory_reserved(rows.device)
        static_rows, static_controls = rows.clone(), controls.clone()
        static_code = self.empty if code is None else code.clone()
        stream = torch.cuda.Stream(device=rows.device)
        stream.wait_stream(torch.cuda.current_stream(rows.device))
        with torch.cuda.stream(stream):
            for _ in range(3):self._compute(static_rows, static_controls, compensation, static_code, local)
        torch.cuda.current_stream(rows.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
            parts = self._compute(static_rows, static_controls, compensation, static_code, local)
        size = max(0, torch.cuda.memory_allocated(rows.device)-before,
                   torch.cuda.memory_reserved(rows.device)-reserved)
        return dict(graph=graph, rows=static_rows, controls=static_controls, code=static_code, parts=parts, bytes=size)

    @torch.no_grad()
    def render_data(self, rows, ratio, compensation=True, code=None, local=False, capture=True):
        if not len(rows):
            from anchor_frustum import compact_gaussians
            from renderer import prepare_render_data
            return prepare_render_data(compact_gaussians(self.pc, rows))
        controls = self.controls(ratio)
        key = (len(rows), bool(compensation), self.decode_fusion, bool(local))
        replayed = False
        self.shape_visits[key] = self.shape_visits.get(key, 0)+1
        self.shape_visits.move_to_end(key)
        if len(self.shape_visits) > 32:self.shape_visits.popitem(last=False)
        if self.graph and (key in self.graphs or (capture and self.shape_visits[key] >= 3)):
            if key not in self.graphs:
                while len(self.graphs) >= self.max_graphs:
                    _, old = self.graphs.popitem(last=False);self.graph_bytes -= old['bytes']
                entry = self._capture(rows, controls, compensation, code, local)
                if entry['bytes'] <= self.max_graph_bytes:
                    while self.graphs and self.graph_bytes+entry['bytes'] > self.max_graph_bytes:
                        _, old = self.graphs.popitem(last=False);self.graph_bytes -= old['bytes']
                    self.graphs[key] = entry;self.graph_bytes += entry['bytes']
                else:
                    parts = self._compute(rows, controls, compensation, code, local)
            if key in self.graphs:
                entry = self.graphs[key];self.graphs.move_to_end(key)
                entry['rows'].copy_(rows);entry['controls'].copy_(controls)
                if code is not None:entry['code'].copy_(code)
                entry['graph'].replay();parts = entry['parts']
                replayed = True
        else:parts = self._compute(rows, controls, compensation, code, local)
        # 独立结果存储，下一次 Graph replay 不得改写已返回的渲染输入。
        flat = (parts.clone() if replayed else parts) if isinstance(parts, torch.Tensor) else self.ext.publish_render(parts)
        if not torch.isfinite(flat).all():
            raise RuntimeError('Nonfinite compensated attributes at the requested ratio')
        names = ('means3D', 'opacities', 'scales', 'rotations', 'dc', 'shs', 'means2D')
        shapes = [(len(rows), 3), (len(rows), 1), (len(rows), 3), (len(rows), 4),
                  (len(rows),)+tuple(self.pc.get_features_dc.shape[1:]),
                  (len(rows),)+tuple(self.pc.get_features_rest.shape[1:]), (len(rows), 3)]
        inputs = {};start = 0
        for name, shape in zip(names, shapes):
            size = math.prod(shape);inputs[name] = flat[start:start+size].view(shape);start += size
        inputs.update(colors_precomp=None, cov3D_precomp=None)
        return dict(sh_degree=self.pc.active_sh_degree, inputs=inputs)

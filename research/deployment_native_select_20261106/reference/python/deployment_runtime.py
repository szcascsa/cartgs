"""训练结束后的独立冻结部署后端；不被训练或常规评价入口导入。"""
import copy
import math
import numbers
import time
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def deployment_extension():
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent
    return load(name='mycartgs_deployment_fp32_v2',
        sources=[str(root/'deployment_runtime_bindings.cpp'), str(root/'deployment_runtime_kernels.cu'),
                 str(root/'deployment_prepare.cpp'), str(root/'frozen_inference_kernels.cu')],
        extra_cflags=['-O3'], extra_cuda_cflags=['-O3'], verbose=False)


class DeploymentRenderer:
    """独立模型快照，无训练梯度；每帧重算视锥、view、ratio和两网络。

    FP32 FMA顺序允许与cuBLAS不同。输出有独立存储，同一实例由调用者串行使用。
    切换checkpoint或地图时构造新实例，不允许在该实例上训练／更新拓扑。
    """
    @torch.inference_mode(False)
    @torch.no_grad()
    def __init__(self, model, gaussians, config, seed=0, *, scheduler='cpp'):
        from anchor_budget import FixedBudgetRenderer
        if scheduler not in ('python', 'cpp'):
            raise ValueError('Deployment scheduler must be python or cpp')
        # A private frozen snapshot prevents a training update from invalidating deployment data.
        model, gaussians = copy.deepcopy(model), copy.deepcopy(gaussians)
        model.eval().requires_grad_(False)
        for name in ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest'):
            getattr(gaussians, name).requires_grad_(False)
        self._reference = FixedBudgetRenderer(model, gaussians, copy.deepcopy(config), seed, 'cuda')
        self._base = self._reference.inference
        self._ext = deployment_extension()
        self._selector = [p.t().contiguous() if p.dim()==2 else p.contiguous()
                          for p in self._base.selector_parameters]
        self._compensator = [p.t().contiguous() if p.dim()==2 else p.contiguous()
                             for p in self._base.compensation_parameters]
        self._view_weight = (self._base.view_weight.t().contiguous()
                             if self._base.view_weight is not None else self._base.empty)
        snapshot = (self._base.base_master + self._base.tables + self._selector + self._compensator +
                    [self._view_weight, self._base.selector_static, self._base.offset])
        if any(not torch.isfinite(value).all() for value in snapshot):
            raise ValueError('Nonfinite frozen deployment snapshot')
        self.total = self._reference.total
        self.knots = self._reference.knots
        self.backend = 'deployment_cpp' if scheduler=='cpp' else 'deployment_fp32'
        self.scheduler = scheduler
        self._all_rows = torch.arange(self.total, device=self._base.owners.device)
        self._background = self._base.pc.get_xyz.new_zeros(3)
        self._full = self._base.ext.publish_selected(self._base.base_master, self._all_rows,
                                                    self._base.empty, self._base.empty, True)
        self._pipeline = None
        if scheduler=='cpp':
            b=self._base
            state=[b.anchors.centers,b.anchors.offset_scales,self._reference.occupied,b.owners,
                b.eligible_counts,b.eligible,b.selector_static,b.offset,b.reference,self._reference.noise,
                b.view_centers4 if b.joint_schema else b.empty,b.view_corners4 if b.joint_schema else b.empty,
                b.model.selector_view_scene_scale if b.joint_schema else b.empty]
            self._pipeline=self._ext.Pipeline(state,self._selector,self._compensator,b.tables,
                b.compensated_master,b.base_master,self._view_weight,list(self.knots),list(b.spatial.ratio_encoding_values),
                b.joint_schema,b.pc.active_sh_degree,hasattr(b.spatial,'softplus_scale_schema'))

    @classmethod
    @torch.inference_mode(False)
    @torch.no_grad()
    def from_checkpoint(cls, checkpoint, seed=0, *, scheduler='cpp'):
        from budget_viewer import read_config
        from gaussian_model import GaussianModel
        from selector import load_selector
        checkpoint = Path(checkpoint).resolve()
        if (checkpoint/'ply').is_dir():
            checkpoint = checkpoint/'ply'
        candidates = list((checkpoint/'point_cloud').glob('iteration_*/point_cloud.ply'))
        if len(candidates)!=1 or not (checkpoint/'selector.pt').is_file():
            raise ValueError('Expected a finished checkpoint with selector.pt and one point_cloud.ply')
        model = load_selector(str(checkpoint/'selector.pt'), 'cuda')
        gaussians = GaussianModel(3)
        gaussians.load_ply(str(candidates[0]))
        config = read_config(checkpoint/'selector_config.yaml')
        model.set_backend(config['backend'])
        return cls(model, gaussians, config, seed, scheduler=scheduler)

    def _controls(self, ratio):
        weights = [1.] + [-1.]*(len(self.knots)-1)
        for j in range(1, len(weights)):
            if ratio <= self.knots[j-1]+1e-8:
                break
            weights[j] = min(1., (ratio-self.knots[j-1])/(self.knots[j]-self.knots[j-1]))
        lo, hi = self._base.spatial.ratio_encoding_values
        t = 2*(math.log(ratio)-math.log(lo))/(math.log(hi)-math.log(lo))-1
        return weights, [t, math.sin(math.pi*t/2), math.cos(math.pi*t/2)]

    def _packet(self, flat, n):
        pc = self._base.pc
        shapes = [(n,3), (n,1), (n,3), (n,4), (n,)+tuple(pc.get_features_dc.shape[1:]),
                  (n,)+tuple(pc.get_features_rest.shape[1:]), (n,3)]
        inputs = {}
        start = 0
        for name, shape in zip(('means3D','opacities','scales','rotations','dc','shs','means2D'), shapes):
            size = math.prod(shape)
            inputs[name] = flat[start:start+size].view(shape)
            start += size
        inputs.update(colors_precomp=None, cov3D_precomp=None)
        return dict(sh_degree=pc.active_sh_degree, inputs=inputs)

    def view_code(self, camera):
        b = self._base
        if b.joint_schema in (0, 1):
            return b.empty
        return self._ext.view(b.anchors.centers, b.view_centers4, b.view_corners4,
            camera.camera_center.contiguous(), camera.world_view_transform,
            camera.full_proj_transform, b.model.selector_view_scene_scale,
            float(camera.image_width)/camera.image_height, float(camera.znear))

    def prepare(self, camera, budget=None, *, ratio_budget=None, view_conditioned=True,
                compensation=True):
        if torch.is_grad_enabled():
            raise RuntimeError('DeploymentRenderer is inference-only; use torch.inference_mode() after training')
        if sum(v is not None for v in (budget, ratio_budget)) != 1:
            raise ValueError('Use one point budget or local ratio budget')
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, numbers.Integral) or budget<0):
            raise ValueError('Point budget must be a nonnegative integer')
        if ratio_budget is not None and (isinstance(ratio_budget, bool) or not isinstance(ratio_budget, numbers.Real)
                                        or not math.isfinite(ratio_budget) or not 0<=ratio_budget<=1):
            raise ValueError('Ratio must be finite and in [0,1]')
        if not isinstance(view_conditioned, bool) or not isinstance(compensation, bool):
            raise ValueError('View and compensation switches must be bool')
        b = self._base
        if view_conditioned and not b.joint_schema:
            raise ValueError('This checkpoint has no view-conditioned branch')
        if self._pipeline is not None:
            return self._pipeline.prepare(camera.world_view_transform,camera.full_proj_transform,camera.camera_center,
                int(camera.image_width),int(camera.image_height),float(camera.znear),int(budget or 0),
                float(ratio_budget or 0),ratio_budget is not None,view_conditioned,compensation)
        started = time.perf_counter()
        # Visibility and CUB compaction preserve the original candidate order and exact count.
        visible = b.frustum(camera.full_proj_transform, self._reference.occupied)
        rows, visible_count, available = b.candidates(visible)
        n = len(rows)
        if ratio_budget is not None:
            ratio = float(ratio_budget) if n and ratio_budget else None
            budget = math.floor(float(ratio_budget)*n)
        else:
            ratio = min(1., int(budget)/n) if n and budget else None
        k = min(int(budget), available)
        if k:
            code = self.view_code(camera) if view_conditioned else b.empty
            scores, logits, status = self._ext.selector(b.selector_static, rows, b.owners, code,
                self._reference.noise, self._selector, ratio, view_conditioned, b.joint_schema)
            if int(status.item()):
                raise RuntimeError('Nonfinite deployment Selector logits')
            selected = rows[b.select(scores, b.eligible[rows], k)]
            if compensation:
                weights, ratio_code = self._controls(ratio)
                flat, residual, status = self._ext.compensate(b.offset, selected, b.owners, b.reference,
                    b.tables, self._compensator, b.compensated_master, code, self._view_weight,
                    weights, ratio_code, view_conditioned and bool(b.joint_schema), b.joint_schema,
                    hasattr(b.spatial, 'softplus_scale_schema'))
                if int(status.item()):
                    raise RuntimeError('Nonfinite deployment compensated attributes')
            else:
                flat = b.ext.publish_selected(b.base_master, selected, b.empty, b.empty, True)
        else:
            selected = rows[:0]
            flat = b.ext.publish_selected(b.base_master, selected, b.empty, b.empty, True)
        data = self._packet(flat, k)
        stats = dict(total_gaussians=self.total, visible_anchors=visible_count, candidate_gaussians=n,
            eligible_candidates=available, requested_budget=int(budget), effective_budget=k,
            active_gaussians=k, ratio=ratio, budget_shortfall=max(0,int(budget)-k),
            inference_backend=self.backend, prepare_ms=(time.perf_counter()-started)*1000,
            cache_hit=False, view_conditioned=view_conditioned, compensation_enabled=compensation)
        return data, selected, stats

    def full(self):
        if torch.is_grad_enabled():
            raise RuntimeError('Deployment full rendering requires inference_mode')
        return self._packet(self._full.clone(), self.total), self._all_rows.clone()

    def render(self, camera, background=None, **request):
        from renderer import render_prepared
        data, ids, stats = self.prepare(camera, **request)
        if background is None:
            background = self._background
        result = render_prepared(camera, data, background, return_radii=True)
        result.update(ids=ids, stats=stats)
        return result

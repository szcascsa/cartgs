"""训练结束后的独立冻结部署后端；不被训练或常规评价入口导入。"""
import copy
import math
import numbers
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def deployment_extension():
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent
    return load(name='mycartgs_tensor_deployment_v1',
        sources=[str(root/'deployment_runtime_bindings.cpp'), str(root/'deployment_runtime_kernels.cu'),
                 str(root/'deployment_prepare.cpp'), str(root/'deployment_selection.cu'),str(root/'deployment_radix.cu'),str(root/'deployment_candidates.cu'),
                 str(root/'frozen_inference_kernels.cu')],
        extra_cflags=['-O3'], extra_cuda_cflags=['-O3','-lineinfo'], build_directory=str(root.parent/'build'), verbose=True)


class DeploymentRenderer:
    """独立模型快照，无训练梯度；每帧重算视锥、view、ratio和两网络。

    FP32 FMA顺序允许与cuBLAS不同。输出有独立存储，同一实例由调用者串行使用。
    切换checkpoint或地图时构造新实例，不允许在该实例上训练／更新拓扑。
    """
    @torch.inference_mode(False)
    @torch.no_grad()
    def __init__(self, model, gaussians, config, seed=0, *, shrinking_radix=True, compensation_tile=16, selector_tiled=True,
                 saturated_selection=True,prepare_graph=True,unroll_compensation=True,fused_selection=True,fused_candidates=True,
                 selector_static_prefix=True,visible_view=True,small_compensation_tile=True,student=None,tensor_selector=1,tensor_compensator=1,tensor_warps=4):
        from anchor_budget import FixedBudgetRenderer
        if (not isinstance(shrinking_radix,bool) or not isinstance(selector_tiled,bool) or
                isinstance(compensation_tile,bool) or compensation_tile not in (0,4,8,16)):
            raise ValueError('Use bool Radix/Selector switches and Compensation tile 0, 4, 8 or 16')
        if any(not isinstance(value,bool) for value in [saturated_selection,prepare_graph,unroll_compensation,fused_selection,fused_candidates,selector_static_prefix,visible_view,small_compensation_tile]):
            raise ValueError('Deployment optimization switches require bool values')
        # A private frozen snapshot prevents a training update from invalidating deployment data.
        model, gaussians = copy.deepcopy(model), copy.deepcopy(gaussians)
        model.eval().requires_grad_(False)
        for name in ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest'):
            getattr(gaussians, name).requires_grad_(False)
        self.student = None
        if student is not None:
            from compensation_student import StudentCompensator, teacher_fingerprint
            if not isinstance(student, StudentCompensator):
                raise ValueError('Deployment student must be a StudentCompensator')
            if student.teacher_fingerprint != teacher_fingerprint(model, gaussians):
                raise ValueError('Student belongs to a different frozen teacher or Gaussian map')
            joint = bool(int(getattr(model, 'joint_view_conditioning_schema', 0)))
            if hasattr(student, 'view_projection') != joint:
                raise ValueError('Student and teacher view schemas must match')
            self.student = copy.deepcopy(student).to(device=gaussians.get_xyz.device, dtype=torch.float32)
            self.student.eval().requires_grad_(False)
        self._reference = FixedBudgetRenderer(model, gaussians, copy.deepcopy(config), seed, 'cuda')
        self._base = self._reference.inference
        self._ext = deployment_extension()
        self._selector = [p.t().contiguous() if p.dim()==2 else p.contiguous()
                          for p in self._base.selector_parameters]
        self._compensator = [p.t().contiguous() if p.dim()==2 else p.contiguous()
                             for p in self._base.compensation_parameters]
        self._view_weight = (self._base.view_weight.t().contiguous()
                             if self._base.view_weight is not None else self._base.empty)
        if self.student is not None:
            self._compensator = [p.t().contiguous() if p.dim()==2 else p.contiguous()
                                 for p in self.student.deployment_parameters()]
            self._view_weight = (self.student.view_projection.weight.t().contiguous()
                                 if hasattr(self.student, 'view_projection') else self._base.empty)
        snapshot = (self._base.base_master + self._base.tables + self._selector + self._compensator +
                    [self._view_weight, self._base.selector_static, self._base.offset])
        if any(not torch.isfinite(value).all() for value in snapshot):
            raise ValueError('Nonfinite frozen deployment snapshot')
        self.total = self._reference.total
        self.knots = self._reference.knots
        self.backend = 'deployment_cpp'
        self._all_rows = torch.arange(self.total, device=self._base.owners.device)
        self._background = self._base.pc.get_xyz.new_zeros(3)
        self._full = self._base.ext.publish_selected(self._base.base_master, self._all_rows,
                                                    self._base.empty, self._base.empty, True)
        b=self._base
        state=[b.anchors.centers,b.anchors.offset_scales,self._reference.occupied,b.owners,
            b.eligible_counts,b.eligible,b.selector_static,b.offset,b.reference,self._reference.noise,
            b.view_centers4 if b.joint_schema else b.empty,b.view_corners4 if b.joint_schema else b.empty,
            b.model.selector_view_scene_scale if b.joint_schema else b.empty]
        self._pipeline=self._ext.Pipeline(state,self._selector,self._compensator,b.tables,
            b.compensated_master,b.base_master,self._view_weight,list(self.knots),list(b.spatial.ratio_encoding_values),
            b.joint_schema,b.pc.active_sh_degree,hasattr(b.spatial,'softplus_scale_schema'))
        self._pipeline._tuning(shrinking_radix,compensation_tile,selector_tiled)
        self._pipeline._optimization_controls(saturated_selection)
        self._pipeline._graph_enabled(prepare_graph)
        self._pipeline._kernel_options(unroll_compensation,fused_selection,fused_candidates)
        self._pipeline._static_options(selector_static_prefix,visible_view,small_compensation_tile)
        self._pipeline._tensor_options(tensor_selector,tensor_compensator,tensor_warps)

    @classmethod
    @torch.inference_mode(False)
    @torch.no_grad()
    def from_checkpoint(cls, checkpoint, seed=0, *, shrinking_radix=True, compensation_tile=16, selector_tiled=True,
                        saturated_selection=True,prepare_graph=True,unroll_compensation=True,fused_selection=True,fused_candidates=True,
                        selector_static_prefix=True,visible_view=True,small_compensation_tile=True,student_checkpoint=None,tensor_selector=1,tensor_compensator=1,tensor_warps=4):
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
        student = None
        if student_checkpoint is not None:
            from compensation_student import load_student
            student, _ = load_student(student_checkpoint)
        return cls(model,gaussians,config,seed,shrinking_radix=shrinking_radix,
            compensation_tile=compensation_tile,selector_tiled=selector_tiled,
            saturated_selection=saturated_selection,prepare_graph=prepare_graph,unroll_compensation=unroll_compensation,
            fused_selection=fused_selection,fused_candidates=fused_candidates,
            selector_static_prefix=selector_static_prefix,visible_view=visible_view,small_compensation_tile=small_compensation_tile,
            student=student,tensor_selector=tensor_selector,tensor_compensator=tensor_compensator,tensor_warps=tensor_warps)

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
        return self._pipeline.prepare(camera.world_view_transform,camera.full_proj_transform,camera.camera_center,
            int(camera.image_width),int(camera.image_height),float(camera.znear),int(budget or 0),
            float(ratio_budget or 0),ratio_budget is not None,view_conditioned,compensation)

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

"""默认优化推理：冻结Student＋Selector Tensor Core＋Mode182渲染器。

当前CPU相机、候选、严格K、Student及渲染在同一C++管线执行。
不导入训练入口，不缓存旧视角预测；同一实例只在一个device/Stream串行使用。
"""
import math
import numbers

import numpy as np
import torch


DEFAULT_RENDER_MODE = 182
NETWORK_OPTIONS = dict(tensor_selector=2, tensor_compensator=11, tensor_compensator_warps=2,
    preparation_frustum=1, preparation_projection=1, device_schedule=1)


def _request(budget, ratio_budget, full):
    if not isinstance(full, bool):
        raise ValueError('full must be boolean')
    if full:
        if budget is not None or ratio_budget is not None:
            raise ValueError('Full rendering does not take a point or ratio budget')
        return 0, 0., False
    if sum(value is not None for value in (budget, ratio_budget)) != 1:
        raise ValueError('Use one point budget or local ratio budget')
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, numbers.Integral) or budget < 0):
        raise ValueError('Point budget must be a nonnegative integer')
    if ratio_budget is not None and (isinstance(ratio_budget, bool) or not isinstance(ratio_budget, numbers.Real)
            or not math.isfinite(ratio_budget) or not 0 <= ratio_budget <= 1):
        raise ValueError('Ratio must be finite and in [0,1]')
    return int(budget or 0), float(ratio_budget or 0), ratio_budget is not None


class JointReleaseRenderer:
    """保守版new_prepare_182；相对原182逐位一致，相对原CaRtGS/F2仍有近似误差。"""

    @torch.inference_mode(False)
    @torch.no_grad()
    def __init__(self, model, gaussians, config, seed=0, *, student=None, mode=DEFAULT_RENDER_MODE):
        self._validate_model_request(student, mode)
        from joint_release_native._network import DeploymentRenderer
        self._initialize(DeploymentRenderer(model, gaussians, config, seed, student=student, **NETWORK_OPTIONS))

    @staticmethod
    def _validate_model_request(student, mode):
        if isinstance(mode, bool) or not isinstance(mode, numbers.Integral) or mode != DEFAULT_RENDER_MODE:
            raise ValueError('The default optimized renderer supports mode=182 only; Alpha3 is not enabled')
        if student is None:
            raise ValueError('Joint182 requires a frozen Student bound to this checkpoint; pass student/--student')

    def _initialize(self, network):
        from joint_release_native._extension import extension
        if network._base.joint_schema not in (1, 2, 3, 4):
            raise ValueError('Joint182 requires a view-conditioned Residual48 checkpoint and matching Student')
        self.network = network
        self.native = extension().PendingRasterizer(512, True, True, 15, 200, 64)
        self.handle = self.native._shared_handle()
        self.total, self.knots = network.total, network.knots
        self.mode, self.backend = DEFAULT_RENDER_MODE, 'joint_release_182'
        self.rest = network._base.base_master[5].numel() // max(1, self.total)
        self.degree = network._base.pc.active_sh_degree
        self._calibration = None
        network._pipeline._small_radix_finish(True)
        network._pipeline._direct_pose(True)
        if not bool(torch.isfinite(network._full).all()):
            raise ValueError('Nonfinite frozen Full attributes')

    @classmethod
    @torch.inference_mode(False)
    @torch.no_grad()
    def from_checkpoint(cls, checkpoint, seed=0, *, student_checkpoint=None, mode=DEFAULT_RENDER_MODE):
        cls._validate_model_request(student_checkpoint, mode)
        from joint_release_native._network import DeploymentRenderer
        network = DeploymentRenderer.from_checkpoint(checkpoint, seed,
            student_checkpoint=student_checkpoint, **NETWORK_OPTIONS)
        result = cls.__new__(cls)
        result._initialize(network)
        return result

    def _background(self, background):
        if torch.is_grad_enabled():
            raise RuntimeError('JointReleaseRenderer is frozen inference only; use torch.no_grad()/inference_mode()')
        if background is None:
            background = self.network._background
        if (background.shape != (3,) or background.device != self.network._background.device
                or background.dtype != torch.float32 or not background.is_contiguous()):
            raise ValueError('Background must be contiguous CUDA FP32 [3] on the model device')
        return background

    def _result(self, values, full):
        image, radii, ids, stats = values
        stats.update(inference_backend=self.backend, rasterizer_mode=self.mode, total_gaussians=self.total,
            compensation_model='frozen_student32', selection='full' if full else 'learned', cache_hit=False,
            budget_shortfall=0 if full else max(0, stats['requested_budget'] - len(ids)))
        return dict(render=image, radii=radii, ids=ids, stats=stats)

    def precompile(self, width, height, background=None):
        """可选启动预热；与当前帧延迟分开计时，容量超限由native明确拒绝。"""
        background = self._background(background)
        self.native.precompile(max(1, self.total), width, height, 5, 256, 32, background, False)

    def render_pose(self, pose, calibration, background=None, *, budget=None, ratio_budget=None, full=False):
        """pose为CPU相机到世界4×4矩阵；局部N取当前名义可见Anchor成员，严格K。"""
        budget, ratio, use_ratio = _request(budget, ratio_budget, full)
        background = self._background(background)
        pose = np.asarray(pose, dtype=np.float64, order='C')
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('Current camera pose must be a finite CPU 4x4 matrix')
        width, height = calibration['width'], calibration['height']
        fx, fy = calibration['fx'], calibration['fy']
        if any(isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0 for value in [width, height]):
            raise ValueError('Camera dimensions must be positive integers')
        if any(isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not math.isfinite(value) or value <= 0 for value in [fx, fy]):
            raise ValueError('Camera focal lengths must be finite and positive')
        fovx, fovy = 2 * math.atan(width / (2 * fx)), 2 * math.atan(height / (2 * fy))
        key = (width, height, fovx, fovy, torch.cuda.current_stream().cuda_stream, torch.get_default_dtype())
        if key != self._calibration:
            from utils import getProjectionMatrix
            self._projection = getProjectionMatrix(znear=.01, zfar=100., fovX=fovx, fovY=fovy).transpose(0, 1).contiguous()
            self._device_reference = self._projection.to(self.network._background.device, non_blocking=True)
            self._calibration = key
        if full:
            values = self.network._pipeline._render_full_pose(self.handle, background, pose, self._projection,
                self.network._full, self.network._all_rows, width, height, self.rest,
                math.tan(fovx * .5), math.tan(fovy * .5))
        else:
            values = self.network._pipeline._render_pose_reused(self.handle, background, pose, self._projection,
                self._device_reference, width, height, budget, ratio, use_ratio, self.rest,
                math.tan(fovx * .5), math.tan(fovy * .5))
        return self._result(values, full)

    def render(self, camera, background=None, *, budget=None, ratio_budget=None, full=False):
        """兼容已有GPU相机；需要最低提交开销时使用render_pose。"""
        budget, ratio, use_ratio = _request(budget, ratio_budget, full)
        background = self._background(background)
        if full:
            packet, ids = self.network.full()
            data, empty = packet['inputs'], self.network._base.empty
            image, radii = self.native.render(background, data['means3D'], empty, data['opacities'], data['scales'],
                data['rotations'], 1., empty, camera.world_view_transform, camera.full_proj_transform,
                math.tan(camera.FoVx * .5), math.tan(camera.FoVy * .5), camera.image_height, camera.image_width,
                data['dc'], data['shs'], self.degree, camera.camera_center, 5, 256, 32, False)
            values = image, radii, ids, dict(candidate_gaussians=self.total, active_gaussians=self.total)
        else:
            values = self.network._pipeline._render_pending_current(self.handle, background,
                camera.world_view_transform, camera.full_proj_transform, camera.camera_center,
                camera.image_width, camera.image_height, float(camera.znear), budget, ratio, use_ratio, True, True,
                self.rest, math.tan(camera.FoVx * .5), math.tan(camera.FoVy * .5))
        return self._result(values, full)

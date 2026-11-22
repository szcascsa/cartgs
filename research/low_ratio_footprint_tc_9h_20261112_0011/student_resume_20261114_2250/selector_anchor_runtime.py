"""实际K控制成员处理候选；与冻结原路径逐位比较。"""
from pathlib import Path
from functools import lru_cache
import importlib.util
import os
import math
import cProfile
import torch
from student_runtime import StudentRenderer

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def extension():
    name = 'mycartgs_student_selector_anchor_20261114_v1'
    root, build = HERE / 'source_selector_anchor', HERE / 'build_selector_anchor'
    binary = build / (name + '.so')
    if binary.exists() and os.environ.get('SELECTOR_ANCHOR_REBUILD') != '1':
        spec = importlib.util.spec_from_file_location(name, binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    os.environ['CUDA_HOME'] = '/usr/local/cuda-11.8'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    os.environ['MAX_JOBS'] = '2'
    os.environ['PATH'] = '/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:' + os.environ.get('PATH', '')
    from torch.utils.cpp_extension import load
    files = ['deployment_runtime_bindings.cpp', 'deployment_runtime_kernels.cu', 'deployment_prepare.cpp', 'raster/graph_main.cu']
    unchanged = ['deployment_radix.cu', 'deployment_selection.cu', 'deployment_candidates.cu', 'frozen_inference_kernels.cu',
                 'preparation_kernels.cu', 'preparation_camera.cpp', 'raster/graph_forward.cu']
    reused = []
    for file in unchanged:
        assert (root / file).read_bytes() == (HERE / 'source_joint_final' / file).read_bytes(), file
        path = HERE / 'build_joint_final' / (Path(file).stem + ('.cuda.o' if file.endswith('.cu') else '.o'))
        assert path.is_file(), path
        reused.append(str(path))
    return load(name=name, sources=[str(root / name) for name in files],
        extra_include_paths=[str(root / 'raster/reference/cuda_rasterizer')],
        extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-lineinfo'], extra_ldflags=reused,
        build_directory=str(build), verbose=True)


@lru_cache(maxsize=1)
def deployment_class():
    spec = importlib.util.spec_from_file_location('student_selector_anchor_wrapper', HERE / 'source_selector_anchor/deployment_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.deployment_extension = extension
    return module.DeploymentRenderer


class SelectorAnchorRenderer(StudentRenderer):
    def __init__(self, source, student, mode=182, reuse=True, fused=False, finish=False, members=15, projected=True):
        assert student is not None and student.teacher_fingerprint == source.fingerprint
        self.reuse = reuse
        self.native_dispatch = True
        self.network = deployment_class()(source.scene.model, source.scene.gaussians, source.scene.config,
            student=student, tensor_selector=2, tensor_compensator=(13 if projected else 9) if fused else 2,
            tensor_compensator_warps=2, preparation_frustum=1,
            preparation_projection=1, device_schedule=1)
        self.native = extension().PendingRasterizer(512, True, True, members, 18 + mode, 64)
        self.handle = self.native._shared_handle()
        self.rest = self.network._base.base_master[5].numel() // max(1, self.network.total)
        self.degree = source.scene.gaussians.active_sh_degree
        self._calibration = None
        self.network._pipeline._small_radix_finish(finish)

    @torch.no_grad()
    def render_from_source(self, source, frame, bg, condition, factory):
        if not self.reuse:
            return super().render_from_source(source, frame, bg, condition, factory)
        if condition == 'full':
            return self.render(factory.from_source(source, frame), bg, condition)
        c = source.scene.calibration
        width, height = c['width'], c['height']
        fovx, fovy = 2 * math.atan(width / (2 * c['fx'])), 2 * math.atan(height / (2 * c['fy']))
        key = (width, height, fovx, fovy, torch.cuda.current_stream().cuda_stream, torch.get_default_dtype())
        if key != self._calibration:
            self._projection = factory.projection_builder(znear=.01, zfar=100., fovX=fovx, fovY=fovy).transpose(0, 1).contiguous()
            self._device_reference = self._projection.cuda(non_blocking=True)
            self._calibration = key
        use_ratio = condition.startswith('local')
        ratio = float(condition[5:]) if use_ratio else 0.
        budget = 0 if use_ratio else int(condition[5:])
        return self.network._pipeline._render_pose_reused(self.handle, bg, source.scene.views[frame]['pose'],
            self._projection, self._device_reference, width, height, budget, ratio, use_ratio,
            self.rest, math.tan(fovx * .5), math.tan(fovy * .5))


if __name__ == '__main__':
    print('STUDENT_SELECTOR_ANCHOR_BUILD_PASS', extension().__file__, flush=True)

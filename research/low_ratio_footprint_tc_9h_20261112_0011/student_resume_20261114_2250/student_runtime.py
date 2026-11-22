"""冻结Student专用候选，调用返回前执行原finite检查。"""
from pathlib import Path
from functools import lru_cache
import importlib.util
import os
import sys
import cProfile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pose_runtime import PoseRenderer

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def extension():
    name = 'mycartgs_student_schedule_20261114_v1'
    root, build = HERE / 'source', HERE / 'build'
    binary = build / (name + '.so')
    if binary.exists() and os.environ.get('STUDENT_REBUILD') != '1':
        spec = importlib.util.spec_from_file_location(name, binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    os.environ['CUDA_HOME'] = '/usr/local/cuda-11.8'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    os.environ['MAX_JOBS'] = '2'
    os.environ['PATH'] = '/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:' + os.environ.get('PATH', '')
    from torch.utils.cpp_extension import load
    files = ['deployment_runtime_bindings.cpp', 'deployment_runtime_kernels.cu', 'deployment_prepare.cpp',
             'deployment_selection.cu', 'deployment_radix.cu', 'deployment_candidates.cu',
             'frozen_inference_kernels.cu', 'preparation_kernels.cu', 'preparation_camera.cpp',
             'raster/graph_main.cu', 'raster/graph_forward.cu']
    return load(name=name, sources=[str(root / name) for name in files],
        extra_include_paths=[str(root / 'raster/reference/cuda_rasterizer')],
        extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-lineinfo'],
        build_directory=str(build), verbose=True)


@lru_cache(maxsize=1)
def deployment_class():
    spec = importlib.util.spec_from_file_location('student_schedule_wrapper', HERE / 'source/deployment_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.deployment_extension = extension
    return module.DeploymentRenderer


class StudentRenderer(PoseRenderer):
    def __init__(self, source, student, mode=38, schedule=3, warps=2, projected=True):
        assert student is not None and student.teacher_fingerprint == source.fingerprint
        assert schedule in [2, 3, 4, 5, 6]
        self.native_dispatch = True
        self.network = deployment_class()(source.scene.model, source.scene.gaussians, source.scene.config,
            student=student, tensor_selector=2, tensor_compensator=schedule,
            tensor_compensator_warps=warps, preparation_frustum=1,
            preparation_projection=int(projected), device_schedule=1)
        self.native = extension().PendingRasterizer(512, True, True, 12, 18 + mode, 64)
        self.handle = self.native._shared_handle()
        self.rest = self.network._base.base_master[5].numel() // max(1, self.network.total)
        self.degree = source.scene.gaussians.active_sh_degree
        self._calibration = None


if __name__ == '__main__':
    print('STUDENT_SCHEDULE_BUILD_PASS', extension().__file__, flush=True)

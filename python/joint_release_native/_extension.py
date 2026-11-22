"""从随项目提交的源码独立构建，不依赖研究目录、历史对象或共享环境改动。"""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent
    sources = ['deployment_runtime_bindings.cpp', 'deployment_runtime_kernels.cu',
        'deployment_prepare.cpp', 'deployment_selection.cu', 'deployment_radix.cu',
        'deployment_candidates.cu', 'frozen_inference_kernels.cu', 'preparation_kernels.cu',
        'preparation_camera.cpp', 'raster/graph_main.cu', 'raster/graph_forward.cu']
    return load(name='mycartgs_joint_release_182_v1', sources=[str(root / name) for name in sources],
        extra_include_paths=[str(root / 'raster/reference/cuda_rasterizer')],
        extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-lineinfo'], verbose=False)

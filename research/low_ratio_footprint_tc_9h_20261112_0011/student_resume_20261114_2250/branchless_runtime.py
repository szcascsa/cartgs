"""有序Tile中心尾部近似，纯渲染；独立开关并保留冻结F2对照。"""
from pathlib import Path
from functools import lru_cache
import importlib.util
import os
import cProfile
import torch
from student_study import Renderer

HERE = Path(__file__).resolve().parent
MODES={38:'old38',141:'merged_guard',142:'predicated_min',143:'no_min',144:'branchless_no_min'}


@lru_cache(maxsize=1)
def extension():
    name = 'mycartgs_student_branchless_20261114_v1'
    root, build = HERE/'source_branchless', HERE/'build_branchless'
    binary = build/(name+'.so')
    if binary.exists() and os.environ.get('BRANCHLESS_REBUILD') != '1':
        spec = importlib.util.spec_from_file_location(name, binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    os.environ['CUDA_HOME']='/usr/local/cuda-11.8'
    os.environ['TORCH_CUDA_ARCH_LIST']='8.6'
    os.environ['MAX_JOBS']='2'
    os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
    from torch.utils.cpp_extension import load
    return load(name=name, sources=[str(root/'graph_main.cu'),str(root/'graph_forward.cu')],
        extra_include_paths=[str(root/'reference/cuda_rasterizer')], extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3','-lineinfo'],build_directory=str(build),verbose=True)


class BranchlessRenderer(Renderer):
    def __init__(self, mode=142):
        assert mode in MODES
        self.mode = mode
        self.native = extension().FootprintRasterizer(512, True, True, 10, 18+mode, 64)


if __name__ == '__main__':
    print('STUDENT_BRANCHLESS_BUILD_PASS', extension().__file__, flush=True)

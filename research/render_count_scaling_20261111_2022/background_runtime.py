"""消除普通背景下单Warp全图空成员扫描，仅用于冻结推理。"""
from pathlib import Path
from functools import lru_cache
import importlib.util
import os
import scaling_runtime as first

HERE=Path(__file__).resolve().parent/'background_stage'


@lru_cache(maxsize=1)
def extension():
    name='mycartgs_render_background_scaling_v1'
    binary=HERE/'build'/(name+'.so')
    if binary.is_file() and os.environ.get('BACKGROUND_REBUILD')!='1':
        spec=importlib.util.spec_from_file_location(name,binary)
        native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
        return native
    os.environ['CUDA_HOME']='/usr/local/cuda-11.8'
    os.environ['TORCH_CUDA_ARCH_LIST']='8.6'
    os.environ['MAX_JOBS']='2'
    os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
    from torch.utils.cpp_extension import load
    return load(name=name,sources=[str(HERE/'source/graph_forward.cu'),str(HERE/'source/graph_main.cu')],
        extra_include_paths=[str(HERE/'source/reference/cuda_rasterizer')],extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3','-lineinfo'],build_directory=str(HERE/'build'),verbose=True)


class Renderer(first.Renderer):
    def __init__(self,mode=18):
        assert mode in (0,18)
        self.mode=18
        self.native=extension().FootprintRasterizer(512,True,True,10,18 if mode==18 else 14,64)


if __name__=='__main__':print('BACKGROUND_BUILD_PASS',extension().__file__,flush=True)

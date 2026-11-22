"""研究专用冻结渲染器；训练和生产入口不导入此模块。"""
from pathlib import Path
import math
import os
import torch
from functools import lru_cache

HERE=Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def extension():
    os.environ.setdefault('CUDA_HOME','/usr/local/cuda-11.8')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.6')
    os.environ.setdefault('MAX_JOBS','2')
    os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
    from torch.utils.cpp_extension import load
    build=HERE/'build';build.mkdir(exist_ok=True)
    return load(name='sparse_rasterizer_v1',sources=[str(HERE/'source/forward_sparse.cu'),str(HERE/'source/sparse_inference.cu')],
        extra_include_paths=[str(HERE/'source/reference/cuda_rasterizer')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo'],build_directory=str(build),verbose=True)


class SparseRenderer:
    def __init__(self,mode=1,short_lists=False):
        self.native=extension().Rasterizer()
        self.mode=mode
        self.short_lists=short_lists

    @torch.no_grad()
    def render(self,camera,packet,background,inspect=False):
        p=packet['inputs'];empty=p['means3D'].new_empty(0)
        def tensor(name):return p[name] if p[name] is not None else empty
        args=[background,p['means3D'],p['colors_precomp'] if p['colors_precomp'] is not None else empty,
            p['opacities'],tensor('scales'),tensor('rotations'),1.,p['cov3D_precomp'] if p['cov3D_precomp'] is not None else empty,
            camera.world_view_transform,camera.full_proj_transform,math.tan(camera.FoVx*.5),math.tan(camera.FoVy*.5),
            camera.image_height,camera.image_width,tensor('dc'),tensor('shs'),packet['sh_degree'],camera.camera_center,False,False,
            self.mode,self.short_lists,inspect]
        return self.native.render(*args)


if __name__=='__main__':extension()

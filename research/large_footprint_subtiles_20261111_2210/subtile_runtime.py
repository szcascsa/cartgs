"""独立冻结推理候选；只缓存有界调度计划，每帧读当前输入。"""
from pathlib import Path
from functools import lru_cache
import importlib.util
import math
import os
import torch

HERE=Path(__file__).resolve().parent
MODES={0:'F2_control',1:'subtile8_compact',2:'subtile4_compact',3:'subtile8_mask',4:'exact8_diagnostic'}
MODES.update({5:'alpha2_pixel2',6:'alpha4_pixel2',7:'alpha8_pixel2',8:'alpha2_pixel4',9:'alpha4_pixel4',10:'alpha2_pixel1'})
MODES.update({11:'macro2x1_unroll8',12:'macro2x1_unroll32',13:'macro2x2_unroll8',14:'macro1x2_unroll8'})
MODES.update({15:'per_thread_exit',16:'warp_exit2',17:'warp_exit4',18:'warp_exit8',19:'warp_exit16'})
MODES.update({20:'direct_tile128',21:'direct_tile64'})
MODES.update({22:'short_cache32',23:'short_cache_adjacent_x',24:'short_cache_adjacent_y',25:'short_cache_pixel1',26:'short_cache_unroll8',27:'short_cache_unroll16'})
MODES.update({28:'tile_cache32',29:'tile_cache64',30:'tile_cache128',31:'tile_cache64_x',32:'tile_cache128_x',33:'tile_cache64_pixel1'})
MODES.update({34:'shared_alpha4',35:'shared_alpha8',36:'shared_alpha16',37:'transpose_alpha4',38:'transpose_alpha8',39:'transpose_alpha16'})
MODES.update({40:'occupancy12',41:'occupancy16',42:'occupancy12_exit8',43:'occupancy16_exit8'})
MODES.update({44:'compact_pixel8',45:'compact_pixel32',46:'compact_pixel16'})
MODES.update({47:'exact_common8',48:'exact_common16',49:'exact_common32',50:'exact_common32_exit8'})


@lru_cache(maxsize=1)
def extension():
    name='mycartgs_large_footprint_subtiles_v1'
    binary=HERE/'build'/(name+'.so')
    if binary.is_file() and os.environ.get('SUBTILE_REBUILD')!='1':
        spec=importlib.util.spec_from_file_location(name,binary);native=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native);return native
    os.environ['CUDA_HOME']='/usr/local/cuda-11.8';os.environ['TORCH_CUDA_ARCH_LIST']='8.6'
    os.environ['MAX_JOBS']='2'
    os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
    from torch.utils.cpp_extension import load
    return load(name=name,sources=[str(HERE/'source/graph_forward.cu'),str(HERE/'source/graph_main.cu')],
        extra_include_paths=[str(HERE/'source/reference/cuda_rasterizer')],extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3','-lineinfo'],build_directory=str(HERE/'build'),verbose=True)


class Renderer:
    def __init__(self,mode=1):
        assert mode in MODES
        self.mode=mode;self.native=extension().FootprintRasterizer(512,True,True,10,18+mode,64)

    def precompile(self,n,camera,background):
        self.native.precompile(n,camera.image_width,camera.image_height,5,256,32,background,False)

    @torch.no_grad()
    def render(self,camera,packet,background):
        x=packet['inputs'];empty=x['means3D'].new_empty(0)
        if not len(x['means3D']):
            return [background[:,None,None].expand(-1,camera.image_height,camera.image_width).clone(),
                    torch.zeros(0,device=empty.device,dtype=torch.int32)]
        def tensor(k):return x[k] if x[k] is not None else empty
        return self.native.render(background,x['means3D'],tensor('colors_precomp'),x['opacities'],
            tensor('scales'),tensor('rotations'),1.,tensor('cov3D_precomp'),
            camera.world_view_transform,camera.full_proj_transform,math.tan(camera.FoVx*.5),math.tan(camera.FoVy*.5),
            camera.image_height,camera.image_width,tensor('dc'),tensor('shs'),packet['sh_degree'],camera.camera_center,
            5,256,32,False)


if __name__=='__main__':print('SUBTILE_BUILD_PASS',extension().__file__,flush=True)

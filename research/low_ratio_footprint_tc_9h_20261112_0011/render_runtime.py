"""仅本轮研究加载；实例有界Workspace、device/Stream归属沿用F2。"""
from pathlib import Path
from functools import lru_cache
import importlib.util
import math
import os
import torch

HERE=Path(__file__).resolve().parent
MODES={0:'rebuilt_F2',1:'fast_exp',2:'stop_3e4',3:'stop_1e3',4:'stop_3e3',5:'stop_1e2',
       6:'stop_3e2',7:'stop_1e1',8:'alpha_2',9:'alpha_4',10:'alpha_8',
       11:'weight_1e4',12:'weight_5e4',13:'weight_1e3',14:'stop_1e2_precise_exp'}
MODES.update({15: 'folded_fast', 16: 'folded_stop_1e2', 17: 'folded_stop_2e2', 18: 'folded_stop_3e3', 19: 'folded_bitexp', 20: 'folded_bitexp_stop', 21: 'folded_polyexp', 22: 'folded_polyexp_stop', 23: 'fast_exp_check32', 24: 'stop_1e2_check32', 25: 'stop_2e2', 26: 'direct_bitexp', 27: 'direct_polyexp'})
MODES.update({28: 'folded_bitexp_safe', 29: 'folded_bitexp_stop_safe', 30: 'folded_bitexp_weight', 31: 'folded_bitexp_weight_stop', 32: 'folded_fast_weight'})
MODES.update({33: 'tensor_half', 34: 'tensor_half_stop', 35: 'tensor_tf32', 36: 'tensor_tf32_stop'})
MODES.update({37: 'tile_safe_bitexp', 38: 'tile_safe_bitexp_stop', 39: 'tile_safe_bitexp_stop2', 40: 'tile_safe_fast'})


@lru_cache(maxsize=1)
def extension():
    name='mycartgs_footprint_ninehour_v1';binary=HERE/'build'/(name+'.so')
    if binary.exists() and os.environ.get('NINE_REBUILD')!='1':
        spec=importlib.util.spec_from_file_location(name,binary);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
    os.environ['CUDA_HOME']='/usr/local/cuda-11.8';os.environ['TORCH_CUDA_ARCH_LIST']='8.6';os.environ['MAX_JOBS']='2'
    os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
    from torch.utils.cpp_extension import load
    return load(name=name,sources=[str(HERE/'source/graph_forward.cu'),str(HERE/'source/graph_main.cu')],
        extra_include_paths=[str(HERE/'source/reference/cuda_rasterizer')],extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3','-lineinfo'],build_directory=str(HERE/'build'),verbose=True)


class Renderer:
    def __init__(self,mode=0):
        assert mode in MODES
        self.mode=mode;self.native=extension().FootprintRasterizer(512,True,True,10,18+mode,64)

    def precompile(self,n,camera,background):
        self.native.precompile(max(1,n),camera.image_width,camera.image_height,5,256,32,background,False)

    @torch.no_grad()
    def render(self,camera,packet,background):
        x=packet['inputs'];empty=x['means3D'].new_empty(0)
        if not len(x['means3D']):return [background[:,None,None].expand(-1,camera.image_height,camera.image_width).clone(),torch.zeros(0,device=empty.device,dtype=torch.int32)]
        def tensor(k):return x[k] if x[k] is not None else empty
        return self.native.render(background,x['means3D'],tensor('colors_precomp'),x['opacities'],tensor('scales'),tensor('rotations'),1.,
            tensor('cov3D_precomp'),camera.world_view_transform,camera.full_proj_transform,math.tan(camera.FoVx*.5),math.tan(camera.FoVy*.5),
            camera.image_height,camera.image_width,tensor('dc'),tensor('shs'),packet['sh_degree'],camera.camera_center,5,256,32,False)


if __name__=='__main__':print('NINE_BUILD_PASS',extension().__file__,flush=True)

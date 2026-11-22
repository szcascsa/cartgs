"""只用于冻结推理的独立渲染候选；动态视角数据每次重读。"""
from pathlib import Path
from functools import lru_cache
import importlib.util
import math
import os
import torch

HERE=Path(__file__).resolve().parent
MODES={0:'control',1:'safe_exp_reject',2:'cached_log_reject',3:'column_common',4:'opacity_reject'}
MODES.update({5:'loop_unroll2',6:'loop_unroll4',7:'loop_unroll8',8:'small_plan32'})
MODES.update({9:'small_plan_unroll8',10:'loop_unroll16',11:'loop_unroll32',12:'small_plan_unroll16'})
MODES.update({13:'packed8',14:'packed16',15:'packed32',16:'small_plan_packed8',17:'small_plan_packed32'})
MODES[18]='small_plan_unroll32'


@lru_cache(maxsize=1)
def extension():
    build=HERE/'build'
    name='mycartgs_render_count_scaling_v1'
    binary=build/(name+'.so')
    if binary.is_file() and os.environ.get('SCALING_REBUILD')!='1':
        spec=importlib.util.spec_from_file_location(name,binary)
        native=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        return native
    os.environ['CUDA_HOME']='/usr/local/cuda-11.8'
    os.environ['TORCH_CUDA_ARCH_LIST']='8.6'
    os.environ['MAX_JOBS']='2'
    os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
    from torch.utils.cpp_extension import load
    return load(name=name,sources=[str(HERE/'source/graph_forward.cu'),str(HERE/'source/graph_main.cu')],
                extra_include_paths=[str(HERE/'source/reference/cuda_rasterizer')],extra_cflags=['-O3'],
                extra_cuda_cflags=['-O3','-lineinfo'],build_directory=str(build),verbose=True)


class Renderer:
    def __init__(self,mode=18):
        assert mode in MODES
        self.mode=mode
        pixel={8:0,9:12,10:13,11:14,12:13,13:15,14:16,15:17,16:15,17:17,18:14}.get(mode,mode+5 if mode else 0)
        self.native=extension().FootprintRasterizer(512,True,True,10,pixel,64)

    def precompile(self,points,camera,background):
        self.native.precompile(points,camera.image_width,camera.image_height,5,256,32 if self.mode in (8,9,12,16,17,18) else 512,background,False)

    @torch.no_grad()
    def render(self,camera,packet,background):
        p=packet['inputs']
        if not len(p['means3D']):
            return [background[:,None,None].expand(-1,camera.image_height,camera.image_width).clone(),
                    torch.zeros(0,device=p['means3D'].device,dtype=torch.int32)]
        empty=p['means3D'].new_empty(0)
        def tensor(name):
            return p[name] if p[name] is not None else empty
        return self.native.render(background,p['means3D'],tensor('colors_precomp'),p['opacities'],tensor('scales'),tensor('rotations'),
            1.,tensor('cov3D_precomp'),camera.world_view_transform,camera.full_proj_transform,math.tan(camera.FoVx*.5),math.tan(camera.FoVy*.5),
            camera.image_height,camera.image_width,tensor('dc'),tensor('shs'),packet['sh_degree'],camera.camera_center,5,256,32 if self.mode in (8,9,12,16,17,18) else 512,False)


if __name__=='__main__':
    print('BUILD_PASS',extension().__file__,flush=True)

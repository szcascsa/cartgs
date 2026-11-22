"""本轮有界Graph成员构建路径，模块名与上一轮runtime隔离。"""
from pathlib import Path
import math,os,torch,importlib.util
from functools import lru_cache

HERE=Path(__file__).resolve().parent
MODES={0:'control',1:'gaussian128',2:'gaussian64',3:'gaussian256',4:'gaussian2blocks',5:'gaussian4blocks',6:'tile_words',7:'tile_compacted'}
MODES.update({8:'tile_compacted_dynamic',9:'tile_words_dynamic',10:'pixel_warp8',11:'pixel_warp4',12:'pixel_warp2',13:'tile_dynamic_warp8',14:'gaussian2_warp8'})

MODES.update({15:'pixel_direct2',16:'pixel_direct1',17:'preprocess64',18:'preprocess128',19:'gaussian4_pre64',20:'gaussian4_pre64_direct2'})

MODES.update({21:'fused_tasks_warp64',22:'fused_tasks_block64',23:'fused_tasks_block128',24:'fused_tasks_warp128'})

@lru_cache(maxsize=1)
def extension():
    existing=HERE/'build_v4/large_footprint_graph_v4.so'
    if existing.exists() and os.environ.get('FOOTPRINT_REBUILD')!='1':
        spec=importlib.util.spec_from_file_location('large_footprint_graph_v4',existing)
        native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native);return native
    os.environ.setdefault('CUDA_HOME','/usr/local/cuda-11.8');os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.6');os.environ.setdefault('MAX_JOBS','2')
    os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
    from torch.utils.cpp_extension import load
    (HERE/'build_v4').mkdir(exist_ok=True)
    return load(name='large_footprint_graph_v4',sources=[str(HERE/'source/graph_forward.cu'),str(HERE/'source/graph_main.cu')],
        extra_include_paths=[str(HERE/'source/reference/cuda_rasterizer')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo'],
        build_directory=str(HERE/'build_v4'),verbose=True)


class Renderer:
    def __init__(self,mode=0,max_plans=512):
        self.mode=mode
        build_mode={10:0,11:0,12:0,13:8,14:4,15:0,16:0,17:0,18:0,19:5,20:5,21:10,22:11,23:11,24:10}.get(mode,mode)
        pixel_mode={10:1,11:2,12:3,13:1,14:1,15:4,16:5,20:4}.get(mode,0)
        self.native=extension().FootprintRasterizer(max_plans,True,True,build_mode,pixel_mode,{17:64,18:128,19:64,20:64,21:64,22:64,23:128,24:128}.get(mode,256))

    def precompile(self,points,camera,background):
        self.native.precompile(points,camera.image_width,camera.image_height,5,256,512,background,False)

    @torch.no_grad()
    def render(self,camera,packet,background):
        p=packet['inputs'];empty=p['means3D'].new_empty(0)
        if not len(p['means3D']):
            return [background[:,None,None].expand(-1,camera.image_height,camera.image_width).clone(),
                torch.zeros(0,device=empty.device,dtype=torch.int32)]
        def tensor(name):return p[name] if p[name] is not None else empty
        return self.native.render(background,p['means3D'],tensor('colors_precomp'),p['opacities'],tensor('scales'),tensor('rotations'),
            1.,tensor('cov3D_precomp'),camera.world_view_transform,camera.full_proj_transform,math.tan(camera.FoVx*.5),math.tan(camera.FoVy*.5),
            camera.image_height,camera.image_width,tensor('dc'),tensor('shs'),packet['sh_degree'],camera.camera_center,5,256,512,False)


if __name__=='__main__':extension()

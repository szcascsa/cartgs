"""实验性容量图：缓存执行计划，每帧重新提交数据指针，不缓存视角内容。"""
from pathlib import Path
import math,os,torch
from functools import lru_cache

HERE=Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def extension():
    os.environ.setdefault('CUDA_HOME','/usr/local/cuda-11.8');os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.6');os.environ.setdefault('MAX_JOBS','2')
    os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
    from torch.utils.cpp_extension import load
    build=HERE/'build_graph';build.mkdir(exist_ok=True)
    return load(name='mycartgs_sparse_graph_study_v1',sources=[str(HERE/'source/graph_forward.cu'),str(HERE/'source/graph_main.cu')],
        extra_include_paths=[str(HERE/'source/reference/cuda_rasterizer')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo'],
        build_directory=str(build),verbose=True)


class GraphRenderer:
    def __init__(self,variant=5,capacity=256,plan_granularity=0,max_plans=64,graph_input_update=False,ordered_coverage=False,device_clear=False):
        self.native=extension().GraphRasterizer(max_plans,ordered_coverage,device_clear);self.variant=variant;self.capacity=capacity;self.plan_granularity=plan_granularity
        self.graph_input_update=graph_input_update

    def reserve(self,points,camera,exemplar):
        self.native.reserve(points,camera.image_width,camera.image_height,self.plan_granularity,exemplar)

    def precompile(self,points,camera,exemplar):
        self.native.precompile(points,camera.image_width,camera.image_height,self.variant,self.capacity,
            self.plan_granularity,exemplar,self.graph_input_update)

    @torch.no_grad()
    def render(self,camera,packet,background):
        p=packet['inputs'];empty=p['means3D'].new_empty(0)
        if not len(p['means3D']):
            return [background[:,None,None].expand(-1,camera.image_height,camera.image_width).clone(),
                torch.zeros(0,device=p['means3D'].device,dtype=torch.int32)]
        def tensor(name):return p[name] if p[name] is not None else empty
        return self.native.render(background,p['means3D'],tensor('colors_precomp'),p['opacities'],tensor('scales'),tensor('rotations'),
            1.,tensor('cov3D_precomp'),camera.world_view_transform,camera.full_proj_transform,math.tan(camera.FoVx*.5),math.tan(camera.FoVy*.5),
            camera.image_height,camera.image_width,tensor('dc'),tensor('shs'),packet['sh_degree'],camera.camera_center,self.variant,self.capacity,self.plan_granularity,self.graph_input_update)


if __name__=='__main__':extension()

"""新联合候选仅用于冻结推理；当前帧准备、渲染和finite检查形成一个请求。"""
from pathlib import Path
from functools import lru_cache
import importlib.util
import math
import os
import torch

HERE=Path(__file__).resolve().parent


@lru_cache(maxsize=2)
def extension(native=False):
    root=HERE/('joint_source_v2' if native else 'joint_source')
    name='mycartgs_pending_joint_ninehour_v2' if native else 'mycartgs_pending_joint_ninehour_v1'
    build=HERE/('joint_build_v2' if native else 'joint_build');binary=build/(name+'.so')
    if binary.exists() and os.environ.get('PENDING_REBUILD')!='1':
        spec=importlib.util.spec_from_file_location(name,binary);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
    os.environ['CUDA_HOME']='/usr/local/cuda-11.8';os.environ['TORCH_CUDA_ARCH_LIST']='8.6';os.environ['MAX_JOBS']='2'
    os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
    from torch.utils.cpp_extension import load
    files=['deployment_runtime_bindings.cpp','deployment_runtime_kernels.cu','deployment_prepare.cpp','deployment_selection.cu',
        'deployment_radix.cu','deployment_candidates.cu','frozen_inference_kernels.cu','preparation_kernels.cu','preparation_camera.cpp',
        'raster/graph_main.cu','raster/graph_forward.cu']
    return load(name=name,sources=[str(root/n) for n in files],extra_include_paths=[str(root/'raster/reference/cuda_rasterizer')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo'],build_directory=str(build),verbose=True)


@lru_cache(maxsize=2)
def deployment_class(native=False):
    root=HERE/('joint_source_v2' if native else 'joint_source')
    spec=importlib.util.spec_from_file_location('pending_ninehour_network_wrapper'+str(int(native)),root/'deployment_runtime.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.deployment_extension=lambda:extension(native)
    return mod.DeploymentRenderer


class PendingRenderer:
    def __init__(self,source,student=None,mode=38,native_dispatch=False):
        assert mode in [37,38,39,40]
        network=deployment_class(native_dispatch);self.native_dispatch=native_dispatch
        self.network=network(source.scene.model,source.scene.gaussians,source.scene.config,student=student,
            tensor_selector=2,tensor_compensator=2 if student is not None else 0,
            preparation_frustum=1,preparation_projection=1 if student is not None else 0,device_schedule=1)
        self.native=extension(native_dispatch).PendingRasterizer(512,True,True,12,18+mode,64)
        self.handle=self.native._shared_handle() if native_dispatch else None
        self.rest=self.network._base.base_master[5].numel()//max(1,self.network.total)
        self.degree=source.scene.gaussians.active_sh_degree

    def precompile(self,cam,bg):
        self.native.precompile(max(1,self.network.total),cam.image_width,cam.image_height,5,256,32,bg,False)

    @torch.no_grad()
    def render(self,cam,bg,condition):
        if condition=='full':
            packet,ids=self.network.full();empty=bg.new_empty(0);x=packet['inputs']
            image,radii=self.native.render(bg,x['means3D'],empty,x['opacities'],x['scales'],x['rotations'],1.,empty,
                cam.world_view_transform,cam.full_proj_transform,math.tan(cam.FoVx*.5),math.tan(cam.FoVy*.5),
                cam.image_height,cam.image_width,x['dc'],x['shs'],self.degree,cam.camera_center,5,256,32,False)
            return image,radii,ids,dict(candidate_gaussians=self.network.total,active_gaussians=self.network.total)
        is_ratio=condition.startswith('local');ratio=float(condition[5:]) if is_ratio else 0.;budget=0 if is_ratio else int(condition[5:])
        pipe=self.network._pipeline
        if self.native_dispatch:
            return pipe._render_pending_current(self.handle,bg,cam.world_view_transform,cam.full_proj_transform,cam.camera_center,
                cam.image_width,cam.image_height,float(cam.znear),budget,ratio,is_ratio,True,True,self.rest,
                math.tan(cam.FoVx*.5),math.tan(cam.FoVy*.5))
        pending=pipe._begin_pending(cam.world_view_transform,cam.full_proj_transform,cam.camera_center,
            cam.image_width,cam.image_height,float(cam.znear),budget,ratio,is_ratio,True,True)
        flat,selected,frame=pending;capacity=selected.numel()
        image,radii=self.native.render_pending(bg,flat,frame,capacity,self.rest,cam.world_view_transform,cam.full_proj_transform,
            math.tan(cam.FoVx*.5),math.tan(cam.FoVy*.5),cam.image_height,cam.image_width,self.degree,cam.camera_center)
        ids,stats=pipe._finish_pending()
        return image,radii[:len(ids)],ids,stats


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--native',action='store_true');args=parser.parse_args()
    print('PENDING_JOINT_BUILD_PASS',extension(args.native).__file__,flush=True)

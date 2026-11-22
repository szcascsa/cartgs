"""当前CPU位姿端到端候选；不保存依赖旧视角或Ratio的预测。"""
from pathlib import Path
from functools import lru_cache
import importlib.util
import math
import os
import torch
from pending_runtime import PendingRenderer

HERE=Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def extension():
    name='mycartgs_pending_joint_pose_v3';root=HERE/'joint_source_pose';build=HERE/'joint_build_pose';binary=build/(name+'.so')
    if binary.exists() and os.environ.get('POSE_REBUILD')!='1':
        spec=importlib.util.spec_from_file_location(name,binary);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
    os.environ['CUDA_HOME']='/usr/local/cuda-11.8';os.environ['TORCH_CUDA_ARCH_LIST']='8.6';os.environ['MAX_JOBS']='2'
    os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
    from torch.utils.cpp_extension import load
    files=['deployment_runtime_bindings.cpp','deployment_runtime_kernels.cu','deployment_prepare.cpp','deployment_selection.cu',
        'deployment_radix.cu','deployment_candidates.cu','frozen_inference_kernels.cu','preparation_kernels.cu','preparation_camera.cpp',
        'raster/graph_main.cu','raster/graph_forward.cu']
    return load(name=name,sources=[str(root/n) for n in files],extra_include_paths=[str(root/'raster/reference/cuda_rasterizer')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo'],build_directory=str(build),verbose=True)


@lru_cache(maxsize=1)
def deployment_class():
    spec=importlib.util.spec_from_file_location('pose_deployment_wrapper',HERE/'joint_source_pose/deployment_runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.deployment_extension=extension
    return module.DeploymentRenderer


class PoseRenderer(PendingRenderer):
    def __init__(self,source,student=None,mode=38):
        self.native_dispatch=True
        self.network=deployment_class()(source.scene.model,source.scene.gaussians,source.scene.config,student=student,
            tensor_selector=2,tensor_compensator=2 if student is not None else 0,
            preparation_frustum=1,preparation_projection=1 if student is not None else 0,device_schedule=1)
        self.native=extension().PendingRasterizer(512,True,True,12,18+mode,64);self.handle=self.native._shared_handle()
        self.rest=self.network._base.base_master[5].numel()//max(1,self.network.total)
        self.degree=source.scene.gaussians.active_sh_degree;self._calibration=None

    @torch.no_grad()
    def render_from_source(self,source,frame,bg,condition,factory):
        if condition=='full':return self.render(factory.from_source(source,frame),bg,condition)
        c=source.scene.calibration;W,H=c['width'],c['height']
        fovx,fovy=2*math.atan(W/(2*c['fx'])),2*math.atan(H/(2*c['fy']))
        key=(W,H,fovx,fovy,torch.cuda.current_stream().cuda_stream,torch.get_default_dtype())
        if key!=self._calibration:
            builder=factory.projection_builder
            self._projection=builder(znear=.01,zfar=100.,fovX=fovx,fovY=fovy).transpose(0,1).contiguous()
            self._device_reference=self._projection.cuda(non_blocking=True);self._calibration=key
        use_ratio=condition.startswith('local');ratio=float(condition[5:]) if use_ratio else 0.
        budget=0 if use_ratio else int(condition[5:])
        return self.network._pipeline._render_pose_current(self.handle,bg,source.scene.views[frame]['pose'],
            self._projection,self._device_reference,W,H,budget,ratio,use_ratio,self.rest,math.tan(fovx*.5),math.tan(fovy*.5))


if __name__=='__main__':print('POSE_DISPATCH_BUILD_PASS',extension().__file__,flush=True)

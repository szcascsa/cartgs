"""冻结部署的延迟预测控制；一次性摘要，每帧按真实CPU位姿重算。"""
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    dest=ROOT/'build';dest.mkdir(exist_ok=True)
    return load(name='mycartgs_latency_predictive_v3',sources=[str(ROOT/'source/latency_predictive_bindings.cpp')],
        extra_cflags=['-O3'],with_cuda=True,build_directory=str(dest),verbose=True)


@torch.no_grad()
def anchor_summary(engine):
    b=engine._base
    centers=b.anchors.centers.detach().cpu().numpy()
    half=b.anchors.offset_scales.detach().cpu().numpy().reshape(-1)*.5
    counts=b.eligible_counts.detach().cpu().numpy().reshape(-1)
    owners=b.owners.detach().cpu().numpy()
    valid=b.eligible.detach().cpu().numpy().astype(bool)
    scales=b.pc.get_scaling.detach().cpu().numpy().astype(np.float64)
    radius2=(scales*scales).mean(axis=1)
    sums=np.bincount(owners[valid],weights=radius2[valid],minlength=len(centers))
    radius2=sums/np.maximum(counts,1)
    data=np.ascontiguousarray(np.column_stack([centers,half,counts,radius2]),dtype=np.float32)
    assert np.isfinite(data).all() and data.shape==(len(centers),6)
    return data


def configure(controller,summary,projection,width,height):
    if isinstance(projection,torch.Tensor):
        if projection.device.type!='cpu':raise ValueError('Pass the existing host projection; do not download per frame')
        projection=projection.numpy()
    controller.configure_view(summary,np.ascontiguousarray(projection,dtype=np.float32),width,height)


class PredictiveLatencyRenderer:
    """仅训练后冻结推理；要求原生计时Renderer及调用方已有的CPU位姿/投影。"""
    def __init__(self,engine,renderer,background,projection,image_size,*,config=None):
        if not getattr(renderer,'native_latency_timing',False):
            raise ValueError('Predictive latency control requires native render timing')
        self.engine,self.renderer,self.background=engine,renderer,background
        self.size=tuple(image_size)
        options=dict(target_ms=.2,max_budget=engine.total)
        options.update(config or {})
        self.controller=extension().AsyncPID(options,background.device.index,8)
        configure(self.controller,anchor_summary(engine),projection,*self.size)

    def render(self,camera,*,camera_to_world,reset=False,timestamp_s=-1.):
        if torch.is_grad_enabled():raise RuntimeError('Predictive renderer is frozen inference only')
        if isinstance(camera_to_world,torch.Tensor):raise ValueError('Use the existing CPU pose, not a Tensor')
        if (camera.image_width,camera.image_height)!=self.size:raise ValueError('Create a new renderer for a new resolution')
        pose=np.asarray(camera_to_world).reshape(-1).tolist()
        budget=self.controller.request(pose,bool(reset),float(timestamp_s))
        packet,ids,stats=self.engine.prepare(camera,budget=budget)
        if len(ids):token,begin,end=self.controller.start_native(len(ids),stats['eligible_candidates'])
        else:token=self.controller.start(0,stats['eligible_candidates']);begin=end=0
        try:image,radii=self.renderer.render(camera,packet,self.background,(begin,end))
        except BaseException:
            self.controller.finish(token,False);raise
        self.controller.finish(token,True)
        return dict(render=image,radii=radii,ids=ids,stats=stats,pid_requested_budget=budget)

    def close(self):self.controller.close()

"""四Head与重训三Head同轮测时，另作同权重删除Head的受控微测。"""
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from common import ROOT,sha,save

OLD=Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105')
CONTROL=Path('/home/prmi/zwj/experiments/view_conditioned_li_20261106/runs/full_both_li/23560_shutdown/ply')
sys.path[:0]=[str(ROOT/'source/python'),str(OLD/'evaluation/python'),str(OLD/'evaluation/python/diff-gaussian-rasterization')]
import numpy as np
import torch
from budget_viewer import BudgetScene
from renderer import render_prepared
from utils import MiniCam,focal2fov


def gpu(label):
    return dict(label=label,unix=time.time(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,temperature.gpu,clocks.sm,memory.used','--format=csv,noheader'],text=True),
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True))


def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    out=ROOT/'speed';out.mkdir(exist_ok=False);start=time.monotonic()
    state=dict(state='loading',pid=os.getpid());save(out/'STATUS.json',state)
    paths={'four_head':CONTROL,'no_other':ROOT/'runs/full_no_other/23560_shutdown/ply'}
    scenes={g:BudgetScene(p,'/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',500,0,None,'pytorch') for g,p in paths.items()}
    for scene in scenes.values():scene.model.eval()
    assert scenes['four_head'].views==scenes['no_other'].views
    assert all(s.renderer.total==13298 for s in scenes.values())
    reference=scenes['four_head'];c=reference.calibration
    cameras=[MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(np.asarray(v['pose']))) for v in reference.views]
    methods=[(g,b) for g in scenes for b in ['points500','ratio50','full']]
    bg=reference.background
    def call(group,budget,frame):
        scene=scenes[group];torch.cuda.synchronize();before=time.perf_counter();stats={}
        if budget=='full':data=scene.full_data;mid=before;K=13298
        else:
            data,ids,stats=scene.renderer.prepare(cameras[frame],view_conditioned=True,**({'budget':500} if budget=='points500' else {'ratio_budget':.5}))
            mid=time.perf_counter();K=len(ids)
            requested=500 if budget=='points500' else math.floor(.5*stats['candidate_gaussians'])
            assert K==min(requested,stats['eligible_candidates']) and not stats['cache_hit']
        result=render_prepared(cameras[frame],data,bg);torch.cuda.synchronize();end=time.perf_counter()
        return dict(group=group,budget=budget,frame=frame,K=K,candidate_N=stats.get('candidate_gaussians',13298),eligible_N=stats.get('eligible_candidates',13298),
            inference_ms=(mid-before)*1000,selector_ms=stats.get('selector_ms',0),compensation_ms=stats.get('compensation_ms',0),
            other_prepare_ms=(mid-before)*1000-stats.get('selector_ms',0)-stats.get('compensation_ms',0),render_ms=(end-mid)*1000,total_ms=(end-before)*1000)
    for f in np.linspace(0,791,64,dtype=int):
        for g,b in methods:call(g,b,int(f))
    states=[gpu('before')];rows=[]
    for repeat in range(3):
        order=list(range(792))
        if repeat==1:order.reverse()
        if repeat==2:random.Random(0).shuffle(order)
        states.append(gpu(f'round_{repeat}_start'))
        for index,f in enumerate(order):
            shift=(index+repeat)%len(methods)
            for g,b in methods[shift:]+methods[:shift]:
                item=call(g,b,f);item.update(repeat=repeat);rows.append(item)
            if index%200==0:state.update(state='timing',repeat=repeat,frame=index+1,seconds=time.monotonic()-start);save(out/'STATUS.json',state)
        states.append(gpu(f'round_{repeat}_end'))
    write(out/'TIMINGS.csv',rows);save(out/'GPU_OBSERVATIONS.json',states)
    parameters={}
    for g,scene in scenes.items():
        parameters[g]=dict(selector=sum(p.numel() for n,p in scene.model.named_parameters() if n.startswith(('local_emd.','soft_net.'))),
            compensation=sum(p.numel() for p in scene.model.spatial_compensation.parameters()),
            feature_tables=sum(p.numel() for n,p in scene.model.named_parameters() if n.startswith('anchors.compensation_features_')))
    assert parameters['four_head']['compensation']-parameters['no_other']['compensation']==4420
    save(out/'PARAMETERS.json',parameters)
    # Same trained four-head weights and identical physical/Feature/view inputs; remove only Other.
    full=reference.model.spatial_compensation;stripped=copy.deepcopy(full);stripped.configure_geometry_heads(1)
    micro=[];training_micro=[];events=[torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)]
    pc=reference.renderer.gaussians;bank=reference.model.anchors;code=reference.model.joint_view_code(cameras[396])
    for k in [500,4096]:
        ids=torch.arange(k,device='cuda');owners=bank.rows[ids]
        for ratio in [.05,.5]:
            condition=torch.cat((code[owners],code.new_ones((k,1))),1)
            arguments=[pc.get_xyz[ids],pc._scaling[ids],pc.get_rotation[ids],pc.get_features_dc[ids],pc._opacity[ids],bank.centers[owners],bank.offset_scales[owners],ratio]
            kwargs=dict(log_scale_input=True,anchor_feature=bank.compensation_lookup(ids,ratio),view_condition=condition)
            a,b=full(*arguments,**kwargs),stripped(*arguments,**kwargs)
            assert torch.equal(a[:,:10].contiguous().view(torch.uint8),b[:,:10].contiguous().view(torch.uint8)) and b[:,10:].count_nonzero()==0
            for _ in range(64):full(*arguments,**kwargs);stripped(*arguments,**kwargs)
            for repeat in range(200):
                for label,net in ([('four_head',full),('no_other',stripped)] if repeat%2==0 else [('no_other',stripped),('four_head',full)]):
                    torch.cuda.synchronize();t=time.perf_counter();events[0].record();value=net(*arguments,**kwargs);events[1].record();events[1].synchronize()
                    micro.append(dict(K=k,ratio=ratio,repeat=repeat,group=label,wall_ms=(time.perf_counter()-t)*1000,cuda_stream_ms=events[0].elapsed_time(events[1])))
            # Same physical inputs and retained weights; nonzero upstream gradients on all 14 channels.
            # These timings include network backward, excluding Feature-table scatter and Adam.
            with torch.enable_grad():
                training_arguments=[v.detach().clone().requires_grad_(True) if isinstance(v,torch.Tensor) and index in [0,5,6] else v
                                    for index,v in enumerate(arguments)]
                training_kwargs=dict(kwargs,anchor_feature=kwargs['anchor_feature'].detach().clone().requires_grad_(True))
                upstream=torch.ones((k,14),device='cuda')
                inputs=[training_arguments[index] for index in [0,5,6]]+[training_kwargs['anchor_feature']]
                for net in [full,stripped]:
                    for parameter in net.parameters():parameter.requires_grad_(True)
                def train_call(net):
                    net.zero_grad(set_to_none=True)
                    for tensor in inputs:tensor.grad=None
                    net(*training_arguments,**training_kwargs).backward(upstream)
                for _ in range(16):train_call(full);train_call(stripped)
                for repeat in range(80):
                    for label,net in ([('four_head',full),('no_other',stripped)] if repeat%2==0 else [('no_other',stripped),('four_head',full)]):
                        torch.cuda.synchronize();t=time.perf_counter();events[0].record();train_call(net);events[1].record();events[1].synchronize()
                        training_micro.append(dict(K=k,ratio=ratio,repeat=repeat,group=label,wall_ms=(time.perf_counter()-t)*1000,cuda_stream_ms=events[0].elapsed_time(events[1])))
                for net in [full,stripped]:
                    net.zero_grad(set_to_none=True)
                    for parameter in net.parameters():parameter.requires_grad_(False)
    write(out/'COMPENSATION_SAME_INPUT.csv',micro)
    write(out/'COMPENSATION_FORWARD_BACKWARD.csv',training_micro)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        for f in [0,264,528,791]:
            for g,b in methods:
                with torch.profiler.record_function('METHOD/'+g+'/'+b):call(g,b,f)
    prof.export_chrome_trace(str(out/'PROFILE.json'))
    save(out/'ENVIRONMENT.json',dict(device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,
        source_python=str(ROOT/'source/python'),tf32=False,checkpoint={g:str(p) for g,p in paths.items()},images_saved=0,source_hash=sha(Path(__file__))))
    state.update(state='complete',rows=len(rows),micro_rows=len(micro),forward_backward_rows=len(training_micro),seconds=time.monotonic()-start);save(out/'STATUS.json',state)


if __name__=='__main__':
    try:main()
    except Exception as error:save(ROOT/'speed/STATUS.json',dict(state='failed',error=repr(error)));raise

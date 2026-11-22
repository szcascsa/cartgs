"""原SpatialLocalXYZ：先实测Profiler和CUDA Events，不改历史实现。"""
import json
import hashlib
import shutil
import sys
import time
import os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity, record_function

ROOT=Path(__file__).resolve().parent
REFERENCE=Path('/home/prmi/zwj/neural-v1-20261027')
sys.path.insert(0,str(REFERENCE))
import run as r


def record_version():
    directory=ROOT/'code_versions';directory.mkdir(exist_ok=True);sources={}
    for path in ROOT.glob('*.py'):
        digest=hashlib.sha256(path.read_bytes()).hexdigest();sources[path.name]=digest
        target=directory/(path.stem+'_'+digest[:12]+'.py')
        if not target.exists():target.write_bytes(path.read_bytes())
    version=ROOT/'run_versions';version.mkdir(exist_ok=True)
    (version/(Path(sys.argv[0]).stem+'_'+str(os.getpid())+'.json')).write_text(json.dumps(dict(
        utc=datetime.now(timezone.utc).isoformat(),argv=sys.argv,source_sha256=sources,baseline_commit='74fc1524d8d6345eda8e732a1da2efa9a4934951'),indent=2))


record_version()


def fixture(scene='fr1_xyz'):
    cfg=json.loads((ROOT/'spatial_localxyz_default.json').read_text())
    sc=r.rt.Scene(scene);f=r.Fixture(sc,3100 if scene=='fr1_xyz' else 4100)
    net=r.make(f,cfg['config'],5100 if scene=='fr1_xyz' else 6100)
    payload=torch.load(cfg['checkpoints'][scene],map_location='cuda')
    net.load_state_dict(payload['state_dict'])
    return sc,f,net,payload


def measure(fn,repeats=40,warmup=10):
    for _ in range(warmup):fn()
    torch.cuda.synchronize();events=[];walls=[]
    torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        start=time.perf_counter();a.record();fn();b.record();events.append((a,b));walls.append((time.perf_counter()-start)*1000)
    torch.cuda.synchronize();times=[a.elapsed_time(b) for a,b in events]
    return dict(cuda_median_ms=float(np.median(times)),cuda_p10_ms=float(np.quantile(times,.1)),cuda_p90_ms=float(np.quantile(times,.9)),
        cpu_submit_median_ms=float(np.median(walls)),peak_increment_bytes=max(0,torch.cuda.max_memory_allocated()-base),repeats=repeats)


def main():
    torch.backends.cuda.matmul.allow_tf32=False
    env=dict(torch=torch.__version__,triton=__import__('triton').__version__,device=torch.cuda.get_device_name(),tf32=False,
        executables={n:shutil.which(n) for n in ['nvcc','nsys','ncu','g++']},reference_sha={n:hashlib.sha256((REFERENCE/n).read_bytes()).hexdigest() for n in ['neural.py','run.py','matrix_runtime.py']})
    (ROOT/'environment.json').write_text(json.dumps(env,indent=2))
    sc,f,net,payload=fixture();rows=[]
    optimizer=torch.optim.Adam(net.parameters(),lr=.001,eps=1e-15,fused=True)
    optimizer.load_state_dict(payload['optimizer'])
    freq=(2**torch.arange(6,device='cuda'))*__import__('math').pi
    for ratio in r.rt.RATIOS:
        ids=f.sets[ratio,'same_k'];K=len(ids)
        with torch.no_grad():d=net(ratio,ids)
        delta=d.detach().requires_grad_();image=r.render(f,net,ids,delta,sc.train[0])[0]
        vjp=torch.autograd.grad(r.rt.photo(image,sc.train[0]['gt']),delta)[0].detach()
        def forward():
            with torch.no_grad():return net(ratio,ids)
        def fb():net.zero_grad(set_to_none=True);net(ratio,ids).backward(vjp)
        def step():optimizer.zero_grad(set_to_none=True);net(ratio,ids).backward(vjp);optimizer.step()
        def e2e():
            optimizer.zero_grad(set_to_none=True);out=net(ratio,ids)
            loss=r.rt.photo(r.render(f,net,ids,out,sc.train[0])[0],sc.train[0]['gt']);loss.backward();optimizer.step()
        for stage,fn in [('forward',forward),('forward_backward',fb),('optimizer_step',step),('renderer_loss_step',e2e)]:
            rows.append(dict(method='reference',scene=sc.name,K=K,ratio=ratio,stage=stage,**measure(fn)))
        if K in [68,1025]:
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as p:
                for _ in range(4):
                    with record_function('input_pack'):x=f.features.inputs(ids,ratio,net.cfg)
                    with record_function('compensation_forward'):out=net(ratio,ids)
                    with record_function('compensation_backward'):net.zero_grad(set_to_none=True);out.backward(vjp)
                    with record_function('adam'):optimizer.step()
            p.export_chrome_trace(str(ROOT/f'baseline_K{K}_trace.json'))
            entries=[]
            for event in p.key_averages():
                entries.append(dict(K=K,operator=event.key,calls=event.count,cpu_us=event.self_cpu_time_total,
                    cuda_us=event.self_cuda_time_total,cpu_memory=event.self_cpu_memory_usage,cuda_memory=event.self_cuda_memory_usage))
            r.rt.write_csv(ROOT/f'baseline_K{K}_operators.csv',entries)
        print('PROFILE',K,flush=True)
    r.rt.write_csv(ROOT/'baseline_events.csv',rows)
    (ROOT/'profile_complete.json').write_text(json.dumps(dict(completed=True,baseline='original SharedNeural; cached immutable 19D; on-the-fly Fourier; FP32',source_checkpoint='SpatialLocalXYZ/5100/step_12000.pt'),indent=2))


if __name__=='__main__':main()

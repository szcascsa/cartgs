"""锁定候选的神经梯度、输出连续性、属性条件替换与开销。"""
import argparse
import gc
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import run as r
from neural import audit_parameters


def timing(fn,n=80):
    for _ in range(10):fn()
    torch.cuda.synchronize();start=time.perf_counter();events=[]
    for _ in range(n):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();fn();b.record();events.append((a,b))
    torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000/n
    a=[x.elapsed_time(y) for x,y in events]
    return dict(cuda_median_ms=float(np.median(a)),cuda_p90_ms=float(np.quantile(a,.9)),wall_mean_ms=wall,repeats=n)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);args=p.parse_args();plan=json.loads(Path(args.plan).read_text())
    scene=r.rt.Scene(plan.get('scene','fr1_xyz'));f=r.Fixture(scene,plan.get('source_seed',3100));rows=[];conditions=[];continuity=[];gradients=[];memory=[]
    for item in plan['models']:
        d=Path(item['directory']);info=json.loads((d/'complete.json').read_text());cfg=info['config'];seed=info['seed'];net=r.make(f,cfg,seed)
        payload=torch.load(info['checkpoint'],map_location='cuda');net.load_state_dict(payload['state_dict']);net.eval()
        if cfg['kind']!='reference59':audit_parameters(net)
        optstates=payload['optimizer']['state'];moments=sum(v.numel()*v.element_size() for state in optstates.values() for name,v in state.items()
            if name in ['exp_avg','exp_avg_sq'] and isinstance(v,torch.Tensor))
        memory.append(dict(config=cfg['id'],seed=seed,parameters=sum(p.numel() for p in net.parameters()),
            parameter_bytes=sum(p.numel()*p.element_size() for p in net.parameters()),adam_moments_bytes=moments,
            feature_bytes=f.features.own.numel()*4,cache_scope='deterministic frozen geometry feature cache; common fixture includes optional neighbor caches'))
        # 梯度只比较同一8Bank视图，同名共享参数；负cosine是描述，不直接判断有害。
        grads={}
        for ratio in [.01,.15]:
            ids=f.sets[ratio,'same_k'];net.zero_grad(set_to_none=True)
            loss=sum(r.rt.photo(r.render(f,net,ids,net(ratio,ids),view)[0],view['gt'])/8 for view in scene.train[::max(1,len(scene.train)//8)][:8])
            loss.backward();grads[ratio]={name:p.grad.detach().clone() for name,p in net.named_parameters() if p.grad is not None}
        for name in grads[.01].keys() & grads[.15].keys():
            a,b=grads[.01][name],grads[.15][name]
            gradients.append(dict(config=cfg['id'],seed=seed,parameter=name,norm1=float(a.norm()),norm15=float(b.norm()),
                cosine=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-20)),scope='snapshot physicalphoto across8Bankframes; no causal claim'))
        for ratio in r.rt.RATIOS:
            ids=f.sets[ratio,'same_k']
            with torch.no_grad():delta=net(ratio,ids)
            # 主候选无context，应只依赖输入属性而不依赖行号顺序。
            if cfg.get('context','none')=='none' and cfg['kind']!='reference59':
                with torch.no_grad():assert torch.allclose(net(ratio,ids.flip(0)),delta.flip(0),atol=2e-6,rtol=2e-5)
            probe=delta.detach().requires_grad_(True);image=r.render(f,net,ids,probe,scene.train[0])[0]
            vjp=torch.autograd.grad(r.rt.photo(image,scene.train[0]['gt']),probe)[0].detach()
            def forward():
                with torch.no_grad():return net(ratio,ids)
            def backward():
                net.zero_grad(set_to_none=True);net(ratio,ids).backward(vjp)
            for stage,fn in [('forward',forward),('forward_backward_photo_vjp',backward)]:
                net.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
                result=timing(fn)
                rows.append(dict(config=cfg['id'],seed=seed,ratio=ratio,N=scene.N,K=len(ids),stage=stage,
                    peak_increment_bytes=max(0,torch.cuda.max_memory_allocated()-base),**result))
            if ratio in [.01,.05]:
                with torch.no_grad():
                    for group,lo,hi in [('xyz',0,3),('scale',3,6),('rotation',6,10),('opacity',10,11),('dc',11,14)]:
                        altered=delta.clone();altered[:,lo:hi]=0
                        for vi,view in enumerate(scene.validation[::max(1,len(scene.validation)//4)][:4]):
                            a=r.render(f,net,ids,delta,view)[0];b=r.render(f,net,ids,altered,view)[0]
                            psnr=lambda x:float(-10*torch.log10((x.clamp(0,1)-view['gt']).square().mean()))
                            conditions.append(dict(config=cfg['id'],seed=seed,ratio=ratio,group=group,view=vi,
                                psnr_loss_removing_component=psnr(a)-psnr(b),scope='conditional nonadditive group ablation; no retraining'))
        if cfg.get('context','none')=='none' and cfg['kind']!='reference59':
            ids=f.sets[.15,'same_k'];previous=None
            with torch.no_grad():
                for ratio in np.linspace(.01,.15,141):
                    value=net(float(ratio),ids);assert value.isfinite().all()
                    if previous is not None:
                        continuity.append(dict(config=cfg['id'],seed=seed,ratio=float(ratio),
                            xyz_step_p99=float((value[:,:3]-previous[:,:3]).norm(dim=1).quantile(.99)),
                            raw_output_step_p99=float((value-previous).abs().flatten().quantile(.99)),
                            xyz_amplitude_p99=float(value[:,:3].norm(dim=1).quantile(.99)),finite=True,
                            scope='fixedID continuity; raw14 distance is numerical stability only, not quality'))
                    previous=value
        out=r.ROOT/'mechanism';out.mkdir(exist_ok=True)
        for file,values in [('timing.csv',rows),('memory.csv',memory),('gradients.csv',gradients),('attribute_conditions.csv',conditions),('continuity.csv',continuity)]:r.rt.write_csv(out/file,values)
        assert scene.unchanged();del net,payload;gc.collect();torch.cuda.empty_cache();print('MECHANISM',cfg['id'],seed,flush=True)


if __name__=='__main__':main()

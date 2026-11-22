"""直接Backward计时与Graph持久显存；不将Forward差值冒充Backward。"""
import copy
import argparse
import gc
import json
import time
import numpy as np
import torch
from baseline_profile import ROOT,r,fixture
from implementation import Execution


def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',default=['reference','graph_fused','graph_all_fused']);p.add_argument('--tag',default='');args=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[];memory=[]
    for scene in ['fr1_xyz','fr2_xyz']:
        sc,f,template,payload=fixture(scene)
        for method in args.methods:
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();base=torch.cuda.memory_allocated()
            net=r.make(f,template.cfg,5100);net.load_state_dict(payload['state_dict']);ex=Execution(net,method)
            opt=torch.optim.Adam(net.parameters(),lr=.001,eps=1e-15,fused=True);opt.load_state_dict(copy.deepcopy(payload['optimizer']))
            for ratio in r.rt.RATIOS:
                ids=f.sets[ratio,'same_k'];leaf=template(ratio,ids).detach().requires_grad_()
                vjp=torch.autograd.grad(r.rt.photo(sc.render(ids,leaf,sc.train[0])[0],sc.train[0]['gt']),leaf)[0].detach()
                times=[];wall=[]
                for j in range(50):
                    net.zero_grad(set_to_none=True);out=ex(ratio,ids)
                    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();start=time.perf_counter()
                    out.backward(vjp);b.record()
                    if j>=10:times.append((a,b));wall.append((time.perf_counter()-start)*1000)
                torch.cuda.synchronize();values=[a.elapsed_time(b) for a,b in times]
                rows.append(dict(scene=scene,method=method,K=len(ids),ratio=ratio,backward_median_ms=float(np.median(values)),
                    backward_p10_ms=float(np.quantile(values,.1)),backward_p90_ms=float(np.quantile(values,.9)),cpu_submit_ms=float(np.median(wall))))
            torch.cuda.synchronize();after=torch.cuda.memory_allocated()
            memory.append(dict(scene=scene,method=method,all_seven_shapes=True,parameters=sum(p.numel() for p in net.parameters()),
                parameter_bytes=sum(p.numel()*p.element_size() for p in net.parameters()),persistent_increment_bytes=after-base,
                graphs=len(ex.graphs),optimizer_moment_bytes=sum(v.numel()*v.element_size() for state in opt.state.values() for k,v in state.items() if k in ['exp_avg','exp_avg_sq']),
                scope='network + original Adam + all seven captured shapes + last backward buffers; fixture caches common/excluded'))
            print('EXTRA_BENCH',scene,method,flush=True)
            del net,ex,opt,out,leaf,vjp;gc.collect();torch.cuda.empty_cache()
        r.rt.write_csv(ROOT/(args.tag+'backward_events.csv'),rows);r.rt.write_csv(ROOT/(args.tag+'persistent_memory.csv'),memory)


if __name__=='__main__':main()

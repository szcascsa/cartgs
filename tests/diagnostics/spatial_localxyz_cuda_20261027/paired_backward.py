"""相同真实photo VJP的Backward-only交错计时；Forward置于事件区间外。"""
import random
import argparse
import time
import numpy as np
import torch
from baseline_profile import ROOT,r,fixture
from implementation import Execution


def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',default=['reference','graph_shared_fused']);p.add_argument('--tag',default='');args=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;rows=[]
    for scene in ['fr1_xyz','fr2_xyz']:
        sc,f,template,payload=fixture(scene);models={}
        for method in args.methods:
            net=r.make(f,template.cfg,5100);net.load_state_dict(payload['state_dict']);models[method]=(net,Execution(net,method))
        for ratio in r.rt.RATIOS:
            ids=f.sets[ratio,'same_k'];leaf=template(ratio,ids).detach().requires_grad_()
            vjp=torch.autograd.grad(r.rt.photo(sc.render(ids,leaf,sc.train[0])[0],sc.train[0]['gt']),leaf)[0].detach()
            for block in range(3):
                methods=list(models);random.Random(len(ids)+block).shuffle(methods)
                for method in methods:
                    net,ex=models[method];events=[];cpu=[]
                    for j in range(45):
                        net.zero_grad(set_to_none=True);out=ex(ratio,ids)
                        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();start=time.perf_counter()
                        out.backward(vjp);b.record()
                        if j>=10:events.append((a,b));cpu.append((time.perf_counter()-start)*1000)
                    torch.cuda.synchronize();values=[a.elapsed_time(b) for a,b in events]
                    rows.append(dict(scene=scene,method=method,ratio=ratio,K=len(ids),block=block,backward_median_ms=float(np.median(values)),
                        backward_p10_ms=float(np.quantile(values,.1)),backward_p90_ms=float(np.quantile(values,.9)),cpu_submit_ms=float(np.median(cpu))))
            r.rt.write_csv(ROOT/(args.tag+'paired_backward.csv'),rows)
        print('PAIRED_BACKWARD_COMPLETE',scene,flush=True)


if __name__=='__main__':main()

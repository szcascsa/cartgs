"""配对交错NN+解码计时；Forward/Backward范围与Compensation-only分开。"""
import argparse
import copy
import gc
import json
import time
import numpy as np
import torch
from baseline_profile import ROOT,r,fixture,measure
from implementation import Execution
from decoded_graph import DecodedExecution,decode,render_decoded


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scene',default='fr1_xyz');parser.add_argument('--blocks',type=int,default=3)
    parser.add_argument('--memory-only',action='store_true');parser.add_argument('--method',default='decoded_graph');args=parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    sc,f,template,payload=fixture(args.scene);rows=[];backward=[];memory=[]
    methods=['reference','graph_cached55_fused','decoded_graph'] if not args.memory_only else [args.method]
    for block in range(1 if args.memory_only else args.blocks):
        order=methods[block%len(methods):]+methods[:block%len(methods)]
        for method in order:
            gc.collect();torch.cuda.empty_cache();before=torch.cuda.memory_allocated()
            net=r.make(f,template.cfg,5100);net.load_state_dict(payload['state_dict'])
            opt=torch.optim.Adam(net.parameters(),lr=.001,eps=1e-15,fused=True);opt.load_state_dict(copy.deepcopy(payload['optimizer']))
            ex=DecodedExecution(net,sc.render_bases) if method=='decoded_graph' else Execution(net,method)
            def values(ratio,ids):
                if method=='decoded_graph':return ex(ratio,ids)
                delta=ex(ratio,ids);return (delta,*decode(delta,ids,(*sc.render_bases[:4],sc.render_bases[5])))
            def reset():
                opt.zero_grad(set_to_none=True);net.load_state_dict(payload['state_dict']);opt.load_state_dict(copy.deepcopy(payload['optimizer']))
            for ratio in r.rt.RATIOS:
                reset()
                ids=f.sets[ratio,'same_k'];view=sc.train[0];K=len(ids)
                start=time.perf_counter();out=values(ratio,ids);torch.cuda.synchronize();setup_ms=(time.perf_counter()-start)*1000
                leaves=tuple(v.detach().requires_grad_() for v in out)
                loss=r.rt.photo(render_decoded(sc,ids,leaves,view),view['gt'])
                gg=torch.autograd.grad(loss,leaves,allow_unused=True);vjp=tuple(torch.zeros_like(p) if g is None else g for p,g in zip(leaves,gg))
                def forward():
                    with torch.no_grad():return values(ratio,ids)
                def fb():opt.zero_grad(set_to_none=True);torch.autograd.backward(values(ratio,ids),vjp)
                def nn_step():fb();opt.step()
                def adam():opt.step()
                def e2e():
                    opt.zero_grad(set_to_none=True);out=values(ratio,ids)
                    loss=r.rt.photo(render_decoded(sc,ids,out,view),view['gt']);loss.backward();opt.step()
                if args.memory_only:fb();continue
                for stage,fn in [('forward_decode',forward),('forward_backward_decode',fb),('adam_only',adam),('network_step_decode',nn_step),('renderer_loss_step',e2e)]:
                    reset()
                    if stage=='adam_only':fb()
                    rows.append(dict(scene=sc.name,method=method,block=block,ratio=ratio,K=K,stage=stage,setup_ms=setup_ms,**measure(fn,35)))
                reset();fb();events=[]
                for _ in range(35):
                    opt.zero_grad(set_to_none=True);out=values(ratio,ids)
                    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                    a.record();torch.autograd.backward(out,vjp);b.record();events.append((a,b))
                torch.cuda.synchronize();ms=[a.elapsed_time(b) for a,b in events]
                backward.append(dict(scene=sc.name,method=method,block=block,ratio=ratio,K=K,backward_median_ms=float(np.median(ms)),backward_p10_ms=float(np.quantile(ms,.1)),backward_p90_ms=float(np.quantile(ms,.9))))
                r.rt.write_csv(ROOT/f'decoded_benchmark_{sc.name}.csv',rows);r.rt.write_csv(ROOT/f'decoded_backward_{sc.name}.csv',backward)
                print('BENCHMARK',sc.name,block,method,K,flush=True)
            if args.memory_only:
                memory.append(dict(scene=sc.name,method=method,persistent_increment_bytes=torch.cuda.memory_allocated()-before,graphs=len(ex.graphs),all_seven_shapes=True,parameters=sum(p.numel() for p in net.parameters()),parameter_bytes=sum(p.numel()*p.element_size() for p in net.parameters()),optimizer_moment_bytes=sum(s[k].numel()*s[k].element_size() for s in opt.state.values() for k in ['exp_avg','exp_avg_sq'])))
                r.rt.write_csv(ROOT/f'decoded_memory_{args.method}_{sc.name}.csv',memory)
            del out,leaves,loss,gg,vjp,ex,opt,net
    print('COMPLETE',args.scene,args.memory_only,flush=True)


if __name__=='__main__':main()

"""相同范围、权重、K与梯度路径；交错配对，独立记录CPU提交与CUDA区间。"""
import argparse
import copy
import json
import random
import torch
from torch.profiler import profile,ProfilerActivity
from baseline_profile import ROOT,r,fixture,measure
from implementation import Execution


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',default='fr1_xyz');p.add_argument('--repeats',type=int,default=35);p.add_argument('--blocks',type=int,default=3);p.add_argument('--tag',default='benchmark')
    p.add_argument('--methods',nargs='+',default=['reference','fused','graph_reference','graph_fused']);args=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    sc,f,original,payload=fixture(args.scene);methods={};rows=[];profiles=[]
    for name in args.methods:
        net=r.make(f,original.cfg,5100);net.load_state_dict(payload['state_dict']);ex=Execution(net,name)
        opt=torch.optim.Adam(net.parameters(),lr=.001,eps=1e-15,fused=True);opt.load_state_dict(copy.deepcopy(payload['optimizer']))
        methods[name]=(net,ex,opt)
    for ratio in r.rt.RATIOS:
        ids=f.sets[ratio,'same_k'];K=len(ids)
        leaf=original(ratio,ids).detach().requires_grad_();image=r.render(f,original,ids,leaf,sc.train[0])[0]
        vjp=torch.autograd.grad(r.rt.photo(image,sc.train[0]['gt']),leaf)[0].detach()
        for block in range(args.blocks):
            names=list(methods);random.Random(23+K+block).shuffle(names)
            for name in names:
                net,ex,opt=methods[name];net.load_state_dict(payload['state_dict']);opt.load_state_dict(copy.deepcopy(payload['optimizer']))
                # 首次图捕获/编译和渐进内存分配不计入稳态计时，单独报告初始化成本。
                import time
                start=time.perf_counter();ex(ratio,ids).backward(vjp);torch.cuda.synchronize();setup_ms=(time.perf_counter()-start)*1000
                def forward():
                    with torch.no_grad():return ex(ratio,ids)
                def backward():net.zero_grad(set_to_none=True);ex(ratio,ids).backward(vjp)
                def step():opt.zero_grad(set_to_none=True);ex(ratio,ids).backward(vjp);opt.step()
                def adam():opt.step()
                def e2e():
                    opt.zero_grad(set_to_none=True)
                    loss=r.rt.photo(r.render(f,net,ids,ex(ratio,ids),sc.train[0])[0],sc.train[0]['gt']);loss.backward();opt.step()
                for stage,fn in [('forward',forward),('forward_backward',backward),('adam_only',adam),('optimizer_step',step),('renderer_loss_step',e2e)]:
                    # 每个计时范围都从同一真实Checkpoint/Adam开始；隔离Adam微基准不污染后续Renderer状态。
                    net.load_state_dict(payload['state_dict']);opt.load_state_dict(copy.deepcopy(payload['optimizer']))
                    result=measure(fn,args.repeats)
                    rows.append(dict(method=name,scene=sc.name,N=sc.N,K=K,ratio=ratio,block=block,stage=stage,setup_ms=setup_ms,**result))
                r.rt.write_csv(ROOT/f'{args.tag}_{args.scene}.csv',rows)
                # Torch2.3.1/CUPTI在全打包Graph的trace后处理停滞；该路径使用CUDA Events，不伪造kernel计数。
                if block==0 and ratio in [.01,.15] and not any(m in args.methods for m in ['graph_all_fused','graph_shared_fused','graph_cached55_fused']):
                    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
                        for _ in range(4):e2e()
                    path=ROOT/f'{args.tag}_profile_{args.scene}_{name}_K{K}.json';prof.export_chrome_trace(str(path))
                    events=json.loads(path.read_text())['traceEvents'];ops=[]
                    for op in prof.key_averages():
                        ops.append(dict(operator=op.key,calls=op.count,cpu_us=op.self_cpu_time_total,cuda_us=op.self_cuda_time_total,
                            cpu_memory_bytes=op.self_cpu_memory_usage,cuda_memory_bytes=op.self_cuda_memory_usage))
                    r.rt.write_csv(ROOT/f'{args.tag}_operators_{args.scene}_{name}_K{K}.csv',ops)
                    profiles.append(dict(method=name,scene=sc.name,K=K,kernels_per_step=sum(e.get('cat')=='kernel' for e in events)/4,
                        cuda_active_us_per_step=sum(e.get('dur',0) for e in events if e.get('cat')=='kernel')/4,
                        memcpy_events_per_step=sum('memcpy' in str(e.get('cat','')).lower() for e in events)/4,
                        allocation_calls_per_step=sum(o['calls'] for o in ops if o['operator'].startswith(('aten::empty','aten::new_empty')))/4))
                r.rt.write_csv(ROOT/f'{args.tag}_{args.scene}.csv',rows);r.rt.write_csv(ROOT/f'{args.tag}_profile_summary_{args.scene}.csv',profiles)
                print('BENCH',sc.name,K,block,name,flush=True)
    (ROOT/f'{args.tag}_{args.scene}_complete.json').write_text(json.dumps(dict(rows=len(rows),scene=sc.name,methods=args.methods),indent=2))
    assert sc.unchanged()


if __name__=='__main__':main()

"""原12k终点精确续训256真实photo步；固定帧序/K，比较真实Renderer轨迹波动。"""
import copy
import argparse
import json
import time
import torch
from baseline_profile import ROOT,r,fixture
from implementation import Execution


def main():
    p=argparse.ArgumentParser();p.add_argument('--optimized-method',default='graph_all_fused');p.add_argument('--tag',default='');a=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    sc,f,template,payload=fixture();steps=256;offset=int(payload['step'])//4
    sequence=r.rt.schedule(sc,5100,offset+steps//4)[offset:];curves=[];quality=[];summaries=[]
    for label,method in [('reference','reference'),('reference_repeat','reference'),('optimized',a.optimized_method)]:
        net=r.make(f,template.cfg,5100);net.load_state_dict(payload['state_dict'])
        if method=='decoded_graph':
            from decoded_graph import DecodedExecution
            ex=DecodedExecution(net,sc.render_bases)
        else:ex=Execution(net,method)
        opt=torch.optim.Adam(net.parameters(),lr=.001,eps=1e-15,fused=True);opt.load_state_dict(copy.deepcopy(payload['optimizer']))
        # 只预建Graph，不更新任何参数或Adam时钟；记录真实首次接入成本。
        before={n:p.detach().clone() for n,p in net.named_parameters()};start=time.perf_counter()
        for ratio in r.rt.TRAIN:ex(ratio,f.sets[ratio,'same_k'])
        torch.cuda.synchronize();setup_ms=(time.perf_counter()-start)*1000
        assert all(torch.equal(p,before[n]) for n,p in net.named_parameters())
        losses=[];torch.cuda.synchronize();start=time.perf_counter()
        for step in range(steps):
            ratio=r.rt.TRAIN[step%4];ids=f.sets[ratio,'same_k'];view=sc.train[sequence[step//4]]
            opt.zero_grad(set_to_none=True)
            image=ex.render(sc,ratio,ids,view) if method=='decoded_graph' else r.render(f,net,ids,ex(ratio,ids),view)[0]
            loss=r.rt.photo(image,view['gt']);loss.backward();opt.step();losses.append(loss.detach())
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        vals=torch.stack(losses).cpu().tolist()
        curves.extend(dict(method=label,step=step+1,ratio=r.rt.TRAIN[step%4],path=sc.train[sequence[step//4]]['path'],photo=value) for step,value in enumerate(vals))
        assert all(int(opt.state[p]['step'])==int(payload['step'])+steps for p in net.parameters())
        with torch.no_grad():
            for ratio in r.rt.RATIOS:
                ids=f.sets[ratio,'same_k'];delta=ex(ratio,ids) if method!='decoded_graph' else None
                for i,view in enumerate(sc.validation):
                    image=ex.render(sc,ratio,ids,view) if method=='decoded_graph' else r.render(f,net,ids,delta,view)[0]
                    quality.append(dict(method=label,ratio=ratio,K=len(ids),view=i,path=view['path'],
                        psnr=float(-10*torch.log10((image.clamp(0,1)-view['gt']).square().mean())),photo=float(r.rt.photo(image,view['gt']))))
        target=ROOT/f'{a.tag}real_training_{label}.pt';torch.save(dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),steps=steps,
            prior_steps=int(payload['step']),source='identical original checkpoint and optimizer',frame_order=sequence),target)
        summaries.append(dict(method=label,backend=method,steps=steps,seconds=seconds,ms_per_step=seconds*1000/steps,setup_ms=setup_ms,
            final_photo=vals[-1],adam_clock=int(payload['step'])+steps))
        r.rt.write_csv(ROOT/(a.tag+'real_training_curve.csv'),curves);r.rt.write_csv(ROOT/(a.tag+'real_training_quality.csv'),quality)
        (ROOT/(a.tag+'real_training_summary.json')).write_text(json.dumps(summaries,indent=2));print('REAL_TRAINING',summaries[-1],flush=True)
    assert sc.unchanged()


if __name__=='__main__':main()

"""Bank训练产生的R3目标仅作只读表达诊断；最终网络没有私有残差。"""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
import run as r
from neural import audit_parameters


def physical_target(delta):
    q=delta[:,6:10].clone();q[:,0]+=1;q=r.rt._normalize_quaternion(q)
    q=q*torch.where(q[:,:1]<0,-torch.ones_like(q[:,:1]),torch.ones_like(q[:,:1]))
    angle=2*torch.atan2(q[:,1:].norm(dim=1,keepdim=True),q[:,:1].clamp_min(1e-9))
    vector=q[:,1:]*angle/q[:,1:].norm(dim=1,keepdim=True).clamp_min(1e-9)
    return torch.cat([delta[:,:6],vector,delta[:,10:]],1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);args=p.parse_args();queue=json.loads(Path(args.queue).read_text())
    fixture=r.Fixture(r.rt.Scene('fr1_xyz'));summaries=[]
    for job in queue['jobs']:
        cfg=job['config'];seed=job['seed'];steps=job['steps'];d=r.ROOT/'results'/queue['tier']/'fr1_xyz'/cfg['id']/str(seed);d.mkdir(parents=True,exist_ok=True)
        if (d/'complete.json').exists():continue
        net=r.make(fixture,cfg,seed);r.rt.save_json(d/'parameter_audit.json',dict(pure_shared=True,parameters=audit_parameters(net)))
        targets={ratio:physical_target(fixture.teacher[ratio,'same_k']).detach() for ratio in r.rt.TRAIN}
        std=torch.cat(list(targets.values())).std(0).clamp_min(.05).detach();std[:3]=std[:3].clamp_min(.02)
        opt=torch.optim.Adam(net.parameters(),lr=.001,eps=1e-15,fused=True);curve=[];start=time.monotonic()
        for step in range(1,steps+1):
            ratio=r.rt.TRAIN[(step-1)%4];ids=fixture.sets[ratio,'same_k'];opt.zero_grad(set_to_none=True)
            predicted=physical_target(net(ratio,ids))
            loss=((predicted-targets[ratio])/std).square().mean();assert loss.isfinite();loss.backward();opt.step()
            if step%100 in [1,2,3,4] or step==steps:curve.append(dict(step=step,ratio=ratio,physical_mse=float(loss)))
        r.rt.write_csv(d/'supervision_curve.csv',curve)
        vals=r.evaluate(fixture,net)
        r.rt.write_csv(d/'validation_curve.csv',[dict(row,step=steps) for row in vals])
        torch.save(dict(config=cfg,seed=seed,step=steps,state_dict=net.state_dict(),optimizer=opt.state_dict()),d/f'step_{steps}.pt')
        info=dict(config=cfg,seed=seed,steps=steps,tier=queue['tier'],training='supervised R3physicaltarget only; photo not optimized',
            parameters=sum(p.numel() for p in net.parameters()),trainable_parameters=sum(p.numel() for p in net.parameters()),pure_shared=True,
            seconds=time.monotonic()-start,checkpoint=str(d/f'step_{steps}.pt'),source=fixture.manifest,
            validation_psnr=float(np.mean([v['psnr'] for v in vals])),validation_train=float(np.mean([v['psnr'] for v in vals if v['ratio'] in r.rt.TRAIN])),
            validation_unseen=float(np.mean([v['psnr'] for v in vals if v['ratio'] not in r.rt.TRAIN])),
            code_sha256={n:r.rt.sha(r.ROOT/n) for n in ['teacher_probe.py','run.py','neural.py']},teacher_is_readonly=True)
        r.rt.save_json(d/'complete.json',info);summaries.append(info)
        print('TEACHER',cfg['id'],info['validation_psnr'],float(loss),info['seconds'],flush=True)
        del net,opt;gc.collect();torch.cuda.empty_cache()
    r.rt.save_json(r.ROOT/(Path(args.queue).stem+'_complete.json'),summaries)


if __name__=='__main__':main()

"""原Adam、完整原始状态、真实Bank顺序；单步和跨ratio多步参数/Moment核验。"""
import copy
import json
import argparse
import torch
from baseline_profile import ROOT,r,fixture
from implementation import Execution
from parity import compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',default=['reference_repeat','fused','graph_reference','graph_fused'])
    p.add_argument('--common-vjp',action='store_true');p.add_argument('--tag',default='step_parity');args=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    sc,f,template,payload=fixture();rows=[];records=[]
    for method in args.methods:
        nets=[];opts=[];executions=[]
        for mode in ['reference','reference' if method=='reference_repeat' else method]:
            n=r.make(f,template.cfg,5100);n.load_state_dict(payload['state_dict'])
            opt=torch.optim.Adam(n.parameters(),lr=.001,eps=1e-15,fused=True);opt.load_state_dict(copy.deepcopy(payload['optimizer']))
            nets.append(n);opts.append(opt);executions.append(Execution(n,mode))
        for step in range(12):
            ratio=r.rt.TRAIN[step%4];ids=f.sets[ratio,'same_k'];view=sc.train[(step//4)%len(sc.train)];losses=[]
            if args.common_vjp:
                leaf=executions[0](ratio,ids).detach().requires_grad_();image=r.render(f,nets[0],ids,leaf,view)[0]
                vjp=torch.autograd.grad(r.rt.photo(image,view['gt']),leaf)[0].detach()
            for n,opt,ex in zip(nets,opts,executions):
                opt.zero_grad(set_to_none=True);out=ex(ratio,ids)
                loss=r.rt.photo(r.render(f,n,ids,out,view)[0],view['gt'])
                if args.common_vjp:out.backward(vjp)
                else:loss.backward()
                opt.step();losses.append(loss.detach())
            for (name,a),(other,b) in zip(nets[0].named_parameters(),nets[1].named_parameters()):
                compare(a,b,f'step{step+1}/parameter/{name}',rows,method,len(ids))
                for moment in ['step','exp_avg','exp_avg_sq']:
                    compare(opts[0].state[a][moment],opts[1].state[b][moment],f'step{step+1}/{moment}/{name}',rows,method,len(ids),True)
            compare(losses[0],losses[1],f'step{step+1}/photo',rows,method,len(ids))
            with torch.no_grad():compare(executions[0](.075,ids),executions[1](.075,ids),f'step{step+1}/next_output',rows,method,len(ids))
            failures=[v for v in rows if v['method']==method and not v['pass_']]
            records.append(dict(method=method,step=step+1,ratio=ratio,loss_reference=float(losses[0]),loss_candidate=float(losses[1]),cumulative_failures=len(failures)))
            print('STEP_PARITY',method,step+1,len(failures),flush=True)
        r.rt.write_csv(ROOT/(args.tag+'_tensors.csv'),rows);r.rt.write_csv(ROOT/(args.tag+'_losses.csv'),records)
    (ROOT/(args.tag+'_summary.json')).write_text(json.dumps(dict(rows=len(rows),common_real_photo_vjp=args.common_vjp,
        failures=[v for v in rows if not v['pass_']]),indent=2))
    if args.common_vjp:assert all(v['pass_'] for v in rows),'Matched real-photo VJP step parity failed'


if __name__=='__main__':main()

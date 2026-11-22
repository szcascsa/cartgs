"""解码Graph：两地图、输入VJP、多步原Adam与安全边界缓存失效。"""
import argparse
import copy
import json
import torch
from baseline_profile import ROOT,r,fixture
from implementation import Execution
from decoded_graph import DecodedExecution,decode,render_decoded
from parity import compare


NAMES=['delta','xyz','scale','rotation','dc','opacity']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scene',default='fr1_xyz');a=parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    sc,f,ref,payload=fixture(a.scene);candidate=r.make(f,ref.cfg,5100);candidate.load_state_dict(payload['state_dict'])
    base=Execution(ref,'reference');ex=DecodedExecution(candidate,sc.render_bases);rows=[]
    def original(ratio,ids):
        delta=base(ratio,ids);return (delta,*decode(delta,ids,(*ex.bases[:4],ex.bases[5])))
    def checks(aa,bb,prefix,K,backward=False):
        for name,x,y in zip(NAMES,aa,bb):compare(x,y,prefix+'/'+name,rows,'decoded_graph',K,backward)
    for ratio in r.rt.RATIOS:
        ids=f.sets[ratio,'same_k'];aa=original(ratio,ids);bb=ex(ratio,ids);checks(aa,bb,'output',len(ids))
        torch.manual_seed(91);vjp=tuple(torch.randn_like(v)*.001 for v in aa)
        ga=torch.autograd.grad(aa,tuple(ref.parameters()),vjp);gb=torch.autograd.grad(bb,tuple(candidate.parameters()),vjp)
        for (name,_),x,y in zip(ref.named_parameters(),ga,gb):compare(x,y,'gradient/'+name,rows,'decoded_graph',len(ids),True)
        with torch.no_grad():
            x=sc.render(ids,base(ratio,ids),sc.train[0])[0];y=ex.render(sc,ratio,ids,sc.train[0])
            compare(x,y,'renderer',rows,'decoded_graph',len(ids));compare(r.rt.photo(x,sc.train[0]['gt']),r.rt.photo(y,sc.train[0]['gt']),'photo',rows,'decoded_graph',len(ids))
    print('DIRECT',a.scene,len(rows),flush=True)
    features=f.features;old=(features.own,features.coords,features.anchor_scale);old_bases=ex.bases
    for K,all_live in [(0,True),(1,True),(17,True),(1,False),(17,False)]:
        ids=torch.arange(K,device='cuda')//2;grads=[]
        for optimized in [False,True]:
            full=tuple(v.detach().clone().requires_grad_(all_live or j>=2) for j,v in enumerate((*old,*old_bases)))
            features.own,features.coords,features.anchor_scale=full[:3];ex.bases=full[3:];ex.invalidate()
            inputs=full if all_live else full[2:]
            aa=ex(.075,ids) if optimized else original(.075,ids)
            torch.manual_seed(99);vjp=tuple(torch.randn_like(v)*.001 for v in aa)
            gg=torch.autograd.grad(aa,inputs,vjp,allow_unused=True)
            grads.append(tuple(torch.zeros_like(p) if g is None else g for p,g in zip(inputs,gg)))
        names=['physical19','coords','anchor_scale','base_xyz','base_scale','base_rotation','base_dc','base_rest_unused','base_opacity']
        for name,x,y in zip(names if all_live else names[2:],*grads):
            compare(x,y,('input_vjp_live/' if all_live else 'input_vjp_graph/')+name,rows,'decoded_graph',K,True)
    features.own,features.coords,features.anchor_scale=old;ex.bases=old_bases;ex.invalidate()
    # 极端合法属性只作数值gate，不将合成图像冒充场景质量。
    extreme=tuple(v.clone() for v in old_bases);extreme[1][:17]=torch.logspace(-6,2,17,device='cuda')[:,None]
    extreme[0][:17]*=1000;extreme[5][:17]=torch.linspace(-20,20,17,device='cuda')[:,None]
    ex.bases=extreme
    for K in [0,1,17]:
        ids=torch.arange(K,device='cuda');checks(original(.025,ids),ex(.025,ids),'boundary',K)
    ex.bases=old_bases;ex.invalidate()
    # 缓存生命周期：所有修改在无未完成反传的安全边界；调用方同步替换属性表。
    saved=None
    with torch.no_grad():
        for label in ['sameK_ids_ratio','inplace_physical','inplace_bases','append','prune_reorder','invalidate']:
            if label=='inplace_physical':features.own.add_(.001);features.coords.add_(.002)
            if label=='inplace_bases':ex.bases[0].add_(.001);features.anchor_scale.mul_(1.01)
            if label in ['append','prune_reorder']:
                N=len(features.own);idx=torch.cat([torch.arange(N,device='cuda'),torch.arange(3,device='cuda')]) if label=='append' else torch.arange(N-2,device='cuda').flip(0)
                features.own=features.own[idx];features.coords=features.coords[idx];features.anchor_scale=features.anchor_scale[idx]
                ex.bases=tuple(v[idx] for v in ex.bases)
            if label=='invalidate':ex.invalidate()
            ids=torch.arange(17,device='cuda').flip(0);aa=original(.125,ids);bb=ex(.125,ids);checks(aa,bb,'lifecycle/'+label,len(ids))
            if saved is not None:checks(saved[0],saved[1],'preserved_output/'+label,len(ids))
            saved=(tuple(v.clone() for v in bb),bb)
    # 恢复原fixture后再做真实photo VJP的12步状态等价。
    # 原地修改过旧表，重载fixture，避免以脏初始状态掩盖错误。
    sc,f,ref,payload=fixture(a.scene);candidate=r.make(f,ref.cfg,5100);candidate.load_state_dict(payload['state_dict'])
    base=Execution(ref,'reference');ex=DecodedExecution(candidate,sc.render_bases)
    opts=[]
    for net in [ref,candidate]:
        opt=torch.optim.Adam(net.parameters(),lr=.001,eps=1e-15,fused=True);opt.load_state_dict(copy.deepcopy(payload['optimizer']));opts.append(opt)
    steps=[]
    for step in range(12):
        ratio=r.rt.TRAIN[step%4];ids=f.sets[ratio,'same_k'];view=sc.train[step//4]
        # 候选先capture，避免原实现有活跃图时与共享输入发生流依赖。
        bb=ex(ratio,ids);aa=original(ratio,ids)
        leaves=tuple(v.detach().requires_grad_() for v in aa)
        loss=r.rt.photo(render_decoded(sc,ids,leaves,view),view['gt'])
        gg=torch.autograd.grad(loss,leaves,allow_unused=True);vjp=tuple(torch.zeros_like(p) if g is None else g for p,g in zip(leaves,gg))
        for net,opt,out in zip([ref,candidate],opts,[aa,bb]):
            opt.zero_grad(set_to_none=True);torch.autograd.backward(out,vjp);opt.step()
        for (name,x),(_,y) in zip(ref.named_parameters(),candidate.named_parameters()):
            compare(x,y,f'step{step+1}/parameter/{name}',steps,'decoded_graph',len(ids))
            for moment in ['step','exp_avg','exp_avg_sq']:compare(opts[0].state[x][moment],opts[1].state[y][moment],f'step{step+1}/{moment}/{name}',steps,'decoded_graph',len(ids),True)
        with torch.no_grad():checks(original(.075,ids),ex(.075,ids),'next_step_output',len(ids))
    r.rt.write_csv(ROOT/f'decoded_direct_{a.scene}.csv',rows);r.rt.write_csv(ROOT/f'decoded_steps_{a.scene}.csv',steps)
    summary=dict(scene=a.scene,direct_rows=len(rows),step_rows=len(steps),failures=[v for v in rows+steps if not v['pass_']],step_max_abs=max(v['max_abs'] for v in steps),full_master_joint=False)
    (ROOT/f'decoded_validation_{a.scene}.json').write_text(json.dumps(summary,indent=2));print('VALIDATION',summary,flush=True)
    assert not summary['failures']


if __name__=='__main__':main()

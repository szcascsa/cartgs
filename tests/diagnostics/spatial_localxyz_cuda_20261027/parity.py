"""真实非零Checkpoint：逐张量、所有网络梯度、输入VJP与Renderer核验。"""
import argparse
import json
import torch
from baseline_profile import fixture,ROOT,r
from implementation import Execution
from kernels import Film


def compare(a,b,name,rows,method,K,backward=False):
    a=a.detach();b=b.detach();assert a.shape==b.shape,(name,a.shape,b.shape)
    diff=(a-b).abs();atol,rtol=(1e-5,1e-4) if backward else (1e-6,1e-5)
    ok=bool(torch.all(diff<=atol+rtol*a.abs()))
    rows.append(dict(method=method,K=K,tensor=name,backward=backward,atol=atol,rtol=rtol,pass_=ok,
        max_abs=float(diff.max()) if diff.numel() else 0.,rms=float(diff.square().mean().sqrt()) if diff.numel() else 0.,
        max_relative=float((diff/a.abs().clamp_min(1e-12)).max()) if diff.numel() else 0.))
    return ok


def intermediates(ex,ratio,ids):
    n=ex.network;x=ex.pack(ids);scale=n.features.anchor_scale[ids];condition=ex.condition(ratio)
    modulation=n.ratio(condition);g1,b1,g2,b2=modulation.chunk(4,1);z1=n.encode(x)
    h1=Film.apply(z1,g1,b1) if ex.fused else torch.relu((1+g1)*z1+b1)
    z2=n.body(h1);h2=Film.apply(z2,g2,b2) if ex.fused else torch.relu((1+g2)*z2+b2)
    return dict(input55=x,ratio_modulation=modulation,encoder_linear=z1,encoder_film=h1,body_linear=z2,body_film=h2,
        **{name+'_head':head(h2) for name,head in n.heads.items()},delta=ex(ratio,ids))


def attributes(sc,ids,res):
    xyz,scale,rot,dc,rest,opacity=[value[ids] for value in sc.render_bases]
    ident=torch.zeros_like(res[:,6:10]);ident[:,0]=1
    return dict(xyz=xyz+res[:,:3],scale=scale*res[:,3:6].exp(),rotation=r.rt._normalize_quaternion(
        r.rt._multiply_quaternion(r.rt._normalize_quaternion(ident+res[:,6:10]),rot)),
        dc=dc+res[:,11:14].unsqueeze(1),sh_rest=rest,opacity=(opacity+res[:,10:11]).sigmoid())


def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',default=['fused','graph_reference','graph_fused']);p.add_argument('--scene',default='fr1_xyz');a=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    sc,f,ref,payload=fixture(a.scene);rows=[];suffix='' if a.scene=='fr1_xyz' else '_'+a.scene
    for method in a.methods:
        candidate=r.make(f,ref.cfg,5100);candidate.load_state_dict(payload['state_dict']);ex=Execution(candidate,method);base=Execution(ref)
        for ratio in r.rt.RATIOS:
            ids=f.sets[ratio,'same_k'];K=len(ids)
            with torch.no_grad():
                aa=intermediates(base,ratio,ids);bb=intermediates(ex,ratio,ids)
                for name in aa:compare(aa[name],bb[name],name,rows,method,K)
            # make_graphed_callables需在训练grad模式首次建图；上述no_grad调用会生成可反传包装。
            torch.manual_seed(123);vjp=torch.randn(K,14,device='cuda')*.01
            ref.zero_grad(set_to_none=True);candidate.zero_grad(set_to_none=True)
            base(ratio,ids).backward(vjp);ex(ratio,ids).backward(vjp)
            for (name,x),(other,y) in zip(ref.named_parameters(),candidate.named_parameters()):
                assert name==other;compare(x.grad,y.grad,'parameter_gradient/'+name,rows,method,K,True)
            with torch.no_grad():
                at=attributes(sc,ids,aa['delta']);bt=attributes(sc,ids,bb['delta'])
                for name in at:compare(at[name],bt[name],'attribute/'+name,rows,method,K)
                ra=sc.render(ids,aa['delta'],sc.train[0])[0];rb=sc.render(ids,bb['delta'],sc.train[0])[0]
                compare(ra,rb,'renderer',rows,method,K);compare(r.rt.photo(ra,sc.train[0]['gt']),r.rt.photo(rb,sc.train[0]['gt']),'photo',rows,method,K)
            print('PARITY',method,K,sum(not row['pass_'] for row in rows),flush=True)
        # 真正输入叶子梯度；原Features冻结行为不变，该gate单独检查算子的数学VJP。
        old=(f.features.own,f.features.coords,f.features.anchor_scale)
        for K in [0,1,68,1025]:
            ids=torch.arange(K,device='cuda');vjp=torch.randn(K,14,device='cuda')*.001;grads=[]
            for execution in [base,ex]:
                values=tuple(t.detach().clone().requires_grad_() for t in old)
                f.features.own,f.features.coords,f.features.anchor_scale=values
                value=execution(.075,ids);grads.append(torch.autograd.grad(value,values,vjp,allow_unused=False))
            for name,x,y in zip(['physical19','world_coordinates','anchor_scale'],*grads):compare(x,y,'input_gradient/'+name,rows,method,K,True)
        f.features.own,f.features.coords,f.features.anchor_scale=old
        r.rt.write_csv(ROOT/('parity_'+('_'.join(a.methods) if a.methods!=['fused','graph_reference','graph_fused'] else 'tensors')+suffix+'.csv'),rows)
    failures=[v for v in rows if not v['pass_']];(ROOT/('parity_'+('_'.join(a.methods) if a.methods!=['fused','graph_reference','graph_fused'] else 'summary')+suffix+'.json')).write_text(json.dumps(dict(rows=len(rows),failures=failures),indent=2))
    print('PARITY_COMPLETE',len(rows),'FAILURES',len(failures),flush=True)
    assert not failures,'Direct parity failed; candidate must not enter accepted benchmarks'


if __name__=='__main__':main()

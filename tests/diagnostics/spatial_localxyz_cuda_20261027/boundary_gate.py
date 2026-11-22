"""FP32、合法Quaternion和极端正Scale的输入/VJP边界；不渲染巨大合成点。"""
import json
import argparse
import torch
from baseline_profile import ROOT,r,fixture
from implementation import Execution
from parity import compare


def main():
    torch.backends.cuda.matmul.allow_tf32=False
    p=argparse.ArgumentParser();p.add_argument('--method',default='graph_all_fused');args=p.parse_args()
    sc,f,net,payload=fixture();ref=Execution(net);opt=Execution(net,args.method);rows=[];torch.manual_seed(77);n=17
    xyz=torch.randn(n,3,device='cuda')*1000
    center=sc.g.get_xyz.median(0).values;spread=(sc.g.get_xyz-center).square().mean(0).sqrt().clamp_min(.1)
    coords=(xyz-center)/spread
    scales=torch.logspace(-6,2,n*3,device='cuda').reshape(n,3)
    q=torch.randn(n,4,device='cuda');q=q/q.norm(dim=1,keepdim=True);w,x,y,z=q.unbind(1)
    rot=torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),
        2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],1).reshape(-1,3,3)
    cov=(rot*scales[:,None,:].square())@rot.transpose(1,2);cov=cov/cov.diagonal(dim1=1,dim2=2).sum(1)[:,None,None]
    shape=cov[:,[0,1,2,0,0,1],[0,1,2,1,2,2]]
    physical=torch.cat([coords,torch.asinh(torch.randn(n,3,device='cuda')*200),scales.log(),torch.randn(n,3,device='cuda'),
        torch.linspace(-20,20,n,device='cuda')[:,None],shape],1)
    own=((physical-f.features.mean)/f.features.std).clamp(-15,15)
    anchors=torch.tensor([.025,.05,.1],device='cuda').repeat(6)[:n,None]
    initial=tuple(v.detach().requires_grad_() for v in [own,coords,anchors]);f.features.own,f.features.coords,f.features.anchor_scale=initial
    with torch.no_grad():
        for ids in [torch.empty(0,dtype=torch.long,device='cuda'),torch.tensor([0],device='cuda'),torch.arange(n,device='cuda'),torch.tensor([2,2,5],device='cuda')]:
            for ratio in [.01,.025,.075,.125,.15]:compare(ref(ratio,ids),opt(ratio,ids),'extreme_delta_'+str(ratio),rows,args.method,len(ids))
    for ids in [torch.tensor([0],device='cuda'),torch.arange(n,device='cuda'),torch.tensor([2,2,5],device='cuda')]:
        vjp=torch.randn(len(ids),14,device='cuda')*.001
        a=torch.autograd.grad(ref(.075,ids),initial,vjp);b=torch.autograd.grad(opt(.075,ids),initial,vjp)
        for name,aa,bb in zip(['physical','coords','anchor_scale'],a,b):compare(aa,bb,'extreme_gradient_'+name,rows,args.method,len(ids),True)
    prefix='' if args.method=='graph_all_fused' else args.method+'_'
    r.rt.write_csv(ROOT/(prefix+'boundary_gate.csv'),rows)
    failures=[v for v in rows if not v['pass_']]
    (ROOT/(prefix+'boundary_gate.json')).write_text(json.dumps(dict(rows=len(rows),failures=failures,scale_range=[1e-6,100.],world_xyz_order=1000,
        opacity_logit_range=[-20,20],quaternion_unit=True,scope='network input/gradient boundary; not synthetic renderer quality'),indent=2))
    assert not failures,failures
    print('BOUNDARY_GATE_PASS',len(rows),flush=True)


if __name__=='__main__':main()

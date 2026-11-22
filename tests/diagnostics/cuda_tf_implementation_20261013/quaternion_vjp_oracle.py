"""Legacy FP32 chain at a small residual quaternion; diagnostic, no optimization."""
import torch
import numpy as np
torch.manual_seed(20261015)
v=[torch.randn(1067,4 if i in (2,5) else 3,device='cuda') for i in range(6)]
v[5][7]=torch.tensor([-1,1e-13,0,0],device='cuda')
v[5].requires_grad_();v[2].requires_grad_()
raw=v[5]+torch.tensor([1,0,0,0],device='cuda')
a=raw/(raw*raw).sum(1,keepdim=True).sqrt().clamp_min(1e-12)
b=v[2]
w=a[:,:1]*b[:,:1]-(a[:,1:]*b[:,1:]).sum(1,keepdim=True)
z=torch.cat((w,a[:,:1]*b[:,1:]+b[:,:1]*a[:,1:]+torch.cross(a[:,1:],b[:,1:],dim=1)),1)
norm=(z*z).sum(1,keepdim=True).sqrt()
denom=norm.clamp_min(1e-12)
y=z/denom
gs=[torch.randn_like(t) for t in (v[0],v[1],y)]
gz,ga,dq,gn=torch.autograd.grad(y,(z,a,v[5],denom),gs[2],retain_graph=True)
for label,t in [('a',a),('b',b),('z',z),('norm',norm),('gy',gs[2]),('gz',gz),('ga',ga),('dq',dq),('gn',gn)]:
    print(label,[float(x).hex() for x in t[7].detach().cpu().flatten()])
terms=-gs[2]*((z/denom)/denom)
for label,t in [('terms',terms),('sum_terms',terms.sum(1,keepdim=True))]:
    print(label,[float(x).hex() for x in t[7].detach().cpu().flatten()])
direct=gs[2]/denom
square_edge=(gn/(2*norm))*z
for label,t in [('grouped_edges',direct+(square_edge+square_edge)),
                ('separate_edges',(direct+square_edge)+square_edge)]:
    print(label,[float(x).hex() for x in t[7].detach().cpu().flatten()])
q=z[7].detach().cpu().numpy();t=terms[7].detach().cpu().numpy()
for label,vv in [('sum_serial',((q[0]*q[0]+q[1]*q[1])+q[2]*q[2])+q[3]*q[3]),
                 ('sum_tree',(q[0]*q[0]+q[2]*q[2])+(q[1]*q[1]+q[3]*q[3])),
                 ('sum_adjacent',(q[0]*q[0]+q[1]*q[1])+(q[2]*q[2]+q[3]*q[3])),
                 ('gn_serial',((t[0]+t[1])+t[2])+t[3]),
                 ('gn_tree',(t[0]+t[2])+(t[1]+t[3])),
                 ('gn_adjacent',(t[0]+t[1])+(t[2]+t[3]))]:
    print(label,float(vv).hex())

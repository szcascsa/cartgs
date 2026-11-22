"""确定ATen quaternion reduction与cross舍入，不调整任何容差。"""
import json
from pathlib import Path
import torch
R=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022');rows=[]
for label in ['lod_0.010000','lod_0.050000','lod_0.100000','lod_0.150000']:
    a=torch.jit.load(str(R/'reduction_v1_failure/legacy_a'/(label+'_parity.pt')),map_location='cuda').state_dict()
    delta=a['residual'];raw=delta[:,6:10].clone();raw[:,0]+=1;sq=raw*raw
    norm=sq.sum(1);balanced=(sq[:,0]+sq[:,1])+(sq[:,2]+sq[:,3]);linear=((sq[:,0]+sq[:,1])+sq[:,2])+sq[:,3]
    q=raw/norm.sqrt()[:,None];b=torch.nn.functional.normalize(a['parameter_before_5'])[a['selected_ids']];u=q[:,1:];v=b[:,1:]
    cross=torch.cross(u,v,dim=1);separate=torch.stack([u[:,1]*v[:,2]-u[:,2]*v[:,1],u[:,2]*v[:,0]-u[:,0]*v[:,2],u[:,0]*v[:,1]-u[:,1]*v[:,0]],1)
    fused=torch.stack([torch.addcmul(-u[:,2]*v[:,1],u[:,1],v[:,2]),torch.addcmul(-u[:,0]*v[:,2],u[:,2],v[:,0]),torch.addcmul(-u[:,1]*v[:,0],u[:,0],v[:,1])],1)
    triple=u*v;sum3=triple.sum(1);sum3manual=(triple[:,0]+triple[:,2])+triple[:,1]
    def diff(name,x,y):rows.append(dict(case=label,probe=name,unequal=int((x!=y).sum()),max_abs=float((x-y).abs().max())))
    diff('norm4_balanced',norm,balanced);diff('norm4_linear',norm,linear);diff('sum3_tree',sum3,sum3manual)
    diff('cross_separate',cross,separate);diff('cross_fma',cross,fused)
    opposite=torch.stack([torch.addcmul(u[:,1]*v[:,2],u[:,2],v[:,1],value=-1),torch.addcmul(u[:,2]*v[:,0],u[:,0],v[:,2],value=-1),torch.addcmul(u[:,0]*v[:,1],u[:,1],v[:,0],value=-1)],1)
    diff('cross_fma_opposite',cross,opposite)
(R/'quaternion_fidelity.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))

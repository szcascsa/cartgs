"""导出真实 checkpoint 原始TF图；只读，记录node及saved tensor shape。"""
import json,sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
dest=Path(__file__).resolve().parent
sys.path.insert(0,str(dest.parents[2]/'python'))
from transform_field import load_transform_field,_multiply_quaternion
inventory=json.loads((dest/'inventory.json').read_text())
model=load_transform_field(inventory['checkpoint'],'cuda')
d=torch.tensor(np.load(dest/'input.npy')[:257],device='cuda')
x=d[:,:3].contiguous().requires_grad_();s=d[:,3:6].contiguous().requires_grad_();q=d[:,6:10].contiguous().requires_grad_()
coords=torch.cat((2.0*(x-model.grid.aabb_min)/(model.grid.aabb_max-model.grid.aabb_min)-1.0,x.new_full((len(x),1),0.05)),1)
features=[];intermediates={}
for level in range(len(model.grid.multires)):
    product=None
    for name,(u,v) in zip(model.grid.plane_names,model.grid.pairs):
        grid=torch.stack((coords.select(1,u),coords.select(1,v)),1).view(1,1,len(x),2)
        sample=F.grid_sample(getattr(model.grid,f'L{level}_{name}'),grid,mode='bilinear',padding_mode='border',align_corners=True).view(model.grid.feature_dim,len(x)).transpose(0,1)
        intermediates[f'sample_L{level}_{name}']=dict(shape=list(sample.shape),stride=list(sample.stride()),contiguous=sample.is_contiguous())
        product=sample if product is None else product*sample
    intermediates[f'product_L{level}']=dict(shape=list(product.shape),stride=list(product.stride()),contiguous=product.is_contiguous())
    features.append(product)
f=torch.cat(features,1);h=model.feature_out(f)
dx=model.position_out(F.relu(model.position_hidden(F.relu(h))))
ds=model.scale_out(F.relu(model.scale_hidden(F.relu(h))))
dq=model.rotation_out(F.relu(model.rotation_hidden(F.relu(h))))
for name,value in [('coordinates',coords),('feature',f),('hidden',h),('delta_xyz',dx),('delta_log_scale',ds),('delta_rotation_raw',dq)]:intermediates[name]=dict(shape=list(value.shape),stride=list(value.stride()),contiguous=value.is_contiguous(),dtype=str(value.dtype))
norm=lambda a:a/torch.sqrt((a*a).sum(1,keepdim=True)).clamp_min(1e-12)
identity=torch.zeros_like(dq);identity[:,0]=1
y=(x+dx,s*ds.exp(),norm(_multiply_quaternion(norm(identity+dq),q)))
loss=sum(v.square().mean() for v in y)
nodes=[];edges=[];seen={};queue=[loss.grad_fn];refs=[]
while queue:
    node=queue.pop()
    if node is None or node in seen:continue
    i=len(nodes);seen[node]=i;refs.append(node)
    saved={}
    for name in dir(node):
        if not name.startswith('_saved_'):continue
        try:value=getattr(node,name)
        except RuntimeError:continue
        if isinstance(value,torch.Tensor):saved[name]=dict(shape=list(value.shape),dtype=str(value.dtype),stride=list(value.stride()))
    nodes.append(dict(id=i,type=type(node).__name__,saved_tensors=saved))
    queue.extend(n for n,_ in node.next_functions if n is not None)
for node,i in seen.items():
    for target,_ in node.next_functions:
        if target in seen:edges.append([i,seen[target]])
(dest/'autograd_graph.json').write_text(json.dumps(dict(nodes=nodes,edges=edges,type_counts=dict(Counter(n['type'] for n in nodes)),intermediates=intermediates,tf32_matmul=torch.backends.cuda.matmul.allow_tf32),indent=2))
print('autograd_nodes',len(nodes))

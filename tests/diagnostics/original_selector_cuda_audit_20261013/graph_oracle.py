"""Native 梯度 archive 的同噪声 Python 对照和 autograd 保存张量统计。"""
import json,sys
from collections import Counter
from pathlib import Path
import torch
dest=Path(__file__).resolve().parent;repo=dest.parents[2];sys.path.insert(0,str(repo/'python'))
from selector import load_selector
checkpoint=json.loads((dest/'inventory.json').read_text())['checkpoint']
model=load_selector(checkpoint+'/selector.pt','cuda')
state=torch.jit.load(str(dest/'native_gradient_oracle.pt')).state_dict()
x,q,s,alpha=[state[k].cuda().detach().requires_grad_(True) for k in ['x','q','s','alpha']]
noise=state['noise'].cuda()
saved=[]
def pack(t):
    saved.append(dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),bytes=t.numel()*t.element_size(),storage_bytes=t.untyped_storage().nbytes(),storage=t.untyped_storage().data_ptr()))
    return t
with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
    ratio=model.time_emd(x.new_full((1,1),.05)).expand(x.shape[0],32)
    logits=model.soft_net(torch.cat([model.pos_emd(x),model.rotation_emd(q),model.scale_emd(s),ratio],1))
    soft=((logits+noise)/1.0).softmax(-1)
    hard=torch.zeros_like(logits).scatter_(-1,soft.max(-1,True).indices,1)
    mask=(hard-soft.detach()+soft)[:,1]
    indices=torch.nonzero(mask.detach()>0).flatten()
    selected_opacity=mask[indices,None]*alpha[indices]
    loss=selected_opacity.mean()+.01*(logits.softmax(-1)[:,1].sum()-.05*x.shape[0]).abs()/(.05*x.shape[0])
nodes=Counter();seen=set();queue=[loss.grad_fn]
while queue:
    node=queue.pop()
    if node is None or node in seen:continue
    seen.add(node);nodes[type(node).__name__]+=1
    queue.extend(n for n,_ in node.next_functions)
loss.backward()
checks=[]
for i,(name,p) in enumerate(model.named_parameters()):
    reference=state['grad_'+str(i)].cuda();error=(p.grad-reference).abs().max().item();den=reference.abs().max().clamp_min(1e-8).item()
    checks.append(dict(name=name,max_abs=error,scaled_error=error/den,passed=torch.allclose(p.grad,reference,atol=2e-6,rtol=2e-4)))
for name,t in zip(['x','q','s','alpha'],[x,q,s,alpha]):
    reference=state[name+'_grad'].cuda();error=(t.grad-reference).abs().max().item()
    checks.append(dict(name=name,max_abs=error,passed=torch.allclose(t.grad,reference,atol=2e-6,rtol=2e-4)))
stores={v['storage']:v['storage_bytes'] for v in saved}
payload=dict(N=x.shape[0],selected=indices.numel(),fixed_noise_hard_exact=torch.equal(mask.cpu(),state['hard']),checks=checks,all_passed=all(v['passed'] for v in checks),autograd_nodes=dict(nodes),total_nodes=sum(nodes.values()),saved_tensor_references=len(saved),unique_saved_storage=len(stores),unique_saved_storage_bytes=sum(stores.values()),saved_tensors=saved,
    limit='257 actual checkpoint rows; native structural opacity VJP + ratio oracle, not full photometric numerical parity; native renderer gradients are separately verified')
(dest/'gradient_parity_and_graph.json').write_text(json.dumps(payload,indent=2))
assert payload['all_passed'] and payload['fixed_noise_hard_exact']
print(json.dumps({k:payload[k] for k in ['N','selected','all_passed','total_nodes','saved_tensor_references','unique_saved_storage_bytes']},indent=2))

"""同函数末层拆分，验证非活跃节点的真实Adam差异。"""
import torch
import run as r

f=r.Fixture(r.rt.Scene('fr1_xyz'));base=dict(kind='self',width=64,bands=6,film=True,knots=True)
a=r.make(f,dict(base,id='Dense'),5100);b=r.make(f,dict(base,id='Sparse',sparse_node_clock=True),5100)
with torch.no_grad():
    for name,head in a.heads.items():
        torch.nn.init.normal_(head[-1].weight,std=.01)
        old=head[-1];width=old.out_features//4
        for j,layer in enumerate(b.heads[name][-1].layers):
            layer.weight.copy_(old.weight[j*width:(j+1)*width]);layer.bias.copy_(old.bias[j*width:(j+1)*width])
    errors=[float((a(ratio,f.sets[ratio,'same_k'])-b(ratio,f.sets[ratio,'same_k'])).abs().max()) for ratio in r.rt.RATIOS]
assert max(errors)<2e-6
optimizers=[torch.optim.Adam(net.parameters(),lr=.001,eps=1e-15,fused=True) for net in [a,b]]
changes=[]
for index,ratio in enumerate([.01,.05]):
    for net,opt in zip([a,b],optimizers):
        opt.zero_grad(set_to_none=True);ids=f.sets[ratio,'same_k'];loss=r.rt.photo(r.render(f,net,ids,net(ratio,ids),f.scene.train[0])[0],f.scene.train[0]['gt']);loss.backward()
        opt.step()
    dense=a.heads['xyz'][-1].weight[:3].detach().clone();sparse=b.heads['xyz'][-1].layers[0].weight.detach().clone()
    changes.append((dense,sparse))
value=dict(nonzero_function_max_error=max(errors),dense_inactive_node_delta=float((changes[1][0]-changes[0][0]).norm()),
    sparse_inactive_node_delta=float((changes[1][1]-changes[0][1]).norm()),
    dense_parameter_clock=float(optimizers[0].state[a.heads['xyz'][-1].weight]['step']),
    sparse_node0_clock=float(optimizers[1].state[b.heads['xyz'][-1].layers[0].weight]['step']),
    actual_photo=True,no_dummy_steps=True)
assert value['sparse_inactive_node_delta']==0 and value['dense_inactive_node_delta']>0
r.rt.save_json(r.ROOT/'node_clock_gate.json',value);print('NODE_CLOCK_GATE',value,flush=True)

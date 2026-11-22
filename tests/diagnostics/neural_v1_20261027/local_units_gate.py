"""Anchor单位输出的逐点梯度预条件核查，不将其等同于更大网络。"""
import torch
import run as r

f=r.Fixture(r.rt.Scene('fr1_xyz'));cfg=dict(id='World',kind='self',width=64,bands=6,film=True)
world=r.make(f,cfg,5100);local=r.make(f,dict(cfg,id='Local',output='local'),5100)
scales=f.features.anchor_scale.flatten();unique,counts=torch.unique(scales,return_counts=True);rows=[]
for ratio in [.01,.05]:
    ids=f.sets[ratio,'same_k'];view=f.scene.train[0];leaf=torch.zeros(len(ids),14,device='cuda',requires_grad=True)
    loss=r.rt.photo(f.scene.render(ids,leaf,view)[0],view['gt']);vjp=torch.autograd.grad(loss,leaf)[0]
    gradients={}
    for tag,net in [('world',world),('local',local)]:
        cache={}
        hook=net.heads['xyz'][-1].register_forward_hook(lambda module,args,value:cache.update(phi=args[0].detach()))
        output=net(ratio,ids);photo=r.rt.photo(r.render(f,net,ids,output,view)[0],view['gt'])
        grad=torch.autograd.grad(photo,net.heads['xyz'][-1].weight)[0]
        factor=scales[ids,None] if tag=='local' else 1.
        expected=(vjp[:,:3]*factor).T@cache['phi']
        relative=float((grad-expected).norm()/expected.norm().clamp_min(1e-15));assert relative<.003,(tag,relative)
        gradients[tag]=grad;hook.remove()
        rows.append(dict(ratio=ratio,tag=tag,gradient_norm=float(grad.norm()),analytic_vjp_relative_error=relative,
            scale_min=float(scales[ids].min()),scale_max=float(scales[ids].max())))
r.rt.save_json(r.ROOT/'local_units_gate.json',dict(anchor_reference_scales=unique.cpu().tolist(),gaussian_counts=counts.cpu().tolist(),
    rows=rows,meaning='deltaXYZ_i = anchorScale_i * neuralOutput_i; deterministic geometry-based preconditioning; no private trainable parameter',
    caveat='if scales vary perpoint this is not simply one globally rescaled learning rate'))
assert f.scene.unchanged();print('LOCAL_UNITS_GATE_PASS',rows,flush=True)

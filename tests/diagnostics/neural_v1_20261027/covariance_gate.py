"""复用Renderer现有covariance接口的零补偿与梯度gate。"""
import torch
import run as r

sc=r.rt.Scene('fr1_xyz');f=r.Fixture(sc);cfg=dict(id='AffineCovariance',kind='self',width=64,bands=6,film=True,output='affine')
net=r.make(f,cfg,5100);rows=[]
for ratio in [.01,.05]:
    ids=f.sets[ratio,'same_k'];v=sc.train[0]
    a=r.render(f,net,ids,net(ratio,ids),v)[0]
    b=sc.render(ids,torch.zeros(len(ids),14,device='cuda'),v)[0]
    error=float((a-b).abs().max());mean=float((a-b).abs().mean());assert mean<2e-5 and error<5e-3,(error,mean)
    net.zero_grad(set_to_none=True);r.rt.photo(a,v['gt']).backward()
    grad=sum(float(p.grad.norm()) for name,p in net.named_parameters() if 'covariance' in name and p.grad is not None)
    assert grad>0
    rows.append(dict(ratio=ratio,zero_output_render_max_error=error,zero_output_render_mean_error=mean,covariance_real_photo_grad=grad))
assert sc.unchanged();r.rt.save_json(r.ROOT/'covariance_gate.json',rows);print('COVARIANCE_GATES_PASS',rows,flush=True)

"""XYZ-only逐步加回Scale和Opacity；不恢复SH、Rotation或局部Offset。"""
import gc
import importlib.util
import json
import math
from pathlib import Path
import torch
from torch import nn
import matrix_runtime as rt

ROOT=Path('/home/prmi/zwj/s00-xyz-scale-opacity-20261026')
XYZ=Path('/home/prmi/zwj/s00-xyz-only-20261026')
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
assert rt.sha(XYZ/'run.py')==PROTOCOL['xyz_run_sha256']
spec=importlib.util.spec_from_file_location('xyz_experiment',XYZ/'run.py')
xyz=importlib.util.module_from_spec(spec);spec.loader.exec_module(xyz)
rt.ROOT=ROOT


def config():
    cfg=xyz.config();cfg.update(id='GlobalXYZScaleOpacityE16Split4',gaussian_input='global_xyz_scale_opacity')
    return cfg


class Network(xyz.Network):
    def __init__(self,scene,cfg,seed):
        super().__init__(scene,cfg,seed)
        encoder=nn.Linear(8,32,device='cuda')
        with torch.no_grad():
            encoder.weight[:,:3].copy_(self.encoder.weight[:,:3])
            encoder.weight[:,3:7].zero_()
            encoder.weight[:,7].copy_(self.encoder.weight[:,3])
            encoder.bias.copy_(self.encoder.bias)
        self.encoder=encoder
        for branch in self.compensation.branches.values():
            old=branch[0];first=nn.Linear(56,32,device='cuda')
            with torch.no_grad():
                first.weight[:,:3].copy_(old.weight[:,:3])
                first.weight[:,3:7].zero_()
                first.weight[:,7].copy_(old.weight[:,3])
                first.weight[:,8:].copy_(old.weight[:,4:])
                first.bias.copy_(old.bias)
            branch[0]=first

    def raw(self,ratio,ids=None):
        g=self.scene.g
        values=[g.get_xyz,g.get_scaling,g.get_opacity]
        if ids is not None:values=[value[ids] for value in values]
        return torch.cat([*values,torch.full((len(values[0]),1),math.log(ratio),device=values[0].device)],1)


rt.Network=Network


def gate(scene,seed):
    net=Network(scene,config(),seed);ref=xyz.Network(scene,xyz.config(),seed)
    errors=[]
    for ratio in rt.RATIOS:
        p,h,_=net(ratio);p0,h0,_=ref(ratio)
        ids=torch.nonzero(p>scene.uniform).flatten()
        assert torch.equal(p,p0)
        assert torch.allclose(h,h0,atol=2e-6,rtol=2e-6)
        assert torch.equal(net.compensate(h,ids,ratio),ref.compensate(h0,ids,ratio))
        errors.append(float((h-h0).abs().max()))
    # 只暴露允许的三个Gaussian属性，禁止SH、Rotation和旧Offset派生输入。
    class Allowed:
        def __init__(self,g):self.get_xyz=g.get_xyz;self.get_scaling=g.get_scaling;self.get_opacity=g.get_opacity
    original_g,original_attrs=scene.g,scene.attrs
    p,h,_=net(.05);ids=torch.nonzero(p>scene.uniform).flatten();delta=net.compensate(h,ids,.05)
    try:
        scene.g=Allowed(original_g);scene.attrs=torch.full_like(original_attrs,float('nan'))
        p1,h1,_=net(.05)
        assert torch.equal(p,p1) and torch.equal(h,h1)
        assert torch.equal(delta,net.compensate(h1,ids,.05))
    finally:scene.g,scene.attrs=original_g,original_attrs
    with torch.no_grad():
        for branch in net.compensation.branches.values():branch[-1].weight.fill_(.01)
    p,h,_=net(.05);ids=torch.nonzero(p>scene.uniform).flatten()
    net.compensate(h,ids,.05).square().mean().backward()
    assert net.encoder.weight.grad[:,3:7].norm()>0
    assert all(branch[0].weight.grad[:,3:7].norm()>0 for branch in net.compensation.branches.values())
    assert net.extra_tables[1].grad.norm()>0
    assert all(net.extra_tables[j].grad is None for j in [0,2,3])
    assert scene.unchanged()
    rt.save_json(ROOT/f'correctness_{seed}.json',dict(seed=seed,input_width=8,head_input_width=56,
        initial_probability_exact=True,initial_compensation_exact=True,max_hidden_error=max(errors),
        allowed_attribute_proxy=True,forbidden_derived_attribute_nan=True,new_attribute_gradient=True,
        inactive_ratio_gradient_none=True,master_frozen=True,
        head_parameters=sum(p.numel() for p in net.compensation.parameters()),
        total_parameters=sum(p.numel() for p in net.parameters()),
        gradient_probe='temporary nonzero final output projection, not actual step1 hidden gradient'))
    del net,ref;gc.collect();torch.cuda.empty_cache()


def main():
    scene=rt.Scene('fr1_xyz')
    assert json.loads((ROOT/'results/fr1_xyz/scene_manifest.json').read_text())==json.loads((XYZ/'results/fr1_xyz/scene_manifest.json').read_text())
    assert rt.sha(ROOT/'matrix_runtime.py')==PROTOCOL['runtime_sha256']
    assert rt.sha(Path(PROTOCOL['renderer_binary']))==PROTOCOL['renderer_sha256']
    attrs=torch.cat([scene.g.get_xyz,scene.g.get_scaling,scene.g.get_opacity],1).detach()
    rt.save_json(ROOT/'input_metadata.json',dict(layout='world XYZ3, activated Scale3, sigmoid Opacity1; actual logratio appended',
        shape=list(attrs.shape),min=attrs.min(0).values.cpu().tolist(),max=attrs.max(0).values.cpu().tolist(),
        mean=attrs.mean(0).cpu().tolist(),std=attrs.std(0).cpu().tolist()))
    rt.np.save(ROOT/'frozen_attributes.npy',attrs.cpu().numpy())
    for seed in PROTOCOL['seeds']:gate(scene,seed)
    runs=[]
    for seed in PROTOCOL['seeds']:
        cfg=config();d=ROOT/'results/fr1_xyz'/cfg['id']/str(seed)
        info=rt.train(scene,cfg,seed,12000,d,checkpoints=[4000,8000,12000])
        net=Network(scene,cfg,seed);net.load_state_dict(torch.load(info['checkpoint'],map_location='cuda')['state_dict'])
        for split in ['bank','validation','test']:
            path=d/(split+'_full.csv')
            if not path.exists():
                rows=rt.evaluate(scene,net.eval(),split,full_metrics=True,policies=('same_k','cached'))
                for row in rows:row.update(config=cfg['id'],seed=seed)
                rt.write_csv(path,rows)
        with torch.no_grad():
            rt.save_json(d/'attribute_weight_updates.json',dict(
                encoder_scale_norm=float(net.encoder.weight[:,3:6].norm()),encoder_opacity_norm=float(net.encoder.weight[:,6].norm()),
                heads={name:dict(scale_norm=float(branch[0].weight[:,3:6].norm()),opacity_norm=float(branch[0].weight[:,6].norm()))
                    for name,branch in net.compensation.branches.items()}))
        assert scene.unchanged()
        runs.append(info);rt.save_json(ROOT/'runs.json',runs)
        print('COMPLETE',seed,flush=True)
        del net;gc.collect();torch.cuda.empty_cache()
    rt.save_json(ROOT/'status.json',dict(stage='COMPLETE',models=2))


if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'run.exit').write_text('1');raise
    (ROOT/'run.exit').write_text('0')

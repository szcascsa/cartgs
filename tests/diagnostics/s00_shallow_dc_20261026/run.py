"""浅层Encoder和四Head同时加回SH DC3，其他实验条件保留。"""
import gc
import importlib.util
import json
from pathlib import Path
import torch
from torch import nn
import matrix_runtime as rt

ROOT=Path('/home/prmi/zwj/s00-shallow-dc-20261026')
SOURCE=Path('/home/prmi/zwj/s00-encoder-optimization-20261026')
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
assert rt.sha(SOURCE/'run.py')==PROTOCOL['source_run_sha256']
spec=importlib.util.spec_from_file_location('encoder_experiment',SOURCE/'run.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
rt.ROOT=ROOT


def config():
    cfg=source.config('shallow32')
    cfg.update(id='GlobalXYZOffsetScaleOpacityDCShallow32Split4',encoder_layout='14-32-ReLU',
        gaussian_input='global_xyz_scale_opacity_offset_logratio_shdc')
    return cfg


class Network(source.Network):
    def __init__(self,scene,cfg,seed):
        super().__init__(scene,cfg,seed)
        old=self.encoder[0];first=nn.Linear(14,32,device='cuda')
        with torch.no_grad():
            first.weight[:,:11].copy_(old.weight)
            first.weight[:,11:].zero_();first.bias.copy_(old.bias)
        self.encoder=nn.Sequential(first,nn.ReLU())
        for branch in self.compensation.branches.values():
            old=branch[0];layer=nn.Linear(62,32,device='cuda')
            with torch.no_grad():
                layer.weight[:,:11].copy_(old.weight[:,:11])
                layer.weight[:,11:14].zero_()
                layer.weight[:,14:].copy_(old.weight[:,11:]);layer.bias.copy_(old.bias)
            branch[0]=layer

    def raw(self,ratio,ids=None):
        original=super().raw(ratio,ids)
        dc=self.scene.g.get_features_dc.squeeze(1)
        if ids is not None:dc=dc[ids]
        return torch.cat([original,dc],1)


rt.Network=Network


def gate(scene,seed):
    net=Network(scene,config(),seed);reference=source.Network(scene,source.config('shallow32'),seed)
    errors=[]
    for ratio in rt.RATIOS:
        p,h,_=net(ratio);p0,h0,_=reference(ratio)
        ids=torch.nonzero(p>scene.uniform).flatten()
        assert torch.equal(p,p0)
        assert torch.allclose(h,h0,atol=2e-6,rtol=2e-6)
        assert torch.equal(net.compensate(h,ids,ratio),reference.compensate(h0,ids,ratio))
        errors.append(float((h-h0).abs().max()))
    assert torch.equal(net.raw(.05)[:,11:14],scene.attrs[:,6:9])
    class Allowed:
        def __init__(self,g):
            self.get_xyz=g.get_xyz;self.get_scaling=g.get_scaling
            self.get_opacity=g.get_opacity;self.get_features_dc=g.get_features_dc
    original_g,original_attrs=scene.g,scene.attrs
    p,h,_=net(.05);ids=torch.nonzero(p>scene.uniform).flatten();delta=net.compensate(h,ids,.05)
    try:
        scene.g=Allowed(original_g);scene.attrs=torch.full_like(original_attrs,float('nan'))
        pp,hh,_=net(.05)
        assert torch.equal(p,pp) and torch.equal(h,hh)
        assert torch.equal(delta,net.compensate(hh,ids,.05))
    finally:scene.g,scene.attrs=original_g,original_attrs
    with torch.no_grad():
        for branch in net.compensation.branches.values():branch[-1].weight.fill_(.01)
    p,h,_=net(.05);ids=torch.nonzero(p>scene.uniform).flatten()
    net.compensate(h,ids,.05).square().mean().backward()
    assert net.encoder[0].weight.grad[:,11:].norm()>0
    assert all(branch[0].weight.grad[:,11:14].norm()>0 for branch in net.compensation.branches.values())
    assert net.extra_tables[1].grad.norm()>0
    assert all(net.extra_tables[j].grad is None for j in [0,2,3])
    probe=Network(scene,config(),seed)
    optimizer=torch.optim.Adam(probe.parameters(),lr=.001,eps=1e-15,fused=True)
    actual=[]
    for step in range(1,4):
        optimizer.zero_grad(set_to_none=True)
        p,h,_=probe(.05);ids=torch.nonzero(p.detach()>scene.uniform).flatten()
        image,_,_=scene.render(ids,probe.compensate(h,ids,.05),scene.train[0],1+p[ids]-p[ids].detach())
        photo=rt.photo(image,scene.train[0]['gt']);budget=(p.sum()-.05*scene.N).abs()/(.05*scene.N)
        (photo+.01*budget).backward()
        grads=dict(encoder=float(probe.encoder[0].weight.grad[:,11:].norm()),
            **{name:float(branch[0].weight.grad[:,11:14].norm()) for name,branch in probe.compensation.branches.items()})
        assert all(rt.math.isfinite(v) for v in grads.values())
        actual.append(dict(step=step,photo=float(photo),dc_gradient_norms=grads));optimizer.step()
    assert all(any(row['dc_gradient_norms'][name]>0 for row in actual[1:]) for name in actual[-1]['dc_gradient_norms'])
    assert scene.unchanged()
    rt.save_json(ROOT/f'correctness_{seed}.json',dict(seed=seed,input_width=14,head_input_width=62,
        initial_probability_exact=True,initial_compensation_exact=True,initial_hidden_max_error=max(errors),
        raw_dc_exact=True,allowed_attribute_proxy=True,forbidden_derived_attribute_nan=True,
        dc_input_gradient=True,inactive_ratio_gradient_none=True,master_frozen=True,
        actual_photo_probe=actual,encoder_parameters=480,head_parameters=12750,
        total_parameters=sum(p.numel() for p in net.parameters())))
    del net,reference,probe,optimizer;gc.collect();torch.cuda.empty_cache()


def main():
    scene=rt.Scene('fr1_xyz')
    assert json.loads((ROOT/'results/fr1_xyz/scene_manifest.json').read_text())==json.loads((SOURCE/'results/fr1_xyz/scene_manifest.json').read_text())
    assert rt.sha(ROOT/'matrix_runtime.py')==PROTOCOL['runtime_sha256']
    assert rt.sha(Path(PROTOCOL['renderer_binary']))==PROTOCOL['renderer_sha256']
    dc=scene.g.get_features_dc.squeeze(1).detach()
    rt.np.save(ROOT/'frozen_sh_dc.npy',dc.cpu().numpy())
    rt.save_json(ROOT/'dc_metadata.json',dict(layout='original SH DC coefficients, no RGB conversion or normalization',
        min=dc.min(0).values.cpu().tolist(),max=dc.max(0).values.cpu().tolist(),mean=dc.mean(0).cpu().tolist(),std=dc.std(0).cpu().tolist()))
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
            rt.save_json(d/'dc_weight_updates.json',dict(encoder_dc_norm=float(net.encoder[0].weight[:,11:].norm()),
                head_dc_norms={name:float(branch[0].weight[:,11:14].norm()) for name,branch in net.compensation.branches.items()}))
        assert scene.unchanged()
        runs.append(info);rt.save_json(ROOT/'runs.json',runs);print('COMPLETE',seed,flush=True)
        del net;gc.collect();torch.cuda.empty_cache()
    rt.save_json(ROOT/'status.json',dict(stage='COMPLETE',models=2))


if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'run.exit').write_text('1');raise
    (ROOT/'run.exit').write_text('0')

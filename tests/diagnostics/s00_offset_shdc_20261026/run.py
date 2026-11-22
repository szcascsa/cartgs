"""世界XYZ+Scale+Opacity基础上，分别只加Offset3或SH DC3。"""
import gc
import importlib.util
import json
from pathlib import Path
import torch
from torch import nn
import matrix_runtime as rt

ROOT=Path('/home/prmi/zwj/s00-offset-shdc-20261026')
SOURCE=Path('/home/prmi/zwj/s00-xyz-scale-opacity-20261026')
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
assert rt.sha(SOURCE/'run.py')==PROTOCOL['source_run_sha256']
spec=importlib.util.spec_from_file_location('scale_opacity_experiment',SOURCE/'run.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
rt.ROOT=ROOT


def config(kind):
    cfg=previous.config();cfg.update(id=PROTOCOL['configs'][kind],extra_input=kind)
    return cfg


class Network(previous.Network):
    def __init__(self,scene,cfg,seed):
        super().__init__(scene,cfg,seed)
        encoder=nn.Linear(11,32,device='cuda')
        with torch.no_grad():
            encoder.weight[:,:8].copy_(self.encoder.weight)
            encoder.weight[:,8:].zero_()
            encoder.bias.copy_(self.encoder.bias)
        self.encoder=encoder
        for branch in self.compensation.branches.values():
            old=branch[0];first=nn.Linear(59,32,device='cuda')
            with torch.no_grad():
                first.weight[:,:8].copy_(old.weight[:,:8])
                first.weight[:,8:11].zero_()
                first.weight[:,11:].copy_(old.weight[:,8:])
                first.bias.copy_(old.bias)
            branch[0]=first

    def extra(self,ids=None):
        if self.cfg['extra_input']=='offset':
            xyz=self.scene.g.get_xyz if ids is None else self.scene.g.get_xyz[ids]
            owners=self.scene.owners if ids is None else self.scene.owners[ids]
            anchors=self.scene.source.anchors
            return (xyz-anchors.centers[owners])/anchors.offset_scales[owners]
        dc=self.scene.g.get_features_dc
        return (dc if ids is None else dc[ids]).squeeze(1)

    def raw(self,ratio,ids=None):
        return torch.cat([super().raw(ratio,ids),self.extra(ids)],1)


rt.Network=Network


def gate(scene,kind,seed):
    net=Network(scene,config(kind),seed);ref=previous.Network(scene,previous.config(),seed)
    errors=[]
    for ratio in rt.RATIOS:
        p,h,_=net(ratio);p0,h0,_=ref(ratio)
        ids=torch.nonzero(p>scene.uniform).flatten()
        assert torch.equal(p,p0)
        assert torch.allclose(h,h0,atol=2e-6,rtol=2e-6)
        assert torch.equal(net.compensate(h,ids,ratio),ref.compensate(h0,ids,ratio))
        errors.append(float((h-h0).abs().max()))
    class Allowed:
        def __init__(self,g):
            self.get_xyz=g.get_xyz;self.get_scaling=g.get_scaling;self.get_opacity=g.get_opacity
            if kind=='shdc':self.get_features_dc=g.get_features_dc
    original_g,original_attrs,original_source=scene.g,scene.attrs,scene.source
    p,h,_=net(.05);ids=torch.nonzero(p>scene.uniform).flatten();delta=net.compensate(h,ids,.05)
    try:
        scene.g=Allowed(original_g);scene.attrs=torch.full_like(original_attrs,float('nan'))
        if kind=='shdc':scene.source=object()  # 此分支不允许读取Anchor中心或尺度。
        p1,h1,_=net(.05)
        assert torch.equal(p,p1) and torch.equal(h,h1)
        assert torch.equal(delta,net.compensate(h1,ids,.05))
    finally:scene.g,scene.attrs,scene.source=original_g,original_attrs,original_source
    with torch.no_grad():
        for branch in net.compensation.branches.values():branch[-1].weight.fill_(.01)
    p,h,_=net(.05);ids=torch.nonzero(p>scene.uniform).flatten()
    net.compensate(h,ids,.05).square().mean().backward()
    assert net.encoder.weight.grad[:,8:].norm()>0
    assert all(b[0].weight.grad[:,8:11].norm()>0 for b in net.compensation.branches.values())
    assert net.extra_tables[1].grad.norm()>0
    assert all(net.extra_tables[j].grad is None for j in [0,2,3])
    assert scene.unchanged()
    rt.save_json(ROOT/f'correctness_{kind}_{seed}.json',dict(kind=kind,seed=seed,
        initial_probability_exact=True,initial_compensation_exact=True,max_hidden_error=max(errors),
        input_width=11,head_input_width=59,forbidden_attribute_proxy_passed=True,
        forbidden_derived_attribute_nan=True,extra_input_gradient=True,inactive_ratio_gradient_none=True,
        total_parameters=sum(p.numel() for p in net.parameters()),master_frozen=True,
        gradient_probe='temporary nonzero final output projection, not actual step1 hidden gradient'))
    del net,ref;gc.collect();torch.cuda.empty_cache()


def main():
    scene=rt.Scene('fr1_xyz')
    assert json.loads((ROOT/'results/fr1_xyz/scene_manifest.json').read_text())==json.loads((SOURCE/'results/fr1_xyz/scene_manifest.json').read_text())
    assert rt.sha(ROOT/'matrix_runtime.py')==PROTOCOL['runtime_sha256']
    assert rt.sha(Path(PROTOCOL['renderer_binary']))==PROTOCOL['renderer_sha256']
    for seed in PROTOCOL['seeds']:
        a=Network(scene,config('offset'),seed);b=Network(scene,config('shdc'),seed)
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
        del a,b
    for kind,seed in PROTOCOL['jobs']:gate(scene,kind,seed)
    runs=[]
    for kind,seed in PROTOCOL['jobs']:
        cfg=config(kind);d=ROOT/'results/fr1_xyz'/cfg['id']/str(seed)
        info=rt.train(scene,cfg,seed,12000,d,checkpoints=[4000,8000,12000])
        net=Network(scene,cfg,seed);net.load_state_dict(torch.load(info['checkpoint'],map_location='cuda')['state_dict'])
        for split in ['bank','validation','test']:
            path=d/(split+'_full.csv')
            if not path.exists():
                rows=rt.evaluate(scene,net.eval(),split,full_metrics=True,policies=('same_k','cached'))
                for row in rows:row.update(config=cfg['id'],seed=seed)
                rt.write_csv(path,rows)
        with torch.no_grad():
            extra=net.extra()
            rt.np.save(ROOT/f'input_{kind}.npy',extra.cpu().numpy())
            rt.save_json(d/'extra_input_updates.json',dict(encoder_weight_norm=float(net.encoder.weight[:,8:].norm()),
                head_weight_norms={name:float(branch[0].weight[:,8:11].norm()) for name,branch in net.compensation.branches.items()},
                input_min=extra.min(0).values.cpu().tolist(),input_max=extra.max(0).values.cpu().tolist(),
                input_std=extra.std(0).cpu().tolist()))
        assert scene.unchanged()
        runs.append(info);rt.save_json(ROOT/'runs.json',runs);print('COMPLETE',kind,seed,flush=True)
        del net;gc.collect();torch.cuda.empty_cache()
    rt.save_json(ROOT/'status.json',dict(stage='COMPLETE',models=4))


if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'run.exit').write_text('1');raise
    (ROOT/'run.exit').write_text('0')

"""加回局部Offset，按指定11→16→ReLU→32→ReLU→32→ReLU共享Encoder训练。"""
import gc
import importlib.util
import json
import math
from pathlib import Path
import torch
from torch import nn
import matrix_runtime as rt

ROOT=Path('/home/prmi/zwj/s00-offset-encoder16-relu-20261026')
PREVIOUS=Path('/home/prmi/zwj/s00-xyz-scale-opacity-20261026')
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
assert rt.sha(PREVIOUS/'run.py')==PROTOCOL['previous_run_sha256']
spec=importlib.util.spec_from_file_location('scale_opacity_experiment',PREVIOUS/'run.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
rt.ROOT=ROOT


def config():
    cfg=previous.config()
    cfg.update(id='GlobalXYZScaleOpacityOffsetE16Encoder16ReLUSplit4',
        gaussian_input='global_xyz_scale_opacity_offset',encoder_layout='11-16-ReLU-32-ReLU-32-ReLU')
    return cfg


class Network(previous.Network):
    def __init__(self,scene,cfg,seed):
        super().__init__(scene,cfg,seed)
        first=nn.Linear(11,16,device='cuda')
        with torch.no_grad():
            first.weight[:,:7].copy_(self.encoder.weight[:16,:7])
            first.weight[:,7:10].zero_()
            first.weight[:,10].copy_(self.encoder.weight[:16,7])
            first.bias.copy_(self.encoder.bias[:16])
        second=nn.Linear(16,32,device='cuda');rt.linear_init(second,seed+110)
        third=nn.Linear(32,32,device='cuda');rt.linear_init(third,seed+120)
        self.encoder=nn.Sequential(first,nn.ReLU(),second,nn.ReLU(),third,nn.ReLU())
        for branch in self.compensation.branches.values():
            old=branch[0];layer=nn.Linear(59,32,device='cuda')
            with torch.no_grad():
                layer.weight[:,:7].copy_(old.weight[:,:7])
                layer.weight[:,7:10].zero_()
                layer.weight[:,10].copy_(old.weight[:,7])
                layer.weight[:,11:].copy_(old.weight[:,8:])
                layer.bias.copy_(old.bias)
            branch[0]=layer

    def raw(self,ratio,ids=None):
        sc=self.scene;g=sc.g
        rows=sc.owners if ids is None else sc.owners[ids]
        position=g.get_xyz if ids is None else g.get_xyz[ids]
        scale=g.get_scaling if ids is None else g.get_scaling[ids]
        opacity=g.get_opacity if ids is None else g.get_opacity[ids]
        offset=(position-sc.source.anchors.centers[rows])/sc.source.anchors.offset_scales[rows]
        return torch.cat([position,scale,opacity,offset,
            torch.full((len(position),1),math.log(ratio),device=position.device)],1)

    def forward(self,ratio):
        h=self.encoder(self.raw(ratio))
        logits=self.selector(h)
        return logits.softmax(-1)[:,1]*self.scene.eligible,h,logits[:,1]-logits[:,0]


rt.Network=Network


def gate(scene,seed):
    net=Network(scene,config(),seed);reference=previous.Network(scene,previous.config(),seed)
    assert [type(layer).__name__ for layer in net.encoder]==['Linear','ReLU','Linear','ReLU','Linear','ReLU']
    assert [(layer.in_features,layer.out_features) for layer in net.encoder if isinstance(layer,nn.Linear)]==[(11,16),(16,32),(32,32)]
    for ratio in rt.RATIOS:
        p,h,_=net(ratio);p0,h0,_=reference(ratio)
        ids=torch.nonzero(p>scene.uniform).flatten()
        assert h.shape==(scene.N,32) and torch.isfinite(h).all()
        assert torch.equal(p,p0)
        assert torch.equal(net.compensate(h,ids,ratio),reference.compensate(h0,ids,ratio))
    raw=net.raw(.05)
    assert torch.equal(raw[:,7:10],scene.attrs[:,:3])
    class Allowed:
        def __init__(self,g):self.get_xyz=g.get_xyz;self.get_scaling=g.get_scaling;self.get_opacity=g.get_opacity
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
    assert all(layer.weight.grad.norm()>0 for layer in net.encoder if isinstance(layer,nn.Linear))
    assert net.encoder[0].weight.grad[:,7:10].norm()>0
    assert all(branch[0].weight.grad[:,7:10].norm()>0 for branch in net.compensation.branches.values())
    assert net.extra_tables[1].grad.norm()>0
    assert all(net.extra_tables[j].grad is None for j in [0,2,3])
    assert scene.unchanged()
    rt.save_json(ROOT/f'correctness_{seed}.json',dict(seed=seed,input_width=11,head_input_width=59,
        encoder_layout='Linear11-16,ReLU,Linear16-32,ReLU,Linear32-32,ReLU',initial_probability_exact=True,
        initial_compensation_exact=True,hidden_function_preserved=False,offset_exact=True,
        allowed_attribute_proxy=True,forbidden_derived_attribute_nan=True,encoder_all_layers_gradient=True,
        offset_input_gradient=True,inactive_ratio_gradient_none=True,master_frozen=True,
        encoder_parameters=sum(p.numel() for p in net.encoder.parameters()),
        head_parameters=sum(p.numel() for p in net.compensation.parameters()),
        total_parameters=sum(p.numel() for p in net.parameters()),
        gradient_probe='temporary nonzero output projection; not actual hidden gradient at zero-output step1'))
    del net,reference;gc.collect();torch.cuda.empty_cache()


def main():
    scene=rt.Scene('fr1_xyz')
    assert json.loads((ROOT/'results/fr1_xyz/scene_manifest.json').read_text())==json.loads((PREVIOUS/'results/fr1_xyz/scene_manifest.json').read_text())
    assert rt.sha(ROOT/'matrix_runtime.py')==PROTOCOL['runtime_sha256']
    assert rt.sha(Path(PROTOCOL['renderer_binary']))==PROTOCOL['renderer_sha256']
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
            rt.save_json(d/'encoder_diagnostics.json',dict(
                encoder_layer_weight_norms=[float(layer.weight.norm()) for layer in net.encoder if isinstance(layer,nn.Linear)],
                offset_weight_norm=float(net.encoder[0].weight[:,7:10].norm()),
                head_offset_weight_norms={name:float(branch[0].weight[:,7:10].norm()) for name,branch in net.compensation.branches.items()}))
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

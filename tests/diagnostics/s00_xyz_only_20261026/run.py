"""仅全局XYZ作为Gaussian输入；保留ratio、latent16、h32和四Head。"""
import gc
import importlib.util
import json
import math
from pathlib import Path
import torch
from torch import nn
import matrix_runtime as rt

ROOT=Path('/home/prmi/zwj/s00-xyz-only-20261026')
SOURCE=Path('/home/prmi/zwj/s00-rotation-encoder-20261026')
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
assert rt.sha(SOURCE/'run.py')==PROTOCOL['source_run_sha256']
spec=importlib.util.spec_from_file_location('baseline_split',SOURCE/'run.py')
baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
rt.ROOT=ROOT


def config():
    cfg=baseline.config('baseline')
    cfg.update(id='GlobalXYZOnlyE16Split4',gaussian_input='global_xyz_only')
    return cfg


class Network(baseline.Network):
    def __init__(self,scene,cfg,seed):
        super().__init__(scene,cfg,seed)
        encoder=nn.Linear(4,32,device='cuda')
        with torch.no_grad():
            encoder.weight[:,:3].copy_(self.encoder.weight[:,:3])
            encoder.weight[:,3].copy_(self.encoder.weight[:,10])
            encoder.bias.copy_(self.encoder.bias)
        self.encoder=encoder
        for branch in self.compensation.branches.values():
            old=branch[0]
            first=nn.Linear(52,32,device='cuda')
            with torch.no_grad():
                first.weight[:,:3].copy_(old.weight[:,:3])
                first.weight[:,3].copy_(old.weight[:,10])
                first.weight[:,4:].copy_(old.weight[:,11:])
                first.bias.copy_(old.bias)
            branch[0]=first

    def raw(self,ratio,ids=None):
        xyz=self.scene.g.get_xyz if ids is None else self.scene.g.get_xyz[ids]
        return torch.cat([xyz,torch.full((len(xyz),1),math.log(ratio),device=xyz.device)],1)

    def forward(self,ratio):
        h=torch.relu(self.encoder(self.raw(ratio)))
        logits=self.selector(h)
        return logits.softmax(-1)[:,1]*self.scene.eligible,h,logits[:,1]-logits[:,0]


rt.Network=Network


def gate(scene,seed):
    net=Network(scene,config(),seed)
    reference=baseline.Network(scene,baseline.config('baseline'),seed)
    checks=[]
    for ratio in rt.RATIOS:
        p,h,_=net(ratio);p0,h0,_=reference(ratio)
        ids=torch.nonzero(p>scene.uniform).flatten()
        assert torch.equal(p,p0)
        assert torch.equal(net.compensate(h,ids,ratio),reference.compensate(h0,ids,ratio))
        checks.append(dict(ratio=ratio,initial_probability_exact=True,initial_zero_compensation_exact=True))
    # 使用仅提供XYZ的代理，禁止网络隐式读取其他Gaussian属性。
    class XYZOnly:
        def __init__(self,xyz):self.get_xyz=xyz
    original_g,original_attrs=scene.g,scene.attrs
    p,h,_=net(.05);ids=torch.nonzero(p>scene.uniform).flatten();delta=net.compensate(h,ids,.05)
    try:
        scene.g=XYZOnly(original_g.get_xyz)
        scene.attrs=torch.full_like(original_attrs,float('nan'))
        pp,hh,_=net(.05)
        assert torch.equal(p,pp) and torch.equal(h,hh)
        assert torch.equal(delta,net.compensate(hh,ids,.05))
    finally:
        scene.g,scene.attrs=original_g,original_attrs
    with torch.no_grad():
        for branch in net.compensation.branches.values():branch[-1].weight.fill_(.01)
    p,h,_=net(.05);ids=torch.nonzero(p>scene.uniform).flatten()
    net.compensate(h,ids,.05).square().mean().backward()
    assert net.encoder.weight.grad.norm()>0
    assert net.extra_tables[1].grad.norm()>0
    assert all(net.extra_tables[j].grad is None for j in [0,2,3])
    assert all(branch[0].weight.grad.norm()>0 for branch in net.compensation.branches.values())
    assert scene.unchanged()
    rt.save_json(ROOT/f'correctness_{seed}.json',dict(seed=seed,input_width=4,head_input_width=52,
        initial_checks=checks,gaussian_attribute_proxy_gate=True,derived_attribute_nan_gate=True,
        encoder_gradient=True,active_latent_gradient=True,inactive_latent_gradient_none=True,
        head_parameters=sum(p.numel() for p in net.compensation.parameters()),
        total_parameters=sum(p.numel() for p in net.parameters()),master_frozen=True,
        gradient_probe='temporary nonzero final projection, not actual step1 hidden gradient'))
    del net,reference;gc.collect();torch.cuda.empty_cache()


def main():
    scene=rt.Scene('fr1_xyz')
    assert json.loads((ROOT/'results/fr1_xyz/scene_manifest.json').read_text())==json.loads((SOURCE/'results/fr1_xyz/scene_manifest.json').read_text())
    assert rt.sha(ROOT/'matrix_runtime.py')==PROTOCOL['runtime_sha256']
    assert rt.sha(Path(PROTOCOL['renderer_binary']))==PROTOCOL['renderer_sha256']
    xyz=scene.g.get_xyz.detach()
    rt.save_json(ROOT/'xyz_input_metadata.json',dict(N=scene.N,coordinate='raw world XYZ, no AABB fit or normalization',
        min=xyz.min(0).values.cpu().tolist(),max=xyz.max(0).values.cpu().tolist(),
        mean=xyz.mean(0).cpu().tolist(),std=xyz.std(0).cpu().tolist()))
    rt.np.save(ROOT/'frozen_global_xyz.npy',xyz.cpu().numpy())
    for seed in PROTOCOL['seeds']:gate(scene,seed)
    runs=[]
    for seed in PROTOCOL['seeds']:
        cfg=config();d=ROOT/'results/fr1_xyz'/cfg['id']/str(seed)
        info=rt.train(scene,cfg,seed,12000,d,checkpoints=[4000,8000,12000])
        net=Network(scene,cfg,seed)
        net.load_state_dict(torch.load(info['checkpoint'],map_location='cuda')['state_dict'])
        for split in ['bank','validation','test']:
            path=d/(split+'_full.csv')
            if not path.exists():
                rows=rt.evaluate(scene,net.eval(),split,full_metrics=True,policies=('same_k','cached'))
                for row in rows:row.update(config=cfg['id'],seed=seed)
                rt.write_csv(path,rows)
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

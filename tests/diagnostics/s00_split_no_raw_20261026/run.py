"""四Head去掉原始11维直连：仅Anchor/ratio16+h32，共48维。"""
import gc
import importlib.util
import json
from pathlib import Path
import torch
from torch import nn
import matrix_runtime as rt

ROOT=Path('/home/prmi/zwj/s00-split-no-raw-20261026')
BASE=Path('/home/prmi/zwj/s00-split-wide-rotation-20261026')
OLD=Path('/home/prmi/zwj/selector-capacity-matrix-20261026')
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
assert rt.sha(BASE/'run.py')==PROTOCOL['parent_run_sha256']
spec=importlib.util.spec_from_file_location('matched_four_head',BASE/'run.py')
parent=importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
rt.ROOT=ROOT


class Network(parent.Network):
    def __init__(self,scene,cfg,seed):
        super().__init__(scene,cfg,seed)
        if cfg.get('drop_raw'):
            for branch in self.compensation.branches.values():
                original=branch[0]
                assert original.in_features==59
                first=nn.Linear(48,32,device='cuda')
                with torch.no_grad():
                    first.weight.copy_(original.weight[:,11:])
                    first.bias.copy_(original.bias)
                branch[0]=first

    def compensate(self,h,ids,ratio):
        if not self.cfg.get('drop_raw'):
            return super().compensate(h,ids,ratio)
        latent=self.query(self.extra_tables,ratio,self.scene.owners[ids])
        return self.compensation(torch.cat([latent,h[ids]],1))


rt.Network=Network


def config(drop_raw):
    return dict(id=PROTOCOL['candidate'] if drop_raw else 'RawSharedE16Split4',shared=0,ratio=0,extra=16,
        selector_depth=1,compensation_depth=2,compensation_input='raw11',attach_shared=True,
        head_kind='split4',drop_raw=drop_raw)


def gate(scene,seed):
    reference=Network(scene,config(False),seed)
    net=Network(scene,config(True),seed)
    for name,value in reference.state_dict().items():
        if name.startswith(('encoder.','selector.','extra_tables.')):
            assert torch.equal(value,net.state_dict()[name]),name
    for name,branch in net.compensation.branches.items():
        old=reference.compensation.branches[name]
        assert torch.equal(branch[0].weight,old[0].weight[:,11:])
        assert torch.equal(branch[0].bias,old[0].bias)
        for j in [2,4]:
            assert torch.equal(branch[j].weight,old[j].weight)
            assert torch.equal(branch[j].bias,old[j].bias)
    for ratio in rt.RATIOS:
        p,h,_=net(ratio)
        p0,h0,_=reference(ratio)
        assert torch.equal(p,p0) and torch.equal(h,h0)
        ids=torch.nonzero(p>scene.uniform).flatten()
        assert net.compensate(h,ids,ratio).shape==(len(ids),14)
        assert torch.equal(net.compensate(h,ids,ratio),reference.compensate(h0,ids,ratio))
        assert torch.count_nonzero(net.compensate(h,ids,ratio))==0
    with torch.no_grad():
        for branch in net.compensation.branches.values():
            branch[-1].weight.fill_(.01)
    net.zero_grad(set_to_none=True)
    p,h,_=net(.05)
    ids=torch.nonzero(p>scene.uniform).flatten()
    net.compensate(h,ids,.05).square().sum().backward()
    assert net.extra_tables[1].grad.norm()>0
    assert all(net.extra_tables[j].grad is None for j in [0,2,3])
    assert any(p.grad is not None and p.grad.norm()>0 for p in net.encoder.parameters())
    assert all(p.grad is None for p in net.selector.parameters())
    for branch in net.compensation.branches.values():
        # latent初始为零，其投影权重梯度此时应为零；latent表梯度已在上面验证。
        assert torch.count_nonzero(branch[0].weight.grad[:,:16])==0
        assert branch[0].weight.grad[:,16:].norm()>0
    rt.save_json(ROOT/f'correctness_{seed}.json',dict(seed=seed,input_width=48,head_outputs=[3,3,4,4],
        same_initial_selector=True,same_initial_compensation_zero=True,
        retained_projection_weights_exact=True,common_hidden_and_output_weights_exact=True,
        direct_raw_feature_gather_removed=True,raw_attributes_still_in_encoder=True,
        encoder_compensation_gradient=True,active_latent_gradient=True,inactive_ratio_gradient_none=True,
        compensation_parameters=sum(p.numel() for p in net.compensation.parameters()),
        reference_compensation_parameters=sum(p.numel() for p in reference.compensation.parameters()),
        gradient_probe='temporary nonzero output projection; not actual step1'))
    assert scene.unchanged()
    del net,reference
    gc.collect()


def main():
    old=json.loads((OLD/'delivery_verification.json').read_text())
    assert rt.sha(ROOT/'matrix_runtime.py')==old['prototype_sha256']['runtime.py']
    assert rt.sha(Path(old['renderer_binary']))==old['renderer_sha256']
    scene=rt.Scene('fr1_xyz')
    assert json.loads((ROOT/'results/fr1_xyz/scene_manifest.json').read_text())==json.loads((BASE/'results/fr1_xyz/scene_manifest.json').read_text())
    runs=[]
    for seed in PROTOCOL['seeds']:
        gate(scene,seed)
        cfg=config(True)
        directory=ROOT/'results/fr1_xyz'/cfg['id']/str(seed)
        info=rt.train(scene,cfg,seed,12000,directory,checkpoints=[4000,8000,12000])
        net=Network(scene,cfg,seed)
        payload=torch.load(info['checkpoint'],map_location='cuda')
        net.load_state_dict(payload['state_dict'])
        for split in ['validation','test']:
            output=directory/(split+'_full.csv')
            if not output.exists():
                rows=rt.evaluate(scene,net.eval(),split,full_metrics=True,policies=('same_k','cached'))
                for row in rows:
                    row.update(config=cfg['id'],seed=seed)
                rt.write_csv(output,rows)
        assert scene.unchanged()
        runs.append(info)
        rt.save_json(ROOT/'runs.json',runs)
        rt.save_json(ROOT/'status.json',dict(stage='COMPLETE_MODEL',seed=seed,completed=len(runs),total=2))
        print('COMPLETE_MODEL',seed,flush=True)
        del net,payload
        gc.collect()
        torch.cuda.empty_cache()
    rt.save_json(ROOT/'status.json',dict(stage='TRAINING_COMPLETE',completed=2,total=2))


if __name__=='__main__':
    try:
        main()
    except Exception:
        (ROOT/'run.exit').write_text('1')
        raise
    (ROOT/'run.exit').write_text('0')

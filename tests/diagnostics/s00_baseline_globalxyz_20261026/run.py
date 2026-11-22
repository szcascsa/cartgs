"""原四Head Baseline仅新增全局XYZ，保留原属性和初始化。"""
import gc
import importlib.util
import json
from pathlib import Path
import torch
from torch import nn
import matrix_runtime as rt

ROOT=Path('/home/prmi/zwj/s00-baseline-globalxyz-20261026')
SOURCE=Path('/home/prmi/zwj/s00-rotation-encoder-20261026')
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
assert rt.sha(SOURCE/'run.py')==PROTOCOL['source_run_sha256']
spec=importlib.util.spec_from_file_location('original_baseline',SOURCE/'run.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
rt.ROOT=ROOT


def config():
    cfg=source.config('baseline')
    cfg.update(id='BaselinePlusGlobalXYZSplit4',encoder_layout='14-32-ReLU',
        gaussian_input='offset_scale_shdc_opacity_logratio_globalxyz')
    return cfg


class Network(source.Network):
    def __init__(self,scene,cfg,seed):
        super().__init__(scene,cfg,seed)
        old=self.encoder;first=nn.Linear(14,32,device='cuda')
        with torch.no_grad():
            first.weight[:,:11].copy_(old.weight)
            first.weight[:,11:].zero_();first.bias.copy_(old.bias)
        self.encoder=first
        for branch in self.compensation.branches.values():
            old=branch[0];first=nn.Linear(62,32,device='cuda')
            with torch.no_grad():
                first.weight[:,:11].copy_(old.weight[:,:11])
                first.weight[:,11:14].zero_()
                first.weight[:,14:].copy_(old.weight[:,11:]);first.bias.copy_(old.bias)
            branch[0]=first

    def raw(self,ratio,ids=None):
        xyz=self.scene.g.get_xyz if ids is None else self.scene.g.get_xyz[ids]
        return torch.cat([super().raw(ratio,ids),xyz],1)

    def forward(self,ratio):
        h=torch.relu(self.encoder(self.raw(ratio)))
        logits=self.selector(h)
        return logits.softmax(-1)[:,1]*self.scene.eligible,h,logits[:,1]-logits[:,0]


rt.Network=Network


def gate(scene,seed):
    net=Network(scene,config(),seed);ref=source.Network(scene,source.config('baseline'),seed)
    for key,value in ref.state_dict().items():
        if key=='encoder.weight':
            assert torch.equal(value,net.encoder.weight[:,:11])
            assert torch.count_nonzero(net.encoder.weight[:,11:])==0
        elif key.startswith('compensation.') and key.endswith('.0.weight'):
            other=net.state_dict()[key]
            assert torch.equal(value[:,:11],other[:,:11])
            assert torch.equal(value[:,11:],other[:,14:])
            assert torch.count_nonzero(other[:,11:14])==0
        else:assert torch.equal(value,net.state_dict()[key]),key
    errors=[]
    for ratio in rt.RATIOS:
        p,h,_=net(ratio);p0,h0,_=ref(ratio)
        ids=torch.nonzero(p>scene.uniform).flatten()
        assert torch.equal(p,p0)
        assert torch.allclose(h,h0,atol=2e-6,rtol=2e-6)
        assert torch.equal(net.compensate(h,ids,ratio),ref.compensate(h0,ids,ratio))
        assert torch.equal(net.raw(ratio)[:,:11],ref.raw(ratio))
        assert torch.equal(net.raw(ratio)[:,11:],scene.g.get_xyz)
        errors.append(float((h-h0).abs().max()))
    # 排除最终零投影掩盖中间层拷贝错误。
    with torch.no_grad():
        for obj in [net,ref]:
            for branch in obj.compensation.branches.values():branch[-1].weight.fill_(.01)
    p,h,_=net(.05);_,h0,_=ref(.05);ids=torch.nonzero(p>scene.uniform).flatten()
    a,b=net.compensate(h,ids,.05),ref.compensate(h0,ids,.05)
    assert torch.allclose(a,b,atol=2e-6,rtol=2e-6)
    probe=Network(scene,config(),seed)
    optimizer=torch.optim.Adam(probe.parameters(),lr=.001,eps=1e-15,fused=True)
    actual=[]
    for step in range(1,4):
        optimizer.zero_grad(set_to_none=True)
        p,h,_=probe(.05);ids=torch.nonzero(p.detach()>scene.uniform).flatten()
        image,_,_=scene.render(ids,probe.compensate(h,ids,.05),scene.train[0],1+p[ids]-p[ids].detach())
        photo=rt.photo(image,scene.train[0]['gt']);budget=(p.sum()-.05*scene.N).abs()/(.05*scene.N)
        (photo+.01*budget).backward()
        grads=dict(encoder=float(probe.encoder.weight.grad[:,11:].norm()),
            **{name:float(branch[0].weight.grad[:,11:14].norm()) for name,branch in probe.compensation.branches.items()})
        assert all(rt.math.isfinite(v) for v in grads.values())
        assert all(probe.extra_tables[j].grad is None for j in [0,2,3])
        actual.append(dict(step=step,photo=float(photo),xyz_gradient_norms=grads));optimizer.step()
    assert all(any(row['xyz_gradient_norms'][name]>0 for row in actual[1:]) for name in actual[-1]['xyz_gradient_norms'])
    assert scene.unchanged()
    rt.save_json(ROOT/f'correctness_{seed}.json',dict(seed=seed,input_width=14,head_input_width=62,
        common_weights_exact=True,new_xyz_columns_zero=True,original_raw11_exact=True,global_xyz_exact=True,
        initial_probability_exact=True,initial_compensation_exact=True,initial_hidden_max_error=max(errors),
        nonzero_output_probe_max_error=float((a-b).abs().max()),inactive_ratio_gradient_none=True,
        master_frozen=True,actual_photo_probe=actual,encoder_parameters=480,head_parameters=12750,
        total_parameters=sum(p.numel() for p in net.parameters())))
    del net,ref,probe,optimizer;gc.collect();torch.cuda.empty_cache()


def main():
    scene=rt.Scene('fr1_xyz')
    assert json.loads((ROOT/'results/fr1_xyz/scene_manifest.json').read_text())==json.loads((SOURCE/'results/fr1_xyz/scene_manifest.json').read_text())
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
        rt.save_json(d/'xyz_weight_updates.json',dict(encoder_xyz_norm=float(net.encoder.weight[:,11:].norm()),
            head_xyz_norms={name:float(branch[0].weight[:,11:14].norm()) for name,branch in net.compensation.branches.items()}))
        assert scene.unchanged()
        runs.append(info);rt.save_json(ROOT/'runs.json',runs);print('COMPLETE',seed,flush=True)
        del net;gc.collect();torch.cuda.empty_cache()
    rt.save_json(ROOT/'status.json',dict(stage='COMPLETE',models=2))


if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'run.exit').write_text('1');raise
    (ROOT/'run.exit').write_text('0')

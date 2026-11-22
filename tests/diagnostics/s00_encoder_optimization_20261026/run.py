"""固定当前11维输入和四Head，只比较共享Encoder的三种小改法。"""
import gc
import importlib.util
import json
from pathlib import Path
import torch
from torch import nn
import matrix_runtime as rt

ROOT=Path('/home/prmi/zwj/s00-encoder-optimization-20261026')
SOURCE=Path('/home/prmi/zwj/s00-offset-encoder16-relu-20261026')
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
assert rt.sha(SOURCE/'run.py')==PROTOCOL['source_run_sha256']
spec=importlib.util.spec_from_file_location('offset_encoder_source',SOURCE/'run.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
rt.ROOT=ROOT


def config(kind):
    layouts=dict(shallow32='11-32-ReLU',deep32='11-32-ReLU-32-ReLU-32-ReLU',
        residual32='ReLU(h+Linear32-32(ReLU(Linear32-32(h)))),h=ReLU(Linear11-32(input))')
    cfg=source.config();cfg.update(id=PROTOCOL['configs'][kind],encoder_variant=kind,encoder_layout=layouts[kind])
    return cfg


class ResidualEncoder(nn.Module):
    def __init__(self,first,seed):
        super().__init__();self.first=first
        self.inner=nn.Linear(32,32,device='cuda');rt.linear_init(self.inner,seed+110)
        self.output=nn.Linear(32,32,device='cuda')
        with torch.no_grad():self.output.weight.zero_();self.output.bias.zero_()

    def forward(self,x):
        h=torch.relu(self.first(x))
        return torch.relu(h+self.output(torch.relu(self.inner(h))))


class Network(source.Network):
    def __init__(self,scene,cfg,seed):
        super().__init__(scene,cfg,seed)
        # 共同首层恢复上一组8→32的全部32行，并增加零初始化Offset列。
        reference=source.previous.Network(scene,source.previous.config(),seed)
        first=nn.Linear(11,32,device='cuda')
        with torch.no_grad():
            first.weight[:,:7].copy_(reference.encoder.weight[:,:7])
            first.weight[:,7:10].zero_()
            first.weight[:,10].copy_(reference.encoder.weight[:,7])
            first.bias.copy_(reference.encoder.bias)
        kind=cfg['encoder_variant']
        if kind=='shallow32':self.encoder=nn.Sequential(first,nn.ReLU())
        elif kind=='deep32':
            second=nn.Linear(32,32,device='cuda');rt.linear_init(second,seed+110)
            third=nn.Linear(32,32,device='cuda');rt.linear_init(third,seed+120)
            self.encoder=nn.Sequential(first,nn.ReLU(),second,nn.ReLU(),third,nn.ReLU())
        else:
            assert kind=='residual32';self.encoder=ResidualEncoder(first,seed)


rt.Network=Network


def encoder_first(net):
    return net.encoder.first if isinstance(net.encoder,ResidualEncoder) else net.encoder[0]


def gate(scene,kind,seed):
    net=Network(scene,config(kind),seed);shallow=Network(scene,config('shallow32'),seed)
    assert all(torch.equal(value,net.state_dict()[key]) for key,value in shallow.state_dict().items()
        if key.startswith(('selector.','compensation.','extra_tables.')))
    hidden_equal=True
    for ratio in rt.RATIOS:
        p,h,_=net(ratio);p0,h0,_=shallow(ratio)
        ids=torch.nonzero(p>scene.uniform).flatten()
        assert torch.equal(p,p0)
        assert torch.equal(net.compensate(h,ids,ratio),shallow.compensate(h0,ids,ratio))
        hidden_equal=hidden_equal and torch.equal(h,h0)
        assert torch.isfinite(h).all() and h.shape==(scene.N,32)
    if kind=='residual32':assert hidden_equal
    raw=net.raw(.05);assert torch.equal(raw[:,7:10],scene.attrs[:,:3])
    with torch.no_grad():
        for branch in net.compensation.branches.values():branch[-1].weight.fill_(.01)
    p,h,_=net(.05);ids=torch.nonzero(p>scene.uniform).flatten()
    net.compensate(h,ids,.05).square().mean().backward()
    assert encoder_first(net).weight.grad.norm()>0
    if kind=='deep32':assert all(layer.weight.grad.norm()>0 for layer in net.encoder if isinstance(layer,nn.Linear))
    if kind=='residual32':
        assert net.encoder.output.weight.grad.norm()>0
        assert torch.count_nonzero(net.encoder.inner.weight.grad)==0
    assert net.extra_tables[1].grad.norm()>0
    assert all(net.extra_tables[j].grad is None for j in [0,2,3])
    # 从真实零补偿初始化检查每个实际Encoder参数随后能获得photo/预算梯度。
    probe=Network(scene,config(kind),seed)
    optimizer=torch.optim.Adam(probe.parameters(),lr=.001,eps=1e-15,fused=True)
    actual=[]
    for step in range(1,5):
        optimizer.zero_grad(set_to_none=True)
        p,h,_=probe(.05);ids=torch.nonzero(p.detach()>scene.uniform).flatten()
        image,_,_=scene.render(ids,probe.compensate(h,ids,.05),scene.train[0],1+p[ids]-p[ids].detach())
        photo=rt.photo(image,scene.train[0]['gt']);budget=(p.sum()-.05*scene.N).abs()/(.05*scene.N)
        (photo+.01*budget).backward()
        norms={name:float(par.grad.norm()) for name,par in probe.encoder.named_parameters()}
        assert all(rt.math.isfinite(v) for v in norms.values())
        actual.append(dict(step=step,photo=float(photo),gradient_norms=norms));optimizer.step()
    for name,par in probe.encoder.named_parameters():
        assert any(r['gradient_norms'][name]>0 for r in actual[1:]),(kind,name)
    assert scene.unchanged()
    rt.save_json(ROOT/f'correctness_{kind}_{seed}.json',dict(kind=kind,seed=seed,
        input_width=11,head_input_width=59,initial_probability_exact=True,initial_compensation_exact=True,
        common_selector_heads_latents_exact=True,initial_hidden_equal_shallow=hidden_equal,
        offset_exact=True,encoder_gradient=True,inactive_ratio_gradient_none=True,
        encoder_parameters=sum(p.numel() for p in net.encoder.parameters()),
        total_parameters=sum(p.numel() for p in net.parameters()),actual_photo_probe=actual,master_frozen=True))
    del net,shallow,probe,optimizer;gc.collect();torch.cuda.empty_cache()


@torch.no_grad()
def diagnostics(scene,net,directory):
    rows=[]
    for ratio in rt.RATIOS:
        p,h,_=net(ratio)
        rows.append(dict(ratio=ratio,shared_rms=float(h.square().mean().sqrt()),shared_max=float(h.abs().max()),
            shared_zero_fraction=float((h==0).float().mean()),dead_channels=int((h==0).all(0).sum()),
            p_mean=float(p.mean()),p10=float(p.quantile(.1)),p90=float(p.quantile(.9)),
            cached_k=int((p>scene.uniform).sum())))
    rt.write_csv(directory/'encoder_output_statistics.csv',rows)
    rt.save_json(directory/'encoder_parameter_statistics.json',
        {name:float(par.norm()) for name,par in net.encoder.named_parameters()})


def main():
    scene=rt.Scene('fr1_xyz')
    assert json.loads((ROOT/'results/fr1_xyz/scene_manifest.json').read_text())==json.loads((SOURCE/'results/fr1_xyz/scene_manifest.json').read_text())
    assert rt.sha(ROOT/'matrix_runtime.py')==PROTOCOL['runtime_sha256']
    assert rt.sha(Path(PROTOCOL['renderer_binary']))==PROTOCOL['renderer_sha256']
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
        diagnostics(scene,net,d)
        assert scene.unchanged()
        runs.append(info);rt.save_json(ROOT/'runs.json',runs)
        print('COMPLETE',kind,seed,flush=True)
        del net;gc.collect();torch.cuda.empty_cache()
    rt.save_json(ROOT/'status.json',dict(stage='COMPLETE',models=6))


if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'run.exit').write_text('1');raise
    (ROOT/'run.exit').write_text('0')

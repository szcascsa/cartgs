"""原生重复fixture校验、Python接口压力测试；后者不代替C++ Mapping耗时。"""
import csv
import json
from pathlib import Path
import sys
import torch

ROOT=Path('/home/prmi/zwj/anchor-selector-r3-performance-20261022')
SOURCE=Path('/home/prmi/zwj/private-compensation-online-20261022/source')
sys.path.insert(0,str(SOURCE/'python'))
from selector import load_selector

torch.set_num_threads(2)
checks=[]
adam_checks=[]
for path in sorted((ROOT/'native1').glob('*_parity.pt')):
    a=torch.jit.load(str(path),map_location='cpu').state_dict()
    b=torch.jit.load(str(ROOT/'native2'/path.name),map_location='cpu').state_dict()
    assert a.keys()==b.keys()
    assert all('gradient_'+str(i) in a for i in range(6)), 'six Master groups must receive gradients'
    for key,x in a.items():
        y=b[key]
        exact=x.dtype in (torch.int32,torch.int64,torch.bool) or key in ('mask','selected_ids','uniform','owner','gaussian_uid')
        passed=torch.equal(x,y) if exact else torch.allclose(x,y,atol=2e-5,rtol=2e-4)
        checks.append(dict(case=path.stem,tensor=key,exact_required=exact,passed=passed,
                           max_abs=float((x.double()-y.double()).abs().max()) if x.numel() else 0))
    if 'selected_ids' in a:
        assert len(a['selected_ids'])==int(a['mask'].sum())
        assert torch.equal(torch.nonzero(a['mask']>0).flatten(),a['selected_ids'])
    assert len(torch.unique(a['gaussian_uid']))==len(a['gaussian_uid'])
    # Fixture包括原生实际step前后的状态，验证基础Adam公式，不切换SparseAdam语义。
    for key,p in a.items():
        if not key.startswith('parameter_before_'):continue
        i=key.removeprefix('parameter_before_')
        if 'gradient_'+i not in a:continue
        grad=a['gradient_'+i];m=a['before_'+i+'_m'];v=a['before_'+i+'_v']
        expected_m=.9*m+.1*grad;expected_v=.999*v+.001*grad.square()
        assert torch.allclose(expected_m,a['after_'+i+'_m'],atol=2e-6,rtol=2e-4)
        assert torch.allclose(expected_v,a['after_'+i+'_v'],atol=2e-6,rtol=2e-4)
        assert int(a['after_'+i+'_step'])==int(a['before_'+i+'_step'])+1
        lr=[.00016,.0025,.000125,.05,.005,.001][int(i)] if int(i)<6 else .001
        step=int(a['after_'+i+'_step'])
        expected=p-(lr/(1-.9**step))*a['after_'+i+'_m']/(a['after_'+i+'_v'].sqrt()/(1-.999**step)**.5+1e-15)
        adam_checks.append(dict(case=path.stem,parameter=i,max_abs=float((expected-a['parameter_after_'+i]).abs().max()),
                                passed=torch.allclose(expected,a['parameter_after_'+i],atol=2e-5,rtol=2e-4)))
gate=all(x['passed'] for x in checks if not x['tensor'].startswith('parameter_after_')) and all(x['passed'] for x in adam_checks)
(ROOT/'fixture_verification.json').write_text(json.dumps(dict(passed=gate,end_to_end_allclose=all(x['passed'] for x in checks),
     tolerances=dict(atol=2e-5,rtol=2e-4),comparisons=checks,fixed_state_adam_checks=adam_checks,
     boundary='renderer atomic nondeterminism can amplify tiny gradients in Adam eps1e-15; do not silently relax kernel parity'),indent=2))
assert gate

model=load_selector(str(SOURCE.parent/'r3_independent14_seed2026/7465_shutdown/ply/selector.pt'),'cuda')
fixture=torch.jit.load(str(ROOT/'native1/lod_0.050000_parity.pt'),map_location='cuda').state_dict()
base=fixture['parameter_before_0'];logscale=fixture['parameter_before_4'];dc=fixture['parameter_before_1'];logopacity=fixture['parameter_before_3']
rows=[]
for n in [len(base),10000,50000,100000]:
    idx=torch.arange(n,device='cuda')%len(base)
    xyz=base[idx].clone().requires_grad_();scale=logscale[idx].exp().detach().requires_grad_()
    color=dc[idx].clone().requires_grad_();opacity=logopacity[idx].sigmoid().detach().requires_grad_()
    owner=model.anchors.rows[idx];centers=model.anchors.centers;features=model.anchors.features
    table=torch.nn.Parameter(model.gaussian_compensation.residual_1[idx].detach().clone())
    for ratio in [.01,.05,.1,.15]:
        k=max(1,int(n*ratio));ids=torch.arange(k,device='cuda')
        def body():
            offset=(xyz-centers[owner])/model.anchors.offset_scales[owner]
            inp=torch.cat((features[owner],offset,scale,color.squeeze(1),opacity,torch.full((n,1),ratio,device='cuda').log()),1)
            h=model.local_emd[1](model.local_emd[0](inp));p=model.soft_net(model.local_emd[3](model.local_emd[2](h))).softmax(-1)[:,1]
            frozen=model.compensation(h[ids]).detach();residual=frozen+table[ids]
            # 仅算子压力测试，不是photo loss、训练实验或实际选点集合。
            (p.mean()+residual.square().mean()).backward()
        for _ in range(5):
            body();model.zero_grad(set_to_none=True);table.grad=None
            for x in [xyz,scale,color,opacity]:x.grad=None
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        samples=[]
        for _ in range(30):
            start.record();body();end.record();end.synchronize();samples.append(start.elapsed_time(end))
            model.zero_grad(set_to_none=True);table.grad=None
            for x in [xyz,scale,color,opacity]:x.grad=None
        rows.append(dict(N=n,K=k,ratio=ratio,synthetic=n!=len(base),method='Python ATen operator stress; synthetic objective;no renderer/Adam',
                         mean_ms=sum(samples)/len(samples),median_ms=sorted(samples)[len(samples)//2],
                         p95_ms=sorted(samples)[28],peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved()))
with (ROOT/'scale_stress.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('PASS fixture repeat/Adam verification and synthetic scaling',flush=True)

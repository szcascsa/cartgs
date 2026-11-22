"""固定地图全链路gate；Adam固定梯度gate独立，atomic放大单独报告。"""
import csv
import json
from pathlib import Path
import torch

R=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022');torch.set_num_threads(2)
rows=[];passed=True
cases=['Full','lod_0.010000','lod_0.050000','lod_0.100000','lod_0.150000']
def load(run,case):return torch.jit.load(str(R/run/(case+'_parity.pt')),map_location='cpu').state_dict()
for case in cases:
    ref=load('legacy_a',case)
    for run in ['legacy_b','adam','adam_decode','fused_a','fused_b']:
        other=load(run,case);assert ref.keys()==other.keys(),(case,run,'GradNone/state/schema mismatch')
        for key,a in ref.items():
            b=other[key];exact=not a.is_floating_point() or key in ['uniform','mask','owner','selected_ids','gaussian_uid']
            d=(a.double()-b.double()).abs()
            ok=torch.equal(a,b) if exact else torch.allclose(a,b,atol=2e-5,rtol=2e-4)
            atomic_update=key.startswith('parameter_after_')
            rows.append(dict(case=case,run=run,tensor=key,exact_required=exact,passed=ok,
                optimizer_amplification=atomic_update,max_abs=float(d.max()) if d.numel() else 0,
                max_rel=float((d/a.double().abs().clamp_min(1e-30)).max()) if d.numel() else 0,
                rms=float(d.square().mean().sqrt()) if d.numel() else 0,
                bad_elements=int((d>(0 if exact else 2e-5+2e-4*a.double().abs())).sum())))
            if not atomic_update:passed &= ok
        for i in range(6):assert 'gradient_'+str(i) in other
        for i in range(17,21):
            if case!='Full':assert ('gradient_'+str(i) in other)==(i==17+cases.index(case)-1)
with (R/'fixed_replay_parity.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(R/'fixed_replay_gate.json').write_text(json.dumps(dict(passed=passed,
    tolerance=dict(atol=2e-5,rtol=2e-4),comparisons=len(rows),
    fixed_gradient_adam_gate='phase1_final/adam_parity.csv, atol2e-7 rtol2e-6',
    post_adam_boundary='parameter_after reported separately; compare baseline-repeat envelope for tiny gradients and eps1e-15'),indent=2))
assert passed,[x for x in rows if not x['passed'] and not x['optimizer_amplification']]
print('PASS Fixed checkpoint outputs,full gradients,exact UID/owner/IDs/K,ratio gradNone/state clocks',flush=True)

"""连续训练的baseline-repeat包络；不将浮点漂移重标为严格Parity。"""
import csv
import json
from pathlib import Path
import torch
import pandas as pd
R=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022');torch.set_num_threads(2)
curves=[];comparisons=[]
for run in ['continuous_legacy_a','continuous_legacy_b','continuous_fused_a','continuous_fused_b']:
    a=pd.read_csv(R/run/'training_curve.csv');a['run']=run;curves.append(a)
curve=pd.concat(curves);curve.to_csv(R/'continuous_training_curve.csv',index=False)
for file in sorted((R/'continuous_legacy_a').glob('*.pt')):
    a=torch.jit.load(str(file),map_location='cpu').state_dict()
    others={run:torch.jit.load(str(R/run/file.name),map_location='cpu').state_dict() for run in ['continuous_legacy_b','continuous_fused_a','continuous_fused_b']}
    assert all(a.keys()==b.keys() for b in others.values()),'gradNone/schema mismatch'
    for run,b in others.items():
        for key,x in a.items():
            y=b[key];exact=key in ['owner','gaussian_uid'] or key.endswith('_step')
            if x.shape!=y.shape:
                comparisons.append(dict(snapshot=file.name,run=run,tensor=key,passed=False,max_abs=float('nan'),rms=float('nan'),
                    exact_required=exact,selected_equal=False,reference_numel=x.numel(),candidate_numel=y.numel()))
                assert not exact,(file.name,run,key,'identity shape mismatch')
                continue
            difference=(x.double()-y.double()).abs()
            ok=torch.equal(x,y) if exact else torch.allclose(x,y,atol=2e-5,rtol=2e-4)
            if exact:assert ok,(file.name,run,key)
            comparisons.append(dict(snapshot=file.name,run=run,tensor=key,passed=ok,
                max_abs=float(difference.max()) if x.numel() else 0,rms=float(difference.square().mean().sqrt()) if x.numel() else 0,
                exact_required=exact,selected_equal=torch.equal(a['selected_ids'],b['selected_ids']) if 'selected_ids' in a else True))
pd.DataFrame(comparisons).to_csv(R/'continuous_comparisons.csv',index=False)
summary=[]
for run,a in curve.groupby('run'):
    for kind,g in a.groupby('kind'):
        for end in [10,25,50,100]:
            tail=g[(g.cycle>=end-5)&(g.cycle<end)]
            summary.append(dict(run=run,kind=kind,end_cycle=end,photo_mean=float(tail.photo.mean()),
                mse_mean=float(tail.render_mse.mean()),psnr_mean=float((-10*torch.log10(torch.tensor(tail.render_mse.values))).mean()),
                K_mean=float(tail.K.mean())))
pd.DataFrame(summary).to_csv(R/'continuous_summary.csv',index=False)
(R/'continuous_complete.json').write_text(json.dumps(dict(completed=True,steps_per_branch=200,
    full_steps=100,lod_steps=100,ratio_exposures_each=25,exact_uid_owner_clocks=True,
    strict_float_allclose=all(x['passed'] for x in comparisons),scope='single fixed real camera/map,topology frozen;not online SLAM'),indent=2))
print(pd.DataFrame(summary).query('end_cycle==100').to_string(index=False));print('PASS continuous analysis',flush=True)

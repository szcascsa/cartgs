"""事后Bank条件替换：非新候选、不训练、不读nonBank选择规则。"""
import json
from pathlib import Path
import numpy as np
import torch
import run as r

rows=[];scene=r.rt.Scene('fr1_xyz')
for tier,family in [('confirm','R3'),('baseline_sets','Baseline59')]:
    fixture=r.Fixture(scene,3100,family)
    for seed in [5100,5101]:
        d=r.ROOT/'results'/tier/'fr1_xyz/SpatialOutputKnots'/str(seed);info=json.loads((d/'complete.json').read_text())
        net=r.make(fixture,info['config'],seed);net.load_state_dict(torch.load(info['checkpoint'],map_location='cuda')['state_dict']);net.eval()
        with torch.no_grad():
            for ratio,lower,upper in [(.025,.01,.05),(.075,.05,.1),(.125,.1,.15)]:
                ids=fixture.sets[ratio,'same_k'];supported=torch.isin(ids,fixture.sets[lower,'same_k'])
                a=net(lower,ids);b=net(upper,ids);t=(ratio-lower)/(upper-lower)
                normal=net(ratio,ids);assert torch.allclose(normal,(1-t)*a+t*b,atol=2e-6,rtol=2e-5)
                unsupported=normal.clone();unsupported[~supported]=b[~supported]
                supported_change=normal.clone();supported_change[supported]=b[supported]
                for label,delta in [('normal',normal),('replace_lower_unsupported',unsupported),('replace_lower_supported',supported_change)]:
                    for vi,view in enumerate(scene.validation):
                        image=r.render(fixture,net,ids,delta,view)[0]
                        rows.append(dict(tier=tier,source_family=family,seed=seed,ratio=ratio,condition=label,view=vi,path=view['path'],
                            K=len(ids),supported_lower=int(supported.sum()),unsupported_lower=int((~supported).sum()),
                            psnr=float(-10*torch.log10((image.clamp(0,1)-view['gt']).square().mean())),photo=float(r.rt.photo(image,view['gt'])),
                            role='posthocBank diagnostic only; conditional nonadditive effect; not a candidate or learned routing policy'))
        assert scene.unchanged()
r.rt.write_csv(r.ROOT/'endpoint_support_probe.csv',rows)
summary=[]
for family in ['R3','Baseline59']:
    for ratio in [.025,.075,.125]:
        group=[v for v in rows if v['source_family']==family and v['ratio']==ratio]
        values={label:np.mean([v['psnr'] for v in group if v['condition']==label]) for label in ['normal','replace_lower_unsupported','replace_lower_supported']}
        summary.append(dict(source=family,ratio=ratio,**values))
r.rt.save_json(r.ROOT/'endpoint_support_probe_summary.json',summary);print(json.dumps(summary),flush=True)

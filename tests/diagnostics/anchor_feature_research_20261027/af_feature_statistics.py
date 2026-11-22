"""CPU读取Feature学习、occupancy与Adam实际行更新，不运行额外训练。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent;rows=[]
for path in ROOT.glob('results/*/*/*/*/complete.json'):
    info=json.loads(path.read_text());diag=torch.load(path.parent/'feature_diagnostics.pt',map_location='cpu');e=diag['embedding']
    if e is None or len(e)!=len(diag['uids']):continue
    for label,mask in [('all',torch.ones(len(e),dtype=torch.bool)),('one',diag['occupancy']==1),('two_three',(diag['occupancy']>=2)&(diag['occupancy']<=3)),('four_ten',diag['occupancy']>=4)]:
        seen=(diag['sample_visits']>0)&mask;value=e[seen]
        if len(value):
            singular=torch.linalg.svdvals(value-value.mean(0));energy=singular.square();p=energy/energy.sum().clamp_min(1e-20)
            rank=float(torch.exp(-(p*p.clamp_min(1e-20).log()).sum()))
        else:rank=0.
        rows.append(dict(tier=info['tier'],scene=info['scene'],model=info['config']['id'],seed=info['seed'],route=info['route'],bin=label,
            anchors=int(mask.sum()),visited=int(seen.sum()),gradient_covered=int(((diag['gradient_visits']>0)&mask).sum()),
            feature_rms=float(value.square().mean().sqrt()) if value.numel() else 0.,effective_rank=rank,
            sample_visits_mean=float(diag['sample_visits'][seen].mean()) if seen.any() else 0.,
            gradient_steps_mean=float(diag['gradient_visits'][seen].mean()) if seen.any() else 0.,
            actual_adam_row_updates_mean=float(diag['actual_updates'][seen].mean()) if seen.any() else 0.))
if rows:pd.DataFrame(rows).to_csv(ROOT/'feature_statistics.csv',index=False)
print(pd.DataFrame(rows).tail(8).to_string(index=False),flush=True)

"""测试前锁定研究候选：两地图两seed低ratio Bank配对收益优先。"""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;entries=[]
for name in ['Head8','AnchorFiLM8','FrozenBaseHead8','FrozenBaseContextHead8']:
    values=[];details=[]
    for scene in ['fr1_xyz','fr2_xyz']:
        tier='confirm_continue' if name in ['Head8','AnchorFiLM8'] else 'mechanism_continue' if name=='FrozenBaseHead8' and scene=='fr1_xyz' else 'frozen_mechanism'
        for seed in [7200,7201]:
            d=json.loads((ROOT/'results'/tier/scene/name/str(seed)/'complete.json').read_text())
            b=json.loads((ROOT/'results/confirm_continue'/scene/'Baseline'/str(seed)/'complete.json').read_text())
            delta=d['bank_low']-b['bank_low'];values.append(delta);details.append(dict(scene=scene,seed=seed,low_delta=delta,seven_delta=d['bank_holdout_psnr']-b['bank_holdout_psnr'],unseen_delta=d['bank_unseen']-b['bank_unseen']))
    entries.append(dict(model=name,criterion='mean paired low(1%,5%) Bank improvement; seven-ratio mean as tie breaker only',score=float(np.mean(values)),details=details))
best=max(entries,key=lambda d:d['score'])['model']
record=dict(selected=best,entries=entries,all_candidates_still_evaluated=True,selection_data='Bank only; no nonBank metrics read',scope='research candidate, not authorization or evidence for production replacement')
(ROOT/'BANK_LOCKED_CANDIDATE.json').write_text(json.dumps(record,indent=2));print('BANK_LOCKED',best,flush=True)

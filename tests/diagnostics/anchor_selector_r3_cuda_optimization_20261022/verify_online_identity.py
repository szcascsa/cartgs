import json
from pathlib import Path
import sys
import torch
R=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022');sys.path.insert(0,str(R/'source/python'))
from selector import load_selector
torch.set_num_threads(2);rows=[]
for run in ['online_legacy_a','online_fused_a','online_fused_b','online_legacy_b']:
    cp=next((R/run).glob('*_shutdown/ply'));m=load_selector(str(cp/'selector.pt'),'cpu');b=m.anchors;p=m.gaussian_compensation
    owner=b.rows;counts=torch.bincount(owner,minlength=len(b.features));n=len(owner)
    assert owner.min()>=0 and owner.max()<len(b.features) and counts.min()>=1 and counts.max()<=10
    assert torch.equal(counts,b.member_counts) and len(torch.unique(p.gaussian_uids))==n
    assert all(getattr(p,'residual_'+str(i)).shape==(n,14) for i in range(4))
    assert b.levels.min()>=0 and b.levels.max()<=2
    rows.append(dict(run=run,N=n,A=len(b.features),unique_uids=n,member_min=int(counts.min()),member_max=int(counts.max()),
        level_counts=torch.bincount(b.levels,minlength=3).tolist(),r3_tables=4,r3_rows=n,passed=True))
(R/'online_identity_verification.json').write_text(json.dumps(rows,indent=2));print('PASS four real online UID/owner/members/levels/R3 table alignment',flush=True)

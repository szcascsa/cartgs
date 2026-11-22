"""只读复核在线固定十点结构、增长和几何事件日志。"""
from pathlib import Path
import json
import re
import sys
import numpy as np

r=Path(sys.argv[1]);v=r/'validation'
result=[]
for seed in (2026,2027):
    name='explicit_seed'+str(seed)
    text=(r/name/'training.log').read_text()
    cells=np.load(v/(name+'_mapping.npz'))
    identity=np.load(v/(name+'_block_identity.npz'))
    m=len(identity['uids']);n=len(cells['rows'])
    assert n==10*m and (np.bincount(cells['rows'],minlength=m)==10).all()
    assert len(np.unique(identity['uids']))==m
    assert set(identity['levels']).issubset({0,1,2})
    keys=np.column_stack((identity['levels'],identity['keys']))
    unique,counts=np.unique(keys,axis=0,return_counts=True)
    insert=list(map(int,re.findall(r'\[Explicit Anchor Insert\] added=(\d+)',text)))
    pruned=list(map(int,re.findall(r'\[Explicit Anchor Prune\] removed=(\d+)',text)))
    assert sum(insert)-sum(pruned)==m
    assert 'splits=' not in text
    result.append(dict(seed=seed,gaussians=n,anchors=m,ten_slots=True,unique_uid=True,
        layers={str(level):int((identity['levels']==level).sum()) for level in range(3)},
        collision_buckets=int((counts>1).sum()),collision_blocks=int(counts[counts>1].sum()),
        created_anchors=sum(insert),pruned_anchors=sum(pruned),
        growth_same_candidates=sum(map(int,re.findall(r'same_candidates=(\d+)',text))),
        growth_fine_candidates=sum(map(int,re.findall(r'fine_candidates=(\d+)',text))),
        max_level_limited=sum(map(int,re.findall(r'max_level_limited=(\d+)',text))),
        tau0=list(map(float,re.findall(r'Calibration.*tau0=([0-9.e+-]+)',text))),
        full_gradient_resolutions=sorted(set(re.findall(r'Explicit Full Gradient.*width=(\d+) height=(\d+)',text))),
        geometry_event_count=len(re.findall(r'\[CD BEGIN\].*type=(?:LOCAL|GLOBAL)_GEOMETRY',text))))
(v/'explicit_events.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

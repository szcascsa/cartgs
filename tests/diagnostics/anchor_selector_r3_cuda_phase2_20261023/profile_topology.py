"""真实checkpoint受控事件的NVTX子阶段，含Profiler开销，不作正式时延。"""
import os
from pathlib import Path
import sqlite3
import pandas as pd
R=Path(os.environ.get('CARTGS_PHASE2_RESULTS','/home/prmi/zwj/anchor-selector-r3-cuda-phase2-20261023'));db=sqlite3.connect(R/'profile_topology.sqlite');db.row_factory=sqlite3.Row
strings={x['id']:x['value'] for x in db.execute('select * from StringIds')};ranges=[dict(x) for x in db.execute('select * from NVTX_EVENTS where end is not null')]
for x in ranges:x['label']=x['text'] or strings.get(x['textId'],'')
parents=[x for x in ranges if x['label'].split('/')[0] in ['noop_gaussian','gaussian_density','anchor_growth','anchor_prune'] and x['label'].split('/')[-1] in ['fused','legacy']]
runtime={x['correlationId']:dict(x) for x in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')};kernels=[dict(x) for x in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL')];rows=[]
for parent in parents:
    scenario,implementation=parent['label'].split('/')
    for r in ranges:
        if r['label'].startswith('topology/') and r['start']>=parent['start'] and r['end']<=parent['end']:
            calls=[x for x in runtime.values() if r['start']<=x['start']<=r['end']];ids={x['correlationId'] for x in calls};work=[x for x in kernels if x['correlationId'] in ids]
            rows.append(dict(scenario=scenario,implementation=implementation,stage=r['label'],host_inclusive_ms=(r['end']-r['start'])/1e6,kernels=len(work),kernel_ms=sum((x['end']-x['start'])/1e6 for x in work),cuda_free_calls=sum('cudaFree' in strings[x['nameId']] for x in calls),cuda_free_ms=sum((x['end']-x['start'])/1e6 for x in calls if 'cudaFree' in strings[x['nameId']]),sync_calls=sum('Synchronize' in strings[x['nameId']] for x in calls)))
pd.DataFrame(rows).to_csv(R/'topology_substage_profile.csv',index=False);print('TOPO_PROFILE_PASS',flush=True)

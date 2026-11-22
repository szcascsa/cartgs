from pathlib import Path
import json
import sqlite3

D=Path(__file__).resolve().parent
db=sqlite3.connect(D/'paired_fb.sqlite');db.row_factory=sqlite3.Row
names=dict(db.execute('select id,value from StringIds'))
ranges=[dict(r) for r in db.execute('select start,end,text from NVTX_EVENTS where end is not null')]
runtime={r['correlationId']:dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')}
kernels=[dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL')]
summary={}
for mode in [1,6]:
    outer=[r for r in ranges if r['text']==f'strategy{mode}'];assert len(outer)==10
    stages={}
    for stage in ['network_forward','network_backward']:
        scopes=[r for r in ranges if r['text']==stage and any(o['start']<=r['start']<=r['end']<=o['end'] for o in outer)]
        owned=[];spans=[]
        for r in scopes:
            kk=[k for k in kernels if k['correlationId'] in runtime and r['start']<=runtime[k['correlationId']]['start']<=r['end']]
            owned+=kk;spans.append((max(k['end'] for k in kk)-min(k['start'] for k in kk))/1e6)
        stages[stage]=dict(kernels_per_iteration=len(owned)/10,
            kernel_active_ms=sum(k['end']-k['start'] for k in owned)/1e7,
            mean_kernel_device_span_ms=sum(spans)/10,
            mean_host_nvtx_ms=sum(r['end']-r['start'] for r in scopes)/1e7)
    summary[str(mode)]=stages
(D/'paired_profile_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

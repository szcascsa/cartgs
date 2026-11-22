"""Nsight runtime correlation → 最内层 NVTX 分解；逻辑字节另列，不冒充 DRAM counter。"""
from collections import defaultdict
from pathlib import Path
import json
import sqlite3
import sys

D = Path(sys.argv[1])
result = {}
for path in sorted(D.glob('profile_*.sqlite')):
    db = sqlite3.connect(path);db.row_factory=sqlite3.Row
    names=dict(db.execute('select id,value from StringIds'))
    ranges=[dict(r) for r in db.execute('select start,end,text from NVTX_EVENTS where end is not null')]
    top=[r for r in ranges if r['text']=='backward']
    assert len(top)==10,(path,len(top))
    runtime={r['correlationId']:dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')}
    stages=defaultdict(lambda:dict(count=0,active_ms=0.,kernels=defaultdict(lambda:dict(count=0,active_ms=0.))))
    kernels=[]
    for row in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
        k=dict(row);launch=runtime.get(k['correlationId'])
        if not launch or not any(r['start']<=launch['start']<=r['end'] for r in top):continue
        owners=[r for r in ranges if r['start']<=launch['start']<=r['end']]
        owners.sort(key=lambda r:r['end']-r['start'],reverse=True)
        stage='/'.join(r['text'] for r in owners if r['text'])
        v=stages[stage];v['count']+=.1;v['active_ms']+=(k['end']-k['start'])/1e7
        name=names[k['demangledName']];v['kernels'][name]['count']+=.1
        v['kernels'][name]['active_ms']+=(k['end']-k['start'])/1e7
        kernels.append(dict(name=name,stage=stage,duration_ms=(k['end']-k['start'])/1e6,
            registers=k.get('registersPerThread'),static_shared=k.get('staticSharedMemory'),dynamic_shared=k.get('dynamicSharedMemory'),
            start=k['start'],end=k['end']))
    spans=[]
    for scope in top:
        owned=[]
        for row in db.execute('select start,end,correlationId from CUPTI_ACTIVITY_KIND_KERNEL'):
            launch=runtime.get(row['correlationId'])
            if launch and scope['start']<=launch['start']<=scope['end']:owned.append(row)
        spans.append((max(k['end'] for k in owned)-min(k['start'] for k in owned))/1e6)
    apis=defaultdict(lambda:dict(count=0,cpu_ms=0.))
    for launch in runtime.values():
        if not any(r['start']<=launch['start']<=r['end'] for r in top):continue
        v=apis[names[launch['nameId']]];v['count']+=.1;v['cpu_ms']+=(launch['end']-launch['start'])/1e7
    transfers=[]
    tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    for table in ['CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
        if table not in tables:continue
        for row in db.execute('select * from '+table):
            r=dict(row);launch=runtime.get(r['correlationId'])
            if launch and any(v['start']<=launch['start']<=v['end'] for v in top):transfers.append(r)
    result[path.stem]=dict(iterations=10,kernels=len(kernels)/10,active_ms=sum(k['duration_ms'] for k in kernels)/10,
        mean_device_span_ms=sum(spans)/10,stages=dict(stages),runtime_apis=dict(apis),kernels_detail=kernels,
        transfers_and_memsets=transfers)
(D/'profile_summary.json').write_text(json.dumps(result,indent=2)+'\n')
for name,r in result.items():
    print(name,'kernels',r['kernels'],'active',r['active_ms'],'span',r['mean_device_span_ms'])
    for stage,v in sorted(r['stages'].items(),key=lambda x:-x[1]['active_ms']):print(round(v['active_ms'],6),round(v['count'],1),stage)

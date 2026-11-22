"""Nsight SQLite只读解析：CUDA runtime correlation -> CPU NVTX阶段。"""
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

prefix=Path(sys.argv[1])
database=prefix.with_suffix('.sqlite')
assert database.exists(),database
db=sqlite3.connect(f'file:{database}?mode=ro',uri=True)
db.row_factory=sqlite3.Row
tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
strings=dict(db.execute('SELECT id,value FROM StringIds'))
ranges=[dict(r) for r in db.execute('SELECT start,end,text FROM NVTX_EVENTS WHERE end IS NOT NULL')]
iterations=[r for r in ranges if r['text']=='iteration']
stages=[r for r in ranges if r['text'] in ('forward','backward','optimizer','regularization','loss','zero_grad','synchronize')]
assert len(iterations)==10,len(iterations)
runtimes=[dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
runtime_by_id={r['correlationId']:r for r in runtimes}

def stage_for(v):
    r=runtime_by_id.get(v['correlationId'])
    if not r:return 'outside'
    return next((s['text'] for s in stages if s['start']<=r['start']<=s['end']),'outside')

def union_ns(intervals):
    end=-1
    total=0
    for a,b in sorted(intervals):
        total+=max(0,b-max(a,end))
        end=max(end,b)
    return total

devices=[]
kernel_names=Counter()
kernel_counts=Counter()
by_stage=defaultdict(list)
for table,kind in [('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_MEMCPY','copy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')]:
    if table not in tables:continue
    for row in db.execute('SELECT * FROM '+table):
        v=dict(row)
        v['kind']=kind
        v['stage']=stage_for(v)
        devices.append(v)
        by_stage[v['stage']].append(v)
        if kind=='kernel':
            name=strings[v['demangledName']]
            kernel_names[name]+=v['end']-v['start']
            kernel_counts[name]+=1

stats={}
for name in sorted({r['text'] for r in stages}):
    markers=[r for r in stages if r['text']==name]
    entries=by_stage[name]
    spans=[]
    for marker in markers:
        current=[v for v in entries if marker['start']<=runtime_by_id[v['correlationId']]['start']<=marker['end']]
        if current:
            span=max(v['end'] for v in current)-min(v['start'] for v in current)
            active=union_ns([(v['start'],v['end']) for v in current])
            spans.append(dict(span_ms=span/1e6,device_work_active_ms=active/1e6,gap_fraction=1-active/span))
    stats[name]=dict(cpu_nvtx_mean_ms=sum(r['end']-r['start'] for r in markers)/len(markers)/1e6,
        kernels_per_iteration=sum(v['kind']=='kernel' for v in entries)/len(iterations),
        kernel_active_ms_per_iteration=sum(v['end']-v['start'] for v in entries if v['kind']=='kernel')/len(iterations)/1e6,
        copies_per_iteration=sum(v['kind']=='copy' for v in entries)/len(iterations),
        memsets_per_iteration=sum(v['kind']=='memset' for v in entries)/len(iterations),per_iteration_gpu=spans)
kernel=[v for v in devices if v['kind']=='kernel']
per_iteration=[]
for r in iterations:
    active=union_ns([(max(v['start'],r['start']),min(v['end'],r['end'])) for v in devices if v['start']<r['end'] and v['end']>r['start']])
    wall=r['end']-r['start']
    per_iteration.append(dict(instrumented_wall_ms=wall/1e6,device_work_active_ms=active/1e6,gap_fraction=1-active/wall))
payload=dict(protocol=dict(N=8491,warmup=100,captured_iterations=10,mode=sys.argv[2] if len(sys.argv)>2 else 'regularized_adam',trace='cuda,nvtx',cpu_sampling=False),
    stages=stats,per_iteration=per_iteration,kernels_per_iteration=len(kernel)/10,
    short_kernel_fraction=sum(v['end']-v['start']<10000 for v in kernel)/len(kernel),
    copy_kinds=dict(Counter(v['copyKind'] for v in devices if v['kind']=='copy')),
    kernel_counts=dict(kernel_counts),top20_kernels=[dict(name=n,total_ms=t/1e6) for n,t in kernel_names.most_common(20)],
    runtime_api_counts=dict(Counter(strings[r['nameId']] for r in runtimes)),
    forbidden_kernel_counts={key:sum(c for n,c in kernel_counts.items() if key.lower() in n.lower()) for key in ('grid_sampler','CatArrayBatchedCopy','elementwise_kernel')},
    limitation='Nsight instrumented wall and span gaps are not uninstrumented Mapping performance.')
prefix.with_name(prefix.name+'_summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
print(json.dumps({k:payload[k] for k in ('protocol','kernels_per_iteration','short_kernel_fraction','forbidden_kernel_counts')},indent=2))

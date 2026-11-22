"""解析native Nsight SQLite；时间单位ns，按runtime correlation归属NVTX阶段。"""
import json, sqlite3
from collections import Counter, defaultdict
from pathlib import Path

dest=Path(__file__).resolve().parent
db=sqlite3.connect(dest/'native_nsys.sqlite')
db.row_factory=sqlite3.Row
strings=dict(db.execute('SELECT id,value FROM StringIds'))
ranges=[dict(r) for r in db.execute('SELECT start,end,text FROM NVTX_EVENTS WHERE end IS NOT NULL')]
iterations=[r for r in ranges if r['text']=='iteration']
stages=[r for r in ranges if r['text'] in ('forward','backward','optimizer','regularization','loss','zero_grad','synchronize')]
runtimes=[dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
runtime_by_correlation={r['correlationId']:r for r in runtimes}

def stage_for(runtime):
    if not runtime:return 'outside'
    return next((r['text'] for r in stages if r['start']<=runtime['start']<=r['end']),'outside')

def union_ns(intervals):
    end=-1;total=0
    for a,b in sorted(intervals):
        total+=max(0,b-max(a,end));end=max(end,b)
    return total

by_stage=defaultdict(list)
kernel_names=Counter()
short_count=0;gemm_ns=0;kernel_ns=0
devices=[]
for table,kind in [('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_MEMCPY','copy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')]:
    for row in db.execute('SELECT * FROM '+table):
        v=dict(row);v['kind']=kind
        v['stage']=stage_for(runtime_by_correlation.get(v['correlationId']))
        by_stage[v['stage']].append(v);devices.append(v)
        if kind=='kernel':
            name=strings[v['demangledName']];duration=v['end']-v['start']
            kernel_names[name]+=duration;kernel_ns+=duration
            short_count+=duration<10000
            if any(s in name.lower() for s in ('gemm','gemv')):gemm_ns+=duration

stats={}
for name in sorted(set(r['text'] for r in stages)):
    spans=[r for r in stages if r['text']==name]
    entries=by_stage[name];kernels=[v for v in entries if v['kind']=='kernel']
    gpu=[]
    for r in spans:
        current=[v for v in entries if r['start']<=(runtime_by_correlation.get(v['correlationId']) or {}).get('start',-1)<=r['end']]
        if current:
            span=max(v['end'] for v in current)-min(v['start'] for v in current)
            active=union_ns([(v['start'],v['end']) for v in current])
            gpu.append(dict(span_ms=span/1e6,device_work_active_ms=active/1e6,gap_fraction=1-active/span))
    stats[name]=dict(cpu_nvtx_mean_ms=sum(r['end']-r['start'] for r in spans)/len(spans)/1e6,
                    kernels_per_iteration=len(kernels)/len(iterations),
                    kernel_active_ms_per_iteration=sum(v['end']-v['start'] for v in kernels)/len(iterations)/1e6,
                    copies_per_iteration=sum(v['kind']=='copy' for v in entries)/len(iterations),
                    memsets_per_iteration=sum(v['kind']=='memset' for v in entries)/len(iterations),
                    per_iteration_gpu=gpu)
apis={}
for r in runtimes:
    name=strings[r['nameId']]
    a=apis.setdefault(name,dict(count=0,total_cpu_ms=0))
    a['count']+=1;a['total_cpu_ms']+=(r['end']-r['start'])/1e6
iteration_gpu=[]
for r in iterations:
    intervals=[(max(v['start'],r['start']),min(v['end'],r['end'])) for v in devices if v['start']<r['end'] and v['end']>r['start']]
    active=union_ns(intervals);wall=r['end']-r['start']
    iteration_gpu.append(dict(wall_ms=wall/1e6,device_work_active_ms=active/1e6,gap_fraction=1-active/wall))
payload=dict(protocol=dict(N=8491,warmup=100,captured_iterations=len(iterations),mode='regularized_adam',
                           trace='cuda,nvtx',cpu_sampling=False,capture='cudaProfilerApi'),
             stages=stats,runtime_apis=apis,per_iteration=iteration_gpu,
             kernel_count=sum(v['kind']=='kernel' for v in devices),
             short_kernel_fraction=short_count/sum(v['kind']=='kernel' for v in devices),
             gemm_kernel_fraction=gemm_ns/kernel_ns,
             top20_kernels=[dict(name=n,total_ms=t/1e6) for n,t in kernel_names.most_common(20)],
             copy_kinds=dict(Counter(v['copyKind'] for v in devices if v['kind']=='copy')),
             limitation='Instrumented CPU/launch timing includes Nsight overhead; gaps are not pure framework cost.')
(dest/'native_nsys_summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
print(json.dumps({k:payload[k] for k in ('protocol','stages','kernel_count','short_kernel_fraction','gemm_kernel_fraction')},indent=2))

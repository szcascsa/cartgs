"""Native Nsight + Kineto 的独立统计；避免用源码调用次数猜测 kernel。"""
import csv,json,sqlite3
from collections import Counter,defaultdict
from pathlib import Path
dest=Path(__file__).resolve().parent
db=sqlite3.connect(dest/'native_nsys.sqlite');db.row_factory=sqlite3.Row
strings=dict(db.execute('SELECT id,value FROM StringIds'))
ranges=[dict(r) for r in db.execute('SELECT start,end,text FROM NVTX_EVENTS WHERE end IS NOT NULL')]
iterations=[r for r in ranges if r['text']=='iteration'];count=len(iterations)
stages=[r for r in ranges if r['text']!='iteration']
runtime=[dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
correlations={r['correlationId']:r for r in runtime}
def stage_for(r):
    return next((v['text'] for v in stages if v['start']<=r['start']<=v['end']),'outside') if r else 'outside'
def union(intervals):
    last=-1;total=0
    for a,b in sorted(intervals):total+=max(0,b-max(a,last));last=max(last,b)
    return total
devices=[];by_stage=defaultdict(list);top=Counter()
for table,kind in [('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_MEMCPY','copy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')]:
    for row in db.execute('SELECT * FROM '+table):
        v=dict(row);v['kind']=kind;v['stage']=stage_for(correlations.get(v['correlationId']));devices.append(v);by_stage[v['stage']].append(v)
        if kind=='kernel' and v['stage']!='outside':top[strings[v['demangledName']]]+=v['end']-v['start']
summary={}
for name,entries in by_stage.items():
    spans=[v for v in stages if v['text']==name]
    kernels=[v for v in entries if v['kind']=='kernel'];span_gaps=[]
    for r in spans:
        current=[v for v in entries if r['start']<=correlations[v['correlationId']]['start']<=r['end']]
        if current:
            span=max(v['end'] for v in current)-min(v['start'] for v in current);active=union([(v['start'],v['end']) for v in current])
            span_gaps.append(dict(span_ms=span/1e6,active_ms=active/1e6,gap_ms=(span-active)/1e6))
    summary[name]=dict(kernels_per_iteration=len(kernels)/count,kernel_active_ms=sum(v['end']-v['start'] for v in kernels)/count/1e6,
        copies_per_iteration=sum(v['kind']=='copy' for v in entries)/count,memsets_per_iteration=sum(v['kind']=='memset' for v in entries)/count,
        nvtx_host_ms=sum(v['end']-v['start'] for v in spans)/len(spans)/1e6 if spans else None,per_iteration_spans=span_gaps)
apis=defaultdict(lambda:dict(count=0,cpu_ms=0));launch_gaps=[]
for r in runtime:
    name=strings[r['nameId']];apis[name]['count']+=1;apis[name]['cpu_ms']+=(r['end']-r['start'])/1e6
for it in iterations:
    launches=sorted((v for v in runtime if it['start']<=v['start']<=it['end'] and 'Launch' in strings[v['nameId']]),key=lambda v:v['start'])
    launch_gaps.extend(max(0,b['start']-a['end'])/1e6 for a,b in zip(launches,launches[1:]))
payload=dict(N=220084,M=219969,ratio=.05,warmup=100,captured_iterations=count,stages=summary,runtime_apis=dict(apis),
    top20_kernels=[dict(name=n,total_ms=t/1e6,per_iteration_ms=t/count/1e6) for n,t in top.most_common(20)],
    copies_by_kind_in_timed_stages=dict(Counter(v['copyKind'] for v in devices if v['kind']=='copy' and v['stage']!='outside')),
    copies_by_kind_outside=dict(Counter(v['copyKind'] for v in devices if v['kind']=='copy' and v['stage']=='outside')),
    total_iteration_kernel_count=sum(v['kind']=='kernel' and v['stage']!='outside' for v in devices),
    short_kernel_fraction=sum(v['kind']=='kernel' and v['stage']!='outside' and v['end']-v['start']<10000 for v in devices)/sum(v['kind']=='kernel' and v['stage']!='outside' for v in devices),
    launch_gap_mean_ms=sum(launch_gaps)/len(launch_gaps),launch_gap_max_ms=max(launch_gaps),
    limitation='Nsight instrumentation and per-stage events affect dispatch; gaps include scheduling, dependencies and synchronization, not pure removable framework cost; outside includes untimed checkpoint restoration')
(dest/'native_nsys_summary.json').write_text(json.dumps(payload,indent=2))

trace=json.loads((dest/'native_operator_trace.json').read_text())['traceEvents']
cpu=[v for v in trace if v.get('cat')=='cpu_op' and v.get('ph')=='X'];kernels=[v for v in trace if v.get('cat')=='kernel']
# This native 2.3.1 Kineto build exports GPU External id == CUPTI correlation,
# not the CPU RecordFunction handle. Direct numeric matching mislabels GEMM as
# aten::transpose. Attribute through runtime correlation + enclosing CPU scope.
runtime_events={v['args'].get('correlation'):v for v in trace if v.get('cat')=='cuda_runtime'}
kernel_ids=defaultdict(list)
for v in kernels:
    rt=runtime_events.get(v['args'].get('correlation'))
    if not rt:continue
    owners=[op for op in cpu if op['tid']==rt['tid'] and op['ts']<=rt['ts'] and op['ts']+op['dur']>=rt['ts']+rt.get('dur',0)]
    if owners:kernel_ids[id(min(owners,key=lambda op:op['dur']))].append(v)
memory=[v for v in trace if v.get('name')=='[memory]' and v['args'].get('Device Type')==1]
aggregate=defaultdict(lambda:dict(count=0,cpu_inclusive_ms=0,kernels=0,kernel_active_ms=0,shapes=Counter()))
for v in cpu:
    record=aggregate[v['name']];record['count']+=1;record['cpu_inclusive_ms']+=v['dur']/1000
    linked=kernel_ids[id(v)];record['kernels']+=len(linked);record['kernel_active_ms']+=sum(k['dur'] for k in linked)/1000
    record['shapes'][json.dumps(v['args'].get('Input Dims',[]))]+=1
operators=[]
for name,v in aggregate.items():
    operators.append(dict(name=name,count_per_iteration=v['count']/10,cpu_inclusive_ms_per_iteration=v['cpu_inclusive_ms']/10,kernels_per_iteration=v['kernels']/10,kernel_active_ms_per_iteration=v['kernel_active_ms']/10,shapes=dict(v['shapes'])))
operators.sort(key=lambda v:v['kernel_active_ms_per_iteration'],reverse=True)
(dest/'operator_breakdown.json').write_text(json.dumps(dict(kind='Native LibTorch Kineto; 10 iterations; inclusive CPU operator times must not be summed; GPU attribution uses launch-runtime correlation and innermost enclosing CPU op because exported External id is inconsistent',operators=operators),indent=2))
positive=[v for v in memory if v['args']['Bytes']>0]
memory_stats=dict(native_cuda_allocation_requests=len(positive),allocation_requests_per_iteration=len(positive)/10,requested_bytes_per_iteration=sum(v['args']['Bytes'] for v in positive)/10,
    peak_allocated_bytes=max(v['args']['Total Allocated'] for v in memory),peak_reserved_bytes=max(v['args']['Total Reserved'] for v in memory),
    note='Native Kineto allocator events; includes profiler-retained tensors and untimed 18-tensor restore; peak is absolute process allocation, not incremental production peak')
allocation_ops=defaultdict(lambda:dict(requests=0,requested_bytes=0))
allocator_names={'aten::empty','aten::empty_strided','aten::empty_like','aten::new_empty','aten::new_empty_strided'}
for event in positive:
    owners=[op for op in cpu if op['tid']==event['tid'] and op['ts']<=event['ts']<=op['ts']+op['dur'] and op['name'] not in allocator_names]
    name=min(owners,key=lambda op:op['dur'])['name'] if owners else 'unscoped_allocator'
    allocation_ops[name]['requests']+=1
    allocation_ops[name]['requested_bytes']+=event['args']['Bytes']
memory_stats['operator_attribution_note']='Allocator instant events attributed to the innermost same-thread CPU operator, excluding empty allocation APIs. Scope attribution is not a separate cudaMalloc or production peak measurement.'
memory_stats['by_operator']=[dict(name=name,requests_per_iteration=value['requests']/10,requested_bytes_per_iteration=value['requested_bytes']/10)
    for name,value in sorted(allocation_ops.items(),key=lambda item:item[1]['requested_bytes'],reverse=True)]
(dest/'tensor_allocations.json').write_text(json.dumps(memory_stats,indent=2))
rows=list(csv.DictReader((dest/'native_benchmark.csv').open()))
(dest/'native_benchmark.json').write_text(json.dumps(dict(protocol=dict(warmup=100,measurements=500,stream='current default CUDA stream; no CUDA graph',final_modes=['cpp_inference','caller_train'],staged_mode='staged_train',staged_limit='Original staged CSV omitted division by tau=1; final profiler rerun includes it. Do not use staged CSV for final kernel count or total performance.',loss='fixed checkpoint opacity VJP + ratio loss; no renderer/TF/GI; private updates restored outside measured region'),rows=rows),indent=2))
print(json.dumps(dict(stages=summary,memory=memory_stats),indent=2))

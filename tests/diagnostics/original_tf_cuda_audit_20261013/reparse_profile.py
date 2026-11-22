"""通过 CUDA correlation 匹配 launch 阶段，避免 Kineto external ID 歧义。"""
import json
from collections import Counter,defaultdict
from pathlib import Path
dest=Path(__file__).resolve().parent
summary=json.loads((dest/'profiler_summary.json').read_text())
for mode,value in summary.items():
    events=json.loads((dest/f'profile_{mode}.json').read_text())['traceEvents']
    stages=('forward','backward','optimizer','zero_grad','loss','regularization')
    rf=[e for e in events if e.get('cat')=='user_annotation' and e.get('name') in stages]
    runtime={e.get('args',{}).get('correlation'):e for e in events if e.get('cat')=='cuda_runtime' and 'Launch' in e.get('name','')}
    def category(e):
        ts=e.get('ts',0)
        spans=[s for s in rf if s['ts']<=ts<=s['ts']+s['dur']]
        return min(spans,key=lambda s:s['dur'])['name'] if spans else 'outside'
    kernels=Counter();times=Counter();by_stage=defaultdict(list);top=Counter();mem=Counter();mem_bytes=Counter()
    for e in events:
        if e.get('cat')=='kernel':
            k=category(runtime.get(e.get('args',{}).get('correlation'),{}))
            kernels[k]+=1;times[k]+=e.get('dur',0);by_stage[k].append(e);top[e['name']]+=e.get('dur',0)
        if e.get('cat')=='gpu_memcpy':
            mem[e['name']]+=1;mem_bytes[e['name']]+=e.get('args',{}).get('bytes',0)
    value['kernels']=dict(kernels);value['kernel_us']=dict(times)
    value['kernel_attribution_method']='CUDA runtime correlation -> CPU record_function stage'
    value['cpu_stage_us']={name:sum(e['dur'] for e in rf if e['name']==name) for name in stages}
    value['kernel_stage_span_us']={name:max(e['ts']+e['dur'] for e in es)-min(e['ts'] for e in es) for name,es in by_stage.items()}
    # Three iterations are interleaved in spans: count per-iteration spans separately.
    spans=defaultdict(list)
    for marker in rf:
        related=[e for e in events if e.get('cat')=='kernel' and marker['ts']<=runtime.get(e.get('args',{}).get('correlation'),{}).get('ts',0)<=marker['ts']+marker['dur']]
        if related:
            span=max(e['ts']+e['dur'] for e in related)-min(e['ts'] for e in related)
            active=sum(e['dur'] for e in related)
            spans[marker['name']].append(dict(span_us=span,kernel_us=active,gap_fraction=1-active/span))
    value['per_iteration_gpu_spans']=dict(spans)
    value['gpu_memcpy_counts']=dict(mem);value['gpu_memcpy_bytes']=dict(mem_bytes)
    value['top20_kernels']=top.most_common(20)
    value['allocation_calls']=sum(e['count'] for e in value['operators'] if e['name'].startswith('aten::empty'))
    print(mode,dict(kernels),dict(times),dict(mem))
(dest/'profiler_summary.json').write_text(json.dumps(summary,indent=2))

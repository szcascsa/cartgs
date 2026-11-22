"""从原始Kineto trace提取真实Kernel，排除覆盖整个区间的GPU annotation。"""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
for name in ['concurrent_diagnostic','concurrent_diagnostic_v2','concurrent_diagnostic_final']:
    folder=HERE/'reports'/name
    path=folder/'SUMMARY.json'
    if not path.exists():continue
    d=json.loads(path.read_text())
    if d.get('kernel_categories_verified'):continue
    (folder/'SUMMARY_BEFORE_PROFILE_CATEGORY_FIX.json').write_text(json.dumps(d,indent=2))
    for p in d['profiles']:
        trace=json.loads((folder/f'trace_{p["condition"]}_{p["variant"]}.json').read_text())['traceEvents']
        kernels=[e for e in trace if e.get('cat')=='kernel']
        copies=[e for e in trace if e.get('cat')=='gpu_memcpy']
        memsets=[e for e in trace if e.get('cat')=='gpu_memset']
        p.update(gpu_us=sum(e['dur'] for e in kernels)/p['iterations'],kernels_per_frame=len(kernels)/p['iterations'],
            copies_per_frame=len(copies)/p['iterations'],memsets_per_frame=len(memsets)/p['iterations'],
            kernel_events=[dict(name=e['name'],us=e['dur']) for e in kernels])
    d['kernel_categories_verified']=True
    path.write_text(json.dumps(d,indent=2))
    print('PROFILE_CATEGORIES_FIXED',name)

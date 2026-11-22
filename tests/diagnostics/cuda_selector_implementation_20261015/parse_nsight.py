"""按 runtime correlation 和最内层 NVTX scope 归属真实 kernel。"""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sqlite3
import sys

root = Path(__file__).resolve().parent
all_results = {}
prefix = sys.argv[1] if len(sys.argv)>1 else ''
for backend in ['legacy', 'cublas', 'warp', 'cublas_strict']:
    path = root/f'{prefix}mlp_{backend}.sqlite'
    if not path.exists():
        continue
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    strings = dict(db.execute('SELECT id,value FROM StringIds'))
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    ranges = [dict(r) for r in db.execute('SELECT start,end,text FROM NVTX_EVENTS WHERE end IS NOT NULL')]
    iterations = [r for r in ranges if r['text'] == 'network_forward_backward']
    stages = [r for r in ranges if r['text'] in ['network_forward', 'network_backward']]
    assert len(iterations) == 10, (backend, len(iterations))
    runtime = [dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
    correlations = {r['correlationId']: r for r in runtime}
    devices = []
    names = Counter()
    for table, kind in [('CUPTI_ACTIVITY_KIND_KERNEL', 'kernel'),
                        ('CUPTI_ACTIVITY_KIND_MEMCPY', 'copy'), ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')]:
        if table not in tables:
            continue
        for row in db.execute('SELECT * FROM '+table):
            value = dict(row)
            launch = correlations.get(value['correlationId'])
            owners = [r for r in stages if launch and r['start'] <= launch['start'] <= r['end']]
            if not owners:
                continue
            value['stage'] = min(owners, key=lambda r: r['end']-r['start'])['text']
            value['kind'] = kind
            devices.append(value)
            if kind == 'kernel':
                names[strings[value['demangledName']]] += 1

    def union(intervals):
        last = -1
        total = 0
        for a, b in sorted(intervals):
            total += max(0, b-max(a, last))
            last = max(last, b)
        return total

    summary = {}
    for stage in ['network_forward', 'network_backward']:
        entries = [v for v in devices if v['stage'] == stage]
        kernels = [v for v in entries if v['kind'] == 'kernel']
        spans = []
        for scope in [r for r in stages if r['text'] == stage]:
            owned = [v for v in entries if scope['start'] <= correlations[v['correlationId']]['start'] <= scope['end']]
            if not owned:
                continue
            span = max(v['end'] for v in owned)-min(v['start'] for v in owned)
            active = union([(v['start'], v['end']) for v in owned])
            spans.append({'device_span_ms': span/1e6, 'active_ms': active/1e6, 'gap_ms': (span-active)/1e6})
        summary[stage] = dict(kernels_per_iteration=len(kernels)/10,
            kernel_active_ms=sum(v['end']-v['start'] for v in kernels)/10/1e6,
            copies_per_iteration=dict(Counter(v['copyKind'] for v in entries if v['kind'] == 'copy')),
            spans=spans)
    apis = defaultdict(lambda: dict(count=0, cpu_ms=0))
    gaps = []
    for r in runtime:
        name = strings[r['nameId']]
        apis[name]['count'] += 1
        apis[name]['cpu_ms'] += (r['end']-r['start'])/1e6
    for it in iterations:
        launches = sorted([r for r in runtime if it['start'] <= r['start'] <= it['end'] and 'Launch' in strings[r['nameId']]], key=lambda r: r['start'])
        gaps.extend(max(0, b['start']-a['end'])/1e6 for a, b in zip(launches, launches[1:]))
    all_results[backend] = dict(captured_iterations=10, M=219969, stages=summary,
        kernel_names=dict(names), runtime_apis=dict(apis),
        launch_gap_mean_ms=sum(gaps)/len(gaps) if gaps else None,
        launch_gap_max_ms=max(gaps, default=0),
        scope='MLP forward and all input/18-parameter VJP only; excludes ST/caller/optimizer/renderer/Mapping')
    db.close()
(root/f'{prefix}nsight_summary.json').write_text(json.dumps(all_results, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({b: {s: {'kernels': v['kernels_per_iteration'], 'active_ms': v['kernel_active_ms']}
                       for s, v in r['stages'].items()} for b, r in all_results.items()}, indent=2))

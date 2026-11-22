"""按 NVTX 和 launch correlation 汇总当前 structural caller；不包含 renderer/TF。"""
from collections import Counter
import json
from pathlib import Path
import sqlite3

root = Path(__file__).resolve().parent
result = {}


def union(intervals):
    total, last = 0, -1
    for start, end in sorted(intervals):
        total += max(0, end-max(start, last))
        last = max(last, end)
    return total


for backend in ('legacy', 'cuda'):
    path = root/f'caller_{backend}.sqlite'
    if not path.exists():
        continue
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    strings = dict(db.execute('SELECT id,value FROM StringIds'))
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    ranges = [dict(r) for r in db.execute('SELECT start,end,text FROM NVTX_EVENTS WHERE end IS NOT NULL')]
    iterations = [r for r in ranges if r['text'] == 'selector_caller']
    assert len(iterations) == 10, len(iterations)
    runtime = [dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
    correlation = {r['correlationId']: r for r in runtime}
    device = []
    for table, kind in [('CUPTI_ACTIVITY_KIND_KERNEL', 'kernel'), ('CUPTI_ACTIVITY_KIND_MEMCPY', 'copy'),
                        ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')]:
        if table in tables:
            for row in db.execute('SELECT * FROM '+table):
                item = dict(row)
                item['kind'] = kind
                launch = correlation.get(item['correlationId'])
                if launch:
                    item['launch_start'] = launch['start']
                    device.append(item)
    summary = {}
    for stage in ('selector_caller', 'caller_preparation', 'probability_network', 'selection_mask_gather', 'loss', 'caller_backward', 'adam'):
        scopes = [r for r in ranges if r['text'] == stage]
        entries, spans = [], []
        for scope in scopes:
            owned = [v for v in device if scope['start'] <= v['launch_start'] <= scope['end']]
            entries.extend(owned)
            if owned:
                active = union([(v['start'], v['end']) for v in owned])
                span = max(v['end'] for v in owned)-min(v['start'] for v in owned)
                spans.append(dict(active_ms=active/1e6, device_span_ms=span/1e6, gap_ms=(span-active)/1e6))
        kernels = [v for v in entries if v['kind'] == 'kernel']
        apis = [r for r in runtime if any(s['start'] <= r['start'] <= s['end'] for s in scopes)]
        summary[stage] = dict(
            kernels_per_iteration=len(kernels)/10,
            kernel_active_ms=sum(v['end']-v['start'] for v in kernels)/1e7,
            spans=spans,
            copy_count_per_iteration={str(k): v/10 for k, v in Counter(v['copyKind'] for v in entries if v['kind'] == 'copy').items()},
            copy_bytes_per_iteration=sum(v['bytes'] for v in entries if v['kind'] == 'copy')/10,
            kernel_names=dict(Counter(strings[v['demangledName']] for v in kernels)),
            runtime_api_count_per_iteration={k: v/10 for k, v in Counter(strings[v['nameId']] for v in apis).items()},
            synchronization_cpu_ms_per_iteration=sum(v['end']-v['start'] for v in apis if 'Synchronize' in strings[v['nameId']])/1e7,
        )
    result[backend] = dict(N=109564, captured_iterations=10, stages=summary,
                           scope='Current elastic caller + Gaussian activation/input gradients + compact/gather + structural opacity VJP + ratio + Adam; excludes renderer/TF/GI/Gaussian optimizer/Mapping')
    db.close()
(root/'caller_nsight_summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({b: {s: v['kernels_per_iteration'] for s, v in r['stages'].items()} for b, r in result.items()}, indent=2))

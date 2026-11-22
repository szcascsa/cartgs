"""由 NVTX Host 发射范围和 CUPTI correlation 归属推理 Kernel。"""
from collections import defaultdict
from pathlib import Path
import json
import sqlite3
import statistics
import sys

root = Path(__file__).resolve().parent
path = Path(sys.argv[1]) if len(sys.argv) > 1 else root/'inference_nodes.sqlite'
db = sqlite3.connect(path);db.row_factory = sqlite3.Row
strings = dict(db.execute('select id,value from StringIds'))
apis = list(db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'))
kernels = defaultdict(list)
for row in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
    kernels[row['correlationId']].append(dict(row))
copies = defaultdict(list)
for row in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY'):
    copies[row['correlationId']].append(dict(row))
groups = defaultdict(list)
for scope in db.execute('select * from NVTX_EVENTS where end is not null'):
    label = scope['text'] or strings.get(scope['textId'], '')
    if not label.startswith(('prepare/', 'render/')):continue
    issued = [row for row in apis if row['globalTid'] == scope['globalTid'] and
              scope['start'] <= row['start'] and row['end'] <= scope['end']]
    work = [kernel for api in issued for kernel in kernels[api['correlationId']]]
    memcpy = [copy for api in issued for copy in copies[api['correlationId']]]
    by_name = defaultdict(lambda: dict(calls=0, gpu_ns=0))
    for kernel in work:
        name = strings[kernel['demangledName']]
        by_name[name]['calls'] += 1;by_name[name]['gpu_ns'] += kernel['end']-kernel['start']
    groups[label].append(dict(wall_us=(scope['end']-scope['start'])/1000,
        kernels=len(work), kernel_us=sum(row['end']-row['start'] for row in work)/1000,
        host_launches=sum('Launch' in strings[row['nameId']] for row in issued),
        host_syncs=sum('Synchronize' in strings[row['nameId']] for row in issued),
        host_sync_us=sum(row['end']-row['start'] for row in issued if 'Synchronize' in strings[row['nameId']])/1000,
        dtoh=sum(row['copyKind'] == 2 for row in memcpy),
        by_name=by_name))
summary = []
for label, rows in groups.items():
    totals = defaultdict(lambda: dict(calls=0, gpu_us=0))
    for row in rows:
        for name, values in row['by_name'].items():
            totals[name]['calls'] += values['calls']/len(rows)
            totals[name]['gpu_us'] += values['gpu_ns']/1000/len(rows)
    summary.append(dict(label=label, samples=len(rows),
        **{key: statistics.median(row[key] for row in rows) for key in
           ('wall_us', 'kernels', 'kernel_us', 'host_launches', 'host_syncs', 'host_sync_us', 'dtoh')},
        top_kernels=sorted([dict(name=name, **values) for name, values in totals.items()],
                           key=lambda row: row['gpu_us'], reverse=True)[:20]))
(path.parent/(path.stem+'_analysis.json')).write_text(json.dumps(dict(summary=summary, raw=groups), indent=2)+'\n')
print(json.dumps([{key: value for key, value in row.items() if key != 'top_kernels'} for row in summary], indent=2))

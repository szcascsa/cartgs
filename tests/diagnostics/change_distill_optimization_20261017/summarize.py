from pathlib import Path
from collections import defaultdict
import csv
import json
import statistics

D = Path(__file__).resolve().parent
timings, components, pooled = [], [], []
for stage in ['baseline', 'stage_a', 'stage_a_grouped', 'stage_b', 'stage_c']:
    paired_rows = defaultdict(list)
    for path in sorted((D / stage / 'raw').glob('*/runs.json')):
        rows = [r for r in json.loads(path.read_text()) if r['label'].startswith('warm_')]
        if not rows: continue
        summary = {'stage': stage, 'case': path.parent.name, 'branch': rows[0]['branch'], 'runs': len(rows),
                   'steps_selector': rows[0]['selector_steps'], 'steps_tf': rows[0]['tf_steps']}
        for field in ['cpu_wall_ms', 'cuda_elapsed_ms', 'peak_allocated_bytes', 'additional_allocated_bytes']:
            values = [r[field] for r in rows]
            for label, value in [('median', statistics.median(values)), ('min', min(values)),
                                 ('max', max(values)), ('std', statistics.pstdev(values))]:
                summary[field + '_' + label] = value
        timings.append(summary)
        if path.parent.name.startswith('paired_'):
            paired_rows[rows[0]['branch']].extend(rows)
    for branch, rows in paired_rows.items():
        record = {'stage': stage, 'branch': branch, 'runs': len(rows)}
        for field in ['cpu_wall_ms', 'cuda_elapsed_ms', 'peak_allocated_bytes', 'additional_allocated_bytes']:
            values = [r[field] for r in rows]
            record[field] = {'median': statistics.median(values), 'min': min(values), 'max': max(values),
                             'std': statistics.pstdev(values)}
        pooled.append(record)
    for path in sorted((D / stage / 'raw').glob('profile_*/warm_0/components.csv')):
        aggregate = defaultdict(lambda: [0., 0., 0])
        for row in csv.DictReader(path.open()):
            aggregate[row['component']][0] += float(row['cpu_us']) / 1000
            aggregate[row['component']][1] += float(row['cuda_span_us']) / 1000
            aggregate[row['component']][2] += 1
        for name, (cpu, span, calls) in aggregate.items():
            components.append({'stage': stage, 'case': path.parents[1].name, 'component': name,
                               'cpu_ms': cpu, 'cuda_span_ms': span, 'calls': calls})
for name, rows in [('timings', timings), ('components', components)]:
    (D / (name + '.json')).write_text(json.dumps(rows, indent=2))
    if rows:
        with (D / (name + '.csv')).open('w') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
print(json.dumps(timings, indent=2))
(D / 'paired_summary.json').write_text(json.dumps(pooled, indent=2))

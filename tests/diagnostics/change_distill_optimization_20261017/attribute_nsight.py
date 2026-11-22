from pathlib import Path
import sqlite3
import json
import csv

D = Path(__file__).resolve().parent
names = ['geometry_transport', 'student_forward', 'teacher_forward', 'backward', 'adam_step',
         'cache_missing_indices', 'target_cache', 'hexplane_backward', 'selector_network_backward',
         'output_backward', 'tf_mlp_backward']
rows = []
for path in D.glob('*/raw/nsys_*.sqlite'):
    stage = path.parents[1].name
    branch = path.stem.removeprefix('nsys_')
    db = sqlite3.connect(path)
    wanted = [branch + '/' + name for name in names]
    query = '''SELECT n.text,count(*),sum(k.end-k.start)/1e6
      FROM NVTX_EVENTS n JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
        ON (r.globalTid=n.globalTid OR n.text=?) AND r.start>=n.start AND r.start<n.end
      JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId
      WHERE n.text IN (''' + ','.join('?' for _ in wanted) + ''') GROUP BY n.text'''
    # total.backward() blocks the caller while autograd workers launch kernels.
    # All captured threads belong to this single isolated benchmark process.
    for name, count, time in db.execute(query, [branch + '/backward', *wanted]).fetchall():
        rows.append({'stage': stage, 'branch': branch, 'component': name,
                     'kernel_calls': count, 'kernel_gpu_ms': time,
                     'launch_threads': 'all_autograd_threads' if name == branch + '/backward' else 'scope_thread'})
    db.close()
(D / 'nsys_components.json').write_text(json.dumps(rows, indent=2))
with (D / 'nsys_components.csv').open('w') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
print(json.dumps(rows, indent=2))

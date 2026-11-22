from pathlib import Path
import subprocess
import sqlite3
import csv
import json

D = Path(__file__).resolve().parent
EVENT = '/home/prmi/zwj/mycartgs/results/change_distill_ablation/20261017-2038/events_supplement/local_loop/event_6207_1'
nsys = '/usr/local/cuda-11.8/bin/nsys'
summary = []
for stage, branch in [('baseline', 'tf'), ('stage_a_grouped', 'tf'), ('stage_b', 'tf'),
                      ('baseline', 'selector'), ('stage_b', 'selector'), ('stage_c', 'selector')]:
    output = D / stage / 'raw' / ('nsys_' + branch)
    args = [nsys, 'profile', '--trace=cuda,nvtx,cublas,osrt', '--sample=none',
            '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--output=' + str(output),
            str(D / stage / 'benchmark_instrumented'), EVENT, str(output) + '_case',
            branch, '5', '8192', '4', '0', '3', '2']
    with Path(str(output) + '.log').open('w') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
    with Path(str(output) + '_stats.csv').open('w') as log:
        subprocess.run([nsys, 'stats', '--report', 'gpukernsum,cudaapisum,nvtxsum', '--format', 'csv',
                        '--force-export=true', str(output) + '.nsys-rep'], stdout=log, stderr=subprocess.STDOUT, check=True)
    db = sqlite3.connect(str(output) + '.sqlite')
    rows = db.execute('''SELECT s.value,count(*),sum(k.end-k.start)/1000.0,avg(k.end-k.start)/1000.0
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
        GROUP BY k.demangledName ORDER BY sum(k.end-k.start) DESC''').fetchall()
    with Path(str(output) + '_kernels.csv').open('w') as file:
        writer = csv.writer(file);writer.writerow(['kernel', 'calls', 'total_gpu_us', 'average_gpu_us']);writer.writerows(rows)
    count, gpu_ms, tiny = db.execute('''SELECT count(*),sum(end-start)/1e6,
        sum(CASE WHEN end-start<10000 THEN 1 ELSE 0 END) FROM CUPTI_ACTIVITY_KIND_KERNEL''').fetchone()
    summary.append({'stage': stage, 'branch': branch, 'kernel_calls': count, 'gpu_kernel_ms': gpu_ms,
                    'tiny_lt_10us': tiny})
    db.close()
    (D / 'nsys_summary.json').write_text(json.dumps(summary, indent=2))
    print('PASS', stage, branch, count, gpu_ms, flush=True)

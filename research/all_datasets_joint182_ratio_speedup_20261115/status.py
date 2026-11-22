"""只读巡检队列、实际进程和日志增量。"""
from pathlib import Path
import json

r = Path(__file__).resolve().parent
q = json.loads((r/'QUEUE.json').read_text())
print('STATE', q['state'])
print('FIRST_PASS_GROUPS', sum(x['formal'] == 'complete' for x in q['scenes']))
print('THREE_ROUND_GROUPS', sum(x.get('remaining') in ['complete', 'already_in_formal'] for x in q['scenes']))
print('ACTIVE', q.get('active_scene'), q.get('active_phase'))
if q.get('active_scene'):
    name = q['active_scene'].replace('/', '__')+'_'+q['active_phase']
    j = json.loads((r/'logs'/f'{name}_job.json').read_text())
    path = Path(f"/proc/{j['child_pid']}/stat")
    live = path.exists() and path.read_text().rpartition(')')[2].split()[19] == j['child_start_ticks']
    print('WORKER_LIVE', live, 'RETURN', j.get('returncode'))
    output = r/j['output_dir']/'TIMINGS.csv'
    print('CSV_BYTES', output.stat().st_size if output.exists() else 0)
    lines = (r/'logs'/f'{name}.log').read_text().splitlines()
    for line in lines[-2:]:
        print('LOG', line)
if q['state'] == 'failed':
    print('ERROR', q.get('error'))
    print((r/'logs/controller_passes.log').read_text()[-3000:])

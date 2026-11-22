"""首轮结束后只重跑已修正的完全重复轨迹失败项。"""
from pathlib import Path
import fcntl
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
controller = json.loads((ROOT/'reports/CONTROLLER.json').read_text())
while True:
    stat = Path(f"/proc/{controller['pid']}/stat")
    if not stat.exists():
        break
    parts=stat.read_text().rpartition(')')[2].split()
    if parts[0]=='Z' or int(parts[19])!=controller['start_ticks']:
        break
    time.sleep(10)

with (ROOT/'reports/CONTROLLER.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=ROOT/'reports/BATCH_STATUS.json'
    state=json.loads(path.read_text())
    assert state['status']=='finished'
    target=next(j for j in state['scenes'] if j['id']=='TUM_RGBD/freiburg1_room/mono')
    assert target['status']=='failed' and len(target['tasks'])==1
    assert 'Duplicate trajectory/RGB association' in Path(target['tasks'][0]['log']).read_text()
    (ROOT/'reports/BATCH_STATUS_BEFORE_TRAJECTORY_RETRY.json').write_text(json.dumps(state,indent=2,ensure_ascii=False))
    target['status']='pending'
    target['previous_error']=target.pop('error')
    state.update(status='running',retry_reason='exact duplicate trajectory entries removed; distinct poses still rejected')
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2,ensure_ascii=False));temporary.replace(path)

code=subprocess.call([sys.executable,'-u',str(HERE/'supervise.py')],cwd=ROOT)
print('TRAJECTORY_RETRY_SUPERVISOR_EXIT',code,flush=True)
sys.exit(code)

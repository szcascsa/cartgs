"""只读进度核查，记录真实CSV增量、进程身份与资源。"""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]


def identity(record):
    path = Path(f'/proc/{record["pid"]}/stat')
    if not path.exists():
        return dict(alive=False)
    parts = path.read_text().rpartition(')')[2].split()
    return dict(alive=parts[0]!='Z' and int(parts[19])==int(record['start_ticks'])
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id'],state=parts[0])


def main():
    state = json.loads((ROOT/'reports/BATCH_STATUS.json').read_text())
    result = dict(status=state['status'],time=time.time(),
        completed=sum(j['status']=='complete' for j in state['scenes']),
        failed=sum(j['status']=='failed' for j in state['scenes']),reused=sum(j.get('reused',False) for j in state['scenes']),
        controller=identity(json.loads((ROOT/'reports/CONTROLLER.json').read_text())),
        disk_free_GiB=round(shutil.disk_usage(ROOT).free/2**30,2))
    running = [j for j in state['scenes'] if j['status']=='running']
    result['active'] = []
    for job in running:
        task = job['tasks'][-1]
        item = dict(scene=job['id'],phase=task['name'],pid=task['pid'],process=identity(task))
        trace = ROOT/'models'/job['slug']/task['name']/'training.csv'
        if trace.exists():
            with trace.open('rb') as f:
                f.seek(max(0,trace.stat().st_size-8192))
                lines = f.read().decode(errors='replace').splitlines()
            if len(lines)>1:
                item.update(step=int(lines[-1].split(',')[0]),trace_age_s=round(time.time()-trace.stat().st_mtime,1))
        result['active'].append(item)
    if running:
        job = running[-1]
        task = job['tasks'][-1]
        result.update(scene=job['id'],phase=task['name'],attempt=task['attempt'],process=identity(task))
        log = Path(task['log'])
        result['log_tail'] = log.read_text(errors='replace').splitlines()[-2:]
        result['log_age_s'] = round(time.time()-log.stat().st_mtime,1)
        csv = ROOT/'models'/job['slug']/task['name']/'training.csv'
        if csv.exists():
            with csv.open('rb') as f:
                f.seek(max(0,csv.stat().st_size-8192))
                lines = f.read().decode(errors='replace').splitlines()
            if len(lines)>1:
                result['step'] = int(lines[-1].split(',')[0])
                result['trace_age_s'] = round(time.time()-csv.stat().st_mtime,1)
    result['failed_scenes'] = [dict(id=j['id'],error=j.get('error')) for j in state['scenes'] if j['status']=='failed']
    result['gpu'] = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
    displayed = result
    if '--brief' in sys.argv:
        displayed = {key:result.get(key) for key in ['status','completed','failed','scene','phase','step','trace_age_s','gpu']}
        displayed['controller_alive'] = result['controller']['alive']
        displayed['failed_scenes'] = result['failed_scenes']
        displayed['active'] = result['active']
    if '--compact' in sys.argv:
        print('PROG',result['completed'],result['failed'],result.get('scene','end').replace('Replica/',''),
            result.get('phase','end'),result.get('step',0),result.get('trace_age_s',0),
            'alive' if result['controller']['alive'] else result['status'],flush=True)
    else:
        print(json.dumps(displayed,ensure_ascii=False),flush=True)
    with (ROOT/'reports/PROGRESS_HISTORY.jsonl').open('a') as stream:
        stream.write(json.dumps(result,ensure_ascii=False)+'\n')


if __name__=='__main__':
    main()

"""本轮串行实验计划：文件锁、PID身份、干扰重测和预算截止。"""
from pathlib import Path
from datetime import datetime
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def main():
    plan_path=Path(sys.argv[1]);plan=json.loads(plan_path.read_text())
    lock=(HERE/'reports/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=HERE/'reports'/(plan['name']+'_CONTROLLER.json');assert not path.exists()
    state=dict(state='running',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rpartition(')')[2].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_at=time.time(),plan=str(plan_path),accepted=[],rejected=[])
    def save():
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(path)
    save()
    try:
        for task in plan['jobs']:
            state['current_task']=task['name'];save()
            for attempt in range(1,5):
                protocol=json.loads((HERE/'reports/PROTOCOL.json').read_text())
                cutoff=protocol['deadline'] if plan.get('final_validation') else protocol['final_validation_reserve']
                assert time.time()<datetime.fromisoformat(cutoff).timestamp(),'已到本阶段预算截止'
                assert shutil.disk_usage(HERE).free>=5*(1<<30),'可用磁盘低于5GiB'
                name=task['name']+f'_a{attempt}'
                command=[value.replace('{job}',name).replace('{python}',sys.executable).replace('{root}',str(HERE)) for value in task['command']]
                subprocess.run([sys.executable,'-B','launch.py']+(['--measure'] if task.get('measure') else [])+[name,*command],cwd=HERE,check=True)
                while True:
                    result=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text())
                    if result['state'] in ['complete','failed']:break
                    time.sleep(1)
                assert result['returncode']==0,(name,result)
                state['accepted' if result['measurement_valid'] else 'rejected'].append(name);save()
                if result['measurement_valid']:break
            else:raise RuntimeError((task['name'],'持续资源干扰，无有效测量'))
        state.update(state='complete',ended_at=time.time());save()
    except BaseException as error:
        state.update(state='failed',error=repr(error),ended_at=time.time());save();raise
    print('STAGE_COMPLETE',plan['name'],len(state['accepted']),flush=True)


if __name__=='__main__':main()

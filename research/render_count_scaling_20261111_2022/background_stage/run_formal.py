"""每段接受规则独立于速度：外部负载干扰整段重测，绝不按结果筛选。"""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def save(value):
    (HERE/'reports/FORMAL_CONTROLLER.json').write_text(json.dumps(value,indent=2))


def main():
    lock=(HERE/'reports/formal.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rpartition(')')[2].split()[19],
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state='running',
                  started_at=time.time(),accepted=[],rejected=[])
    save(identity)
    for iteration in (1,2):
        for group in (range(16) if iteration==1 else reversed(range(16))):
            accepted=False
            for attempt in range(1,4):
                name=f'balanced_r{iteration}_g{group}_a{attempt}'
                command=[sys.executable,'-B','launch.py','--measure',name,sys.executable,'-u','-B','formal.py',
                         '--output',name,'--round',str(iteration),'--group',str(group)]
                if iteration==1:command.append('--diagnostics')
                subprocess.run(command,cwd=HERE,check=True)
                path=HERE/'reports/jobs'/(name+'.json')
                while True:
                    status=json.loads(path.read_text())
                    if status['state'] in ('complete','failed'):break
                    time.sleep(1)
                assert status['returncode']==0, (name,status)
                if status['measurement_valid']:
                    identity['accepted'].append(name);accepted=True;save(identity);break
                identity['rejected'].append(name);save(identity)
            assert accepted,(iteration,group,'external load repeated')
    identity.update(state='complete',ended_at=time.time())
    save(identity)
    print('FORMAL_COMPLETE',len(identity['accepted']),len(identity['rejected']),flush=True)


if __name__=='__main__':main()

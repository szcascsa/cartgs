"""独占本轮控制器；外部负载污染整段重测，与性能结果无关。"""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def main():
    lock=(HERE/'reports/formal.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=HERE/'reports/FORMAL_CONTROLLER.json';assert not p.exists()
    state=dict(state='running',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rpartition(')')[2].split()[19],
               boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_at=time.time(),accepted=[],rejected=[])
    def save():p.write_text(json.dumps(state,indent=2))
    save()
    for iteration in (1,2):
        for group in (range(16) if iteration==1 else reversed(range(16))):
            valid=False
            for attempt in range(1,5):
                name=f'formal_r{iteration}_g{group}_a{attempt}'
                command=[sys.executable,'-B','launch.py','--measure',name,sys.executable,'-u','-B','formal.py',
                         '--output',name,'--round',str(iteration),'--group',str(group)]
                if iteration==1:command.append('--diagnostics')
                subprocess.run(command,cwd=HERE,check=True)
                job=HERE/'reports/jobs'/(name+'.json')
                while True:
                    value=json.loads(job.read_text())
                    if value['state'] in ('complete','failed'):break
                    time.sleep(1)
                assert value['returncode']==0,(name,value)
                if value['measurement_valid']:
                    state['accepted'].append(name);valid=True;save();break
                state['rejected'].append(name);save()
            assert valid,(iteration,group,'external load')
    state.update(state='complete',ended_at=time.time());save()
    print('FORMAL_COMPLETE',len(state['accepted']),len(state['rejected']),flush=True)


if __name__=='__main__':main()

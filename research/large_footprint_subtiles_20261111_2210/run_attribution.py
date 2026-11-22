"""仅追加同二进制对照归因，不改变已冻结候选或此前正式测量门限。"""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def main():
    lock=(HERE/'reports/attribution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=HERE/'reports/ATTRIBUTION_CONTROLLER.json';assert not path.exists()
    state=dict(state='running',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rpartition(')')[2].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_at=time.time(),accepted=[],rejected=[])
    def save():path.write_text(json.dumps(state,indent=2))
    save()
    try:
        for iteration in (1,2):
            for group in (range(16) if iteration==1 else reversed(range(16))):
                valid=False
                for attempt in range(1,5):
                    name=f'attr_r{iteration}_g{group}_a{attempt}'
                    subprocess.run([sys.executable,'-B','launch.py','--measure',name,sys.executable,'-u','-B','attribution.py',
                        '--output',name,'--round',str(iteration),'--group',str(group)],cwd=HERE,check=True)
                    while True:
                        job=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text())
                        if job['state'] in ('complete','failed'):break
                        time.sleep(1)
                    assert job['returncode']==0,(name,job)
                    if job['measurement_valid']:
                        state['accepted'].append(name);valid=True;save();break
                    state['rejected'].append(name);save()
                assert valid,(iteration,group,'external load')
        state.update(state='complete',ended_at=time.time());save()
    except BaseException as error:
        state.update(state='failed',error=repr(error),ended_at=time.time());save();raise
    print('ATTRIBUTION_COMPLETE',len(state['accepted']),len(state['rejected']),flush=True)


if __name__=='__main__':main()

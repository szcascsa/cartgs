"""只补正式协议中的空/单点和嵌套固定预算；不重复已覆盖的质量条件。"""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def main():
    lock=(HERE/'reports/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=HERE/'reports/QUALITY_SUPPLEMENT_CONTROLLER.json';assert not path.exists()
    state=dict(state='running',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rpartition(')')[2].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),accepted=[],started_at=time.time())
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(path)
    save()
    try:
        for workload,conditions in [('lod','fixed0,fixed1'),('nested','fixed0,fixed1,fixed1000,fixed5000')]:
            for group in range(16):
                name=f'quality_extra_{workload}_g{group}'
                command=[sys.executable,'-B','launch.py',name,sys.executable,'-u','-B','quality.py','--output',name,
                    '--modes','37,38,39,40','--group',str(group),'--conditions',conditions]+(['--nested'] if workload=='nested' else [])
                subprocess.run(command,cwd=HERE,check=True)
                while True:
                    job=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text())
                    if job['state'] in ('complete','failed'):break
                    time.sleep(1)
                assert job['returncode']==0,(name,job)
                state['accepted'].append(name);save()
        state.update(state='complete',ended_at=time.time());save()
    except BaseException as error:
        state.update(state='failed',error=repr(error),ended_at=time.time());save();raise


if __name__=='__main__':main()

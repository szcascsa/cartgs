"""开发集双轮联合测速与代表性Profile；持有本轮唯一调度锁。"""
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
    path=HERE/'reports/JOINT_DEVELOPMENT_CONTROLLER.json';assert not path.exists()
    state=dict(state='running',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rpartition(')')[2].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_at=time.time(),accepted=[],rejected=[])
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(path)
    def job(label,command,measure=False):
        for attempt in range(1,5):
            name=f'{label}_a{attempt}'
            subprocess.run([sys.executable,'-B','launch.py']+(['--measure'] if measure else [])+[name,*command(name)],cwd=HERE,check=True)
            while True:
                result=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text())
                if result['state'] in ['complete','failed']:break
                time.sleep(1)
            assert result['returncode']==0,(name,result)
            state['accepted' if result['measurement_valid'] else 'rejected'].append(name);save()
            if result['measurement_valid']:return
        raise RuntimeError((label,'持续外部干扰，无有效测量'))
    save()
    try:
        assert json.loads((HERE/'reports/jobs/pending_smoke01.json').read_text())['returncode']==0
        for iteration in [1,2]:
            for group in ([0,1] if iteration==1 else [1,0]):
                job(f'joint_dev_r{iteration}_g{group}',lambda name:[sys.executable,'-u','-B','joint_benchmark.py',
                    '--output',name,'--round',str(iteration),'--group',str(group),'--pending','--tune','--quick'],True)
        job('render_profile01',lambda name:['/usr/local/cuda-11.8/bin/nsys','profile','--trace=cuda,nvtx',
            '--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--sample=none',
            '--force-overwrite=false','--output',str(HERE/'reports'/name),sys.executable,'-u','-B','profile_render.py'])
        name=state['accepted'][-1]
        subprocess.run(['/usr/local/cuda-11.8/bin/nsys','export','--type','sqlite','--output',str(HERE/'reports'/(name+'.sqlite')),
            str(HERE/'reports'/(name+'.nsys-rep'))],check=True)
        state.update(state='complete',ended_at=time.time());save()
    except BaseException as error:
        state.update(state='failed',error=repr(error),ended_at=time.time());save();raise


if __name__=='__main__':main()

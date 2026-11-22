"""完成正式计时后再运行压力、生命周期、Sanitizer和Nsight，不扰动计时。"""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PYTHON=sys.executable


def main():
    lock=(HERE/'reports/post.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=HERE/'reports/POST_CHECKS.json';assert not path.exists()
    state=dict(state='waiting',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rpartition(')')[2].split()[19],
               boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),jobs=[])
    def save():path.write_text(json.dumps(state,indent=2))
    save()
    while True:
        c=json.loads((HERE/'reports/FORMAL_CONTROLLER.json').read_text())
        if c['state']=='complete':break
        time.sleep(2)
    state['state']='running';save()
    def job(name,command,measured=False):
        args=[PYTHON,'-B','launch.py']+(['--measure'] if measured else [])+[name,*command]
        subprocess.run(args,cwd=HERE,check=True)
        while True:
            j=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text())
            if j['state'] in ('complete','failed'):break
            time.sleep(1)
        state['jobs'].append(name);save()
        assert j['returncode']==0,(name,j)
        return j
    for attempt in range(1,5):
        name=f'stress_clean{attempt:02d}'
        j=job(name,[PYTHON,'-u','-B','stress.py','--output',name,'--modes','49,18,20,1'],True)
        if j['measurement_valid']:
            state['stress']=name;save();break
    assert state.get('stress')
    job('lifecycle01',[PYTHON,'-u','-B','lifecycle.py','--output','lifecycle01','--modes','49,18'])
    for tool in ('memcheck','initcheck','synccheck'):
        name=tool+'01'
        job(name,['/usr/local/cuda-11.8/bin/compute-sanitizer','--tool',tool,'--error-exitcode','87',
                  PYTHON,'-u','-B','lifecycle.py','--output',name,'--modes','49,18','--sanitizer'])
        assert 'ERROR SUMMARY: 0 errors' in (HERE/'logs'/(name+'.log')).read_text(errors='replace'),name
    for attempt in range(1,4):
        name=f'profile{attempt:02d}'
        j=job(name,['/usr/local/cuda-11.8/bin/nsys','profile','--trace=cuda,nvtx','--cuda-graph-trace=node',
                    '--capture-range=cudaProfilerApi','--capture-range-end=stop','--sample=none','--force-overwrite=false',
                    '-o',str(HERE/'reports'/name),PYTHON,'-u','-B','profile.py'])
        if j['foreign_active_samples']==0:
            state['profile']=name;save();break
    assert state.get('profile')
    name=state['profile']
    subprocess.run(['/usr/local/cuda-11.8/bin/nsys','export','--type','sqlite','--output',str(HERE/'reports'/(name+'.sqlite')),
                    str(HERE/'reports'/(name+'.nsys-rep'))],check=True)
    subprocess.run([PYTHON,'-B','profile_analyze.py',name],cwd=HERE,check=True)
    state.update(state='complete',ended_at=time.time());save()
    print('POST_CHECKS_COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        path=HERE/'reports/POST_CHECKS.json'
        if path.exists():
            value=json.loads(path.read_text())
            if value.get('pid')==os.getpid():
                value.update(state='failed',error=repr(error),ended_at=time.time())
                path.write_text(json.dumps(value,indent=2))
        raise

"""最终Profile与Sanitizer顺序运行，记录进程身份，失败保留现场。"""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
STAGE=ROOT/'background_stage'
PYTHON=sys.executable


def run_job(name,command):
    subprocess.run([PYTHON,'-B','launch.py',name,*command],cwd=ROOT,check=True)
    p=ROOT/'reports/jobs'/(name+'.json')
    while True:
        x=json.loads(p.read_text())
        if x['state'] in ('complete','failed'):break
        time.sleep(1)
    assert x['returncode']==0,(name,x)
    return x


def main():
    lock=(STAGE/'reports/post.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=STAGE/'reports/POST_CHECKS.json'
    state=dict(state='running',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rpartition(')')[2].split()[19],
               boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),jobs=[],profile=None)
    def save():p.write_text(json.dumps(state,indent=2))
    save()
    for attempt in (1,2,3):
        name=f'bg_profile{attempt:02d}'
        trace=STAGE/'reports'/name
        job=run_job(name,['/usr/local/cuda-11.8/bin/nsys','profile','--trace=cuda,nvtx','--cuda-graph-trace=node',
            '--capture-range=cudaProfilerApi','--capture-range-end=stop','--sample=none','--force-overwrite=false',
            '-o',str(trace),PYTHON,'-u','-B','background_profile.py'])
        state['jobs'].append(name);save()
        if job['foreign_active_samples']==0:
            state['profile']=name;save();break
    assert state['profile'],'No undisturbed Profile run'
    for tool in ('memcheck','initcheck','synccheck'):
        name='bg_'+tool+'01'
        run_job(name,['/usr/local/cuda-11.8/bin/compute-sanitizer','--tool',tool,'--error-exitcode','87',
                     PYTHON,'-u','-B','background_validation.py','--output',tool+'01','--sanitizer'])
        text=(ROOT/'logs'/(name+'.log')).read_text(errors='replace')
        assert ('ERROR SUMMARY: 0 errors' in text or 'ERROR SUMMARY: 0' in text),name
        state['jobs'].append(name);save()
    name=state['profile']
    subprocess.run(['/usr/local/cuda-11.8/bin/nsys','export','--type','sqlite','--output',str(STAGE/'reports'/(name+'.sqlite')),
                    str(STAGE/'reports'/(name+'.nsys-rep'))],check=True)
    subprocess.run([PYTHON,'-B','profile_analyze.py',str(STAGE.relative_to(ROOT)/'reports'/(name+'.sqlite')),
                    '--out',str(STAGE.relative_to(ROOT)/'reports/PROFILE')],cwd=ROOT,check=True)
    state['state']='complete';state['completed_at']=time.time();save()
    print('POST_CHECKS_COMPLETE',flush=True)


if __name__=='__main__':main()

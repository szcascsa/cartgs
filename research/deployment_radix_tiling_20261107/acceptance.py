"""串行验收与测量，不并发运行本轮 GPU 任务。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PYTHON='/home/prmi/anaconda3/envs/zwj/bin/python'
CUDA='/usr/local/cuda-11.8/bin'


def run(name,command,timeout=900):
    state=dict(stage=name,status='running',started=time.time(),command=command)
    (ROOT/'reports/ACCEPTANCE_STATUS.json').write_text(json.dumps(state,indent=2))
    with (ROOT/'logs'/f'{name}.log').open('w') as out:
        process=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(pid=process.pid,start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        (ROOT/'reports/ACCEPTANCE_STATUS.json').write_text(json.dumps(state,indent=2))
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Terminate only this explicitly-created test process group.
            import signal
            os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=30)
            code=124
    state.update(status='passed' if code==0 else 'failed',exit_code=code,finished=time.time())
    (ROOT/'reports/ACCEPTANCE_STATUS.json').write_text(json.dumps(state,indent=2))
    print(name,code,flush=True)
    if code:raise RuntimeError(state)


def main():
    assert json.loads((ROOT/'reports/VALIDATION.json').read_text())['passed']
    stages=[('radix_final',[PYTHON,'-u',str(HERE/'radix_test.py')]),
        ('components_final',[PYTHON,'-u',str(HERE/'components.py')]),
        ('extra_validation',[PYTHON,'-u',str(HERE/'extra_checks.py')]),
        ('extra_replica',[PYTHON,'-u',str(HERE/'extra_checks.py'),'replica']),
        ('control_audit',[PYTHON,'-u',str(HERE/'control_audit.py')]),
        ('control_replica',[PYTHON,'-u',str(HERE/'control_audit.py'),'replica']),
        ('trajectory_tum',[PYTHON,'-u',str(HERE/'trajectory.py'),'tum']),
        ('trajectory_replica',[PYTHON,'-u',str(HERE/'trajectory.py'),'replica'])]
    for name,cmd in stages:run(name,cmd)
    for repeat in [1,2]:
        for group in ['tum','replica']:
            run(f'benchmark_{group}_{repeat}',[PYTHON,'-u',str(HERE/'benchmark.py'),str(repeat),'792' if group=='tum' else '411',group])
    for group in ['tum','replica']:
        for tag in ['v7','optimized']:run(f'memory_{group}_{tag}',[PYTHON,'-u',str(HERE/'memory.py'),tag,group])
    run('sanitizer',[CUDA+'/compute-sanitizer','--tool','memcheck','--error-exitcode','97',PYTHON,'-u',str(HERE/'smoke.py')])
    run('sanitizer_replica',[CUDA+'/compute-sanitizer','--tool','memcheck','--error-exitcode','97',PYTHON,'-u',str(HERE/'smoke.py'),'replica'])
    run('ablation',[PYTHON,'-u',str(HERE/'ablation.py')])
    run('nsight',[CUDA+'/nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
        '-o',str(ROOT/'reports/PROFILE'),PYTHON,'-u',str(HERE/'profile_runtime.py')])
    run('nsight_export',[CUDA+'/nsys','export','--type','sqlite','--output',str(ROOT/'reports/PROFILE.sqlite'),str(ROOT/'reports/PROFILE.nsys-rep')])
    run('nsight_analysis',[PYTHON,'-u',str(HERE/'analyze_profile.py')])
    (ROOT/'reports/ACCEPTANCE_COMPLETE.json').write_text(json.dumps(dict(passed=True,finished=time.time())))


if __name__=='__main__':main()

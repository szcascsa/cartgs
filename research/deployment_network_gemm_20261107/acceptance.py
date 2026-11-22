"""串行执行本轮GPU验收，避免自身并发影响测量。"""
from pathlib import Path
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
PYTHON='/home/prmi/anaconda3/envs/zwj/bin/python';CUDA='/usr/local/cuda-11.8/bin'


def run(name,command,timeout=900):
    state=dict(stage=name,status='running',started=time.time(),command=command)
    with (ROOT/'logs'/f'{name}.log').open('w') as out:
        p=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        (ROOT/'reports/ACCEPTANCE_STATUS.json').write_text(json.dumps(state,indent=2))
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30);code=124
    state.update(status='passed' if code==0 else 'failed',exit_code=code,finished=time.time())
    (ROOT/'reports/ACCEPTANCE_STATUS.json').write_text(json.dumps(state,indent=2));print(name,code,flush=True)
    if code:raise RuntimeError(state)


def main():
    assert json.loads((ROOT/'reports/VALIDATION.json').read_text())['groups']==5
    tests=[('register_final',['register_test.py']),('selector_final',['selector_test.py']),('synthetic',['synthetic_checks.py']),
        ('extra_tum',['extra_checks.py']),('extra_replica',['extra_checks.py','replica']),
        ('entry',['entry_checks.py']),('control_tum',['control_audit.py']),('control_replica',['control_audit.py','replica']),
        ('trajectory_tum',['trajectory.py','tum']),('trajectory_replica',['trajectory.py','replica'])]
    for name,args in tests:run(name,[PYTHON,'-u',str(HERE/args[0]),*args[1:]])
    for repeat in [1,2]:
        for group in ['tum','replica']:run(f'benchmark_{group}_{repeat}',[PYTHON,'-u',str(HERE/'benchmark.py'),str(repeat),group])
    run('ablation',[PYTHON,'-u',str(HERE/'ablation.py')])
    for group in ['tum','replica']:
        for tag in ['v11','optimized']:run(f'memory_{group}_{tag}',[PYTHON,'-u',str(HERE/'memory.py'),tag,group])
    for group in ['tum','replica']:run(f'sanitizer_{group}',[CUDA+'/compute-sanitizer','--tool','memcheck','--error-exitcode','97',PYTHON,'-u',str(HERE/'smoke.py'),group])
    run('nsight',[CUDA+'/nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
        '-o',str(ROOT/'reports/PROFILE'),PYTHON,'-u',str(HERE/'profile_runtime.py')])
    run('nsight_export',[CUDA+'/nsys','export','--type','sqlite','--output',str(ROOT/'reports/PROFILE.sqlite'),str(ROOT/'reports/PROFILE.nsys-rep')])
    run('nsight_analysis',[PYTHON,'-u',str(HERE/'analyze_profile.py')])
    run('nsight_components',[CUDA+'/nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
        '-o',str(ROOT/'reports/COMPONENT_PROFILE'),PYTHON,'-u',str(HERE/'profile_components.py')])
    run('component_export',[CUDA+'/nsys','export','--type','sqlite','--output',str(ROOT/'reports/COMPONENT_PROFILE.sqlite'),str(ROOT/'reports/COMPONENT_PROFILE.nsys-rep')])
    run('component_analysis',[PYTHON,'-u',str(HERE/'analyze_profile.py'),'COMPONENT_PROFILE'])
    (ROOT/'reports/ACCEPTANCE_COMPLETE.json').write_text(json.dumps(dict(passed=True,finished=time.time())))


if __name__=='__main__':main()

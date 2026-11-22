from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PY=sys.executable


def main():
    expected=json.loads((ROOT/'reports/RUNNER_STATUS.json').read_text())
    while True:
        state=json.loads((ROOT/'reports/RUNNER_STATUS.json').read_text())
        assert state['pid']==expected['pid'] and state['start_ticks']==expected['start_ticks'] and state['boot_id']==expected['boot_id']
        assert all(t.get('exit',0)==0 for t in state['tasks']),state
        if state.get('complete'):break
        process=Path(f'/proc/{state["pid"]}/stat')
        assert process.exists() and process.read_text().split()[21]==state['start_ticks'],'Validation controller exited'
        time.sleep(3)
    tasks=[('cli',[PY,'-u',str(HERE/'cli_smoke.py')]),
        ('sanitize_tum',['/usr/local/cuda-11.8/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','97',PY,'-u',str(HERE/'extra_checks.py'),'tum']),
        ('sanitize_replica',['/usr/local/cuda-11.8/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','97',PY,'-u',str(HERE/'extra_checks.py'),'replica']),
        ('initcheck_tum',['/usr/local/cuda-11.8/bin/compute-sanitizer','--tool','initcheck','--error-exitcode','98',PY,'-u',str(HERE/'extra_checks.py'),'tum']),
        ('initcheck_replica',['/usr/local/cuda-11.8/bin/compute-sanitizer','--tool','initcheck','--error-exitcode','98',PY,'-u',str(HERE/'extra_checks.py'),'replica']),
        ('idle_windows',[PY,'-u',str(HERE/'idle_windows.py')]),
        ('profile',['/usr/local/cuda-11.8/bin/nsys','profile','--force-overwrite=true','--trace=cuda,nvtx,osrt',
            '--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--output='+str(ROOT/'reports/PREPARE_PROFILE'),
            PY,'-u',str(HERE/'benchmark.py'),'10','--profile']),
        ('export',['/usr/local/cuda-11.8/bin/nsys','export','--type=sqlite','--force-overwrite=true',
            '--output='+str(ROOT/'reports/PREPARE_PROFILE.sqlite'),str(ROOT/'reports/PREPARE_PROFILE.nsys-rep')]),
        ('analysis',[PY,'-u',str(HERE/'profile_graph.py')]),
        ('integrity',[PY,'-u',str(HERE/'integrity.py')])]
    status=dict(pid=os.getpid(),tasks=[])
    for name,command in tasks:
        with (ROOT/'logs'/f'finish_{name}.log').open('w') as out:
            p=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT)
            row=dict(name=name,pid=p.pid,started=time.time());status['tasks'].append(row)
            (ROOT/'reports/FINISH_STATUS.json').write_text(json.dumps(status,indent=2))
            code=p.wait();row.update(exit=code,ended=time.time())
        (ROOT/'reports/FINISH_STATUS.json').write_text(json.dumps(status,indent=2))
        print('FINISH',name,code,flush=True)
        if code:raise SystemExit(code)
    status['complete']=True;(ROOT/'reports/FINISH_STATUS.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__':main()

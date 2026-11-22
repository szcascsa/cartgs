"""本轮训练完成后串行评价与诊断，失败保留现场，不启动重复控制器。"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from common import ROOT,HERE,PYTHON,identity,save,environment


def main():
    lock=(ROOT/'reports/finish.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(state='waiting_training',controller=identity(os.getpid()));save(ROOT/'reports/FINISH.json',state)
    while True:
        pipeline=json.loads((ROOT/'reports/PIPELINE.json').read_text())
        if pipeline['state']=='training_complete':break
        assert pipeline['state']=='running',pipeline
        process=pipeline['controller'];pid=process['pid']
        assert Path(f'/proc/{pid}/stat').exists() and identity(pid)==process,'训练控制器退出，需检查现场'
        time.sleep(5)
    for name in ['postprocess.py','validate_final_conditioning.py','diagnose_tail.py','summarize_tail.py','master_drift.py']:
        state.update(state='running',stage=name);save(ROOT/'reports/FINISH.json',state)
        with (ROOT/'logs'/('finish_'+name+'.log')).open('w') as f:
            p=subprocess.run([PYTHON,'-u',str(HERE/name)],cwd=ROOT,env=environment(),stdout=f,stderr=subprocess.STDOUT)
        assert p.returncode==0,name+'失败，检查对应日志'
    state.update(state='metrics_diagnostics_complete',stage=None);save(ROOT/'reports/FINISH.json',state)


if __name__=='__main__':
    try:main()
    except Exception as error:
        p=ROOT/'reports/FINISH.json';v=json.loads(p.read_text());v.update(state='failed',error=repr(error));save(p,v);raise

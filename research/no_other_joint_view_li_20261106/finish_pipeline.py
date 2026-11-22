"""训练验收完成后串行速度、质量和统计，失败保留现场。"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from common import ROOT,HERE,PYTHON,save,identity,environment


def main():
    lock=(ROOT/'reports/finish.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(state='waiting_training',controller=identity(os.getpid()));save(ROOT/'reports/FINISH.json',state)
    while True:
        training=json.loads((ROOT/'reports/TRAINING_STATUS.json').read_text())
        if training['state']=='training_complete':break
        assert training['state']!='failed',training
        assert identity(training['controller']['pid'])==training['controller'],'训练控制器退出'
        time.sleep(5)
    for stage in ['speed.py','evaluate.py','summarize.py']:
        state.update(state='running',stage=stage);save(ROOT/'reports/FINISH.json',state)
        with (ROOT/'logs'/('finish_'+stage+'.log')).open('w') as log:
            result=subprocess.run([PYTHON,'-u',str(HERE/stage)],cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,stage+'失败，检查对应日志'
    state.update(state='metrics_complete',stage=None);save(ROOT/'reports/FINISH.json',state)


if __name__=='__main__':
    try:main()
    except Exception as error:
        path=ROOT/'reports/FINISH.json';state=json.loads(path.read_text());state.update(state='failed',error=repr(error));save(path,state);raise

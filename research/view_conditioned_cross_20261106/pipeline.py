"""串行构建、验收和正式训练；任何失败保留现场。"""
import fcntl
import os
import subprocess
import time
import sys
import json
from common import ROOT,HERE,PYTHON,identity,save,sha


def main():
    lock=(ROOT/'reports/pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(controller=identity(os.getpid()),state='running')
    stages=['preflight.py','build_isolated.py','validate_native.py','validate_inference.py','run_full.py']
    if sys.argv[1:] in [['resume-validation'],['resume-inference']]:
        build=json.loads((ROOT/'reports/BUILD_HASHES.json').read_text());assert all(sha(ROOT/n)==h for n,h in build.items())
        stages=stages[2 if sys.argv[1]=='resume-validation' else 3:]
    else:assert not sys.argv[1:]
    for name in stages:
        state.update(stage=name,updated_unix=time.time());save(ROOT/'reports/PIPELINE.json',state)
        with (ROOT/'logs'/f'pipeline_{name}.log').open('w') as f:
            p=subprocess.Popen([PYTHON,'-u',str(HERE/name)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            code=p.wait()
        if code:
            state.update(state='failed',returncode=code);save(ROOT/'reports/PIPELINE.json',state)
            raise RuntimeError(f'{name}失败，检查对应日志')
    state.update(state='training_complete',updated_unix=time.time());save(ROOT/'reports/PIPELINE.json',state)


if __name__=='__main__':main()

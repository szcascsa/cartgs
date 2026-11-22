"""独立编译CPU核心与CUDA Event扩展，保留日志和检查结果。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
os.environ['CUDA_HOME']='/usr/local/cuda-11.8'
os.environ['MAX_JOBS']='2'
os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
commands=[['g++','-std=c++17','-O2','-I',str(ROOT/'source'),str(ROOT/'tests/test_latency_pid_core.cpp'),'-o',str(ROOT/'build/test_pid_core')],
          [str(ROOT/'build/test_pid_core')],
          [sys.executable,'-B','-u',str(ROOT/'tests/test_gpu.py')]]
results=[]
for index,command in enumerate(commands):
    begin=time.time()
    with (ROOT/'logs'/f'test_{index}.log').open('w') as stream:
        run=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    results.append(dict(command=command,returncode=run.returncode,elapsed_s=time.time()-begin))
    print('TEST',index,run.returncode,flush=True)
    (ROOT/'reports/TEST_RESULTS.json').write_text(json.dumps(results,indent=2))
    if run.returncode:raise SystemExit(run.returncode)
print('BUILD_AND_TEST_PASS',flush=True)

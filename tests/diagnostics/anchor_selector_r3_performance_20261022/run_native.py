"""正式计时与Nsight分开；其他GPU进程只记录，不终止。"""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/prmi/zwj/anchor-selector-r3-performance-20261022')
SOURCE = Path('/home/prmi/zwj/private-compensation-online-20261022/source')
CP = SOURCE.parent/'r3_independent14_seed2026/7465_shutdown/ply'
env = dict(os.environ, OMP_NUM_THREADS='2', LD_LIBRARY_PATH=str(SOURCE/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
cmd = [str(ROOT/'native_benchmark'), str(CP), '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz']
samples = []
for name, profile in [('native1',False),('native2',False),('nsight',True)]:
    case = cmd+[str(ROOT/name), '4' if profile else '61']
    if profile:
        case = ['/usr/local/cuda-11.8/bin/nsys','profile','--trace=cuda,nvtx,osrt',
                '--sample=none','--cpuctxsw=none','--force-overwrite=true','-o',str(ROOT/'native_profile')]+case
    start = time.monotonic()
    with (ROOT/(name+'.log')).open('w') as log:
        p = subprocess.Popen(case, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        while p.poll() is None:
            gpu = subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw',
                                           '--format=csv,noheader,nounits'],text=True).strip()
            apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                                            '--format=csv,noheader,nounits'],text=True).strip()
            samples.append(dict(run=name,seconds=time.monotonic()-start,gpu=gpu,apps=apps))
            time.sleep(.5)
        (ROOT/'native_gpu_samples.json').write_text(json.dumps(samples,indent=2))
        assert p.returncode==0,(name,p.returncode)
    print('FINISHED',name,time.monotonic()-start,flush=True)
if (ROOT/'native_profile.nsys-rep').exists():
    subprocess.run(['/usr/local/cuda-11.8/bin/nsys','export','--type=sqlite','--force-overwrite=true',
                    '-o',str(ROOT/'native_profile.sqlite'),str(ROOT/'native_profile.nsys-rep')],check=True)

"""真实生产SLAM的有界Nsight窗口；不终止应用、不修改配置。"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/prmi/zwj/anchor-selector-r3-performance-20261022')
SOURCE=Path('/home/prmi/zwj/private-compensation-online-20261022/source')
from_start='--from-start' in sys.argv
label='online_profile_start2' if from_start else 'online_profile'
run=ROOT/('online_nsight_start2' if from_start else 'online_nsight');run.mkdir(exist_ok=False)
env=dict(os.environ,OMP_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',MPLBACKEND='Agg',CARTGS_AUDIT_SEED='2026',
         CARTGS_MAPPING_TRACE=str(run/'mapping_trace.csv'),LD_LIBRARY_PATH=str(SOURCE/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),
         PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'])
for key in ['CARTGS_COMPENSATION_SPLIT4','CARTGS_BUDGET_METHOD','CARTGS_BUDGET_AUDIT','CARTGS_MLP_GRAD_AUDIT']:
    env.pop(key,None)
profile_options=[] if from_start else ['--delay=22','--duration=10']
cmd=['/usr/local/cuda-11.8/bin/nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none']+profile_options+[
     '--kill=none','--wait=all','-o',str(ROOT/label),
     str(SOURCE/'bin/tum_rgbd'),str(SOURCE/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt'),
     str(SOURCE/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml'),str(SOURCE/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml'),
     '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',str(SOURCE/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt'),
     str(run),'no_viewer','--selector-config',str(SOURCE/'cfg/selector/selector_config.yaml')]
(ROOT/(label+'_command.json')).write_text(json.dumps(cmd,indent=2))
with (run/'training.log').open('w') as log:
    subprocess.run(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run(['/usr/local/cuda-11.8/bin/nsys','export','--type=sqlite','-o',str(ROOT/(label+'.sqlite')),
                str(ROOT/(label+'.nsys-rep'))],check=True)
print('PASS real online Nsight captured; normal application exit',flush=True)

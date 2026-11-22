"""隔离构建完成后依次执行功能 gate、固定协议训练和质量评价。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

r=Path(__file__).resolve().parents[3];repo=r/'source';v=r/'validation'
env=dict(os.environ,OMP_NUM_THREADS='2',MPLBACKEND='Agg',QT_QPA_PLATFORM='offscreen',
         PATH='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ['PATH'],
         LD_LIBRARY_PATH=str(repo/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
def state(stage,**extra):
    (v/'pipeline_status.json').write_text(json.dumps(dict(stage=stage,**extra),indent=2))
state('WAITING_FOR_BUILD')
while not (v/'build.exit').exists():time.sleep(5)
assert (v/'build.exit').read_text()=='0','build failed'
manifest=json.loads((v/'source_manifest.json').read_text())
assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==sha for p,sha in manifest.items())
gates={}
def run(name,command,cwd=v):
    state('CORRECTNESS',test=name)
    with (v/(name+'.log')).open('w') as output:
        result=subprocess.run(list(map(str,command)),cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT)
    gates[name]=dict(passed=result.returncode==0,exit_code=result.returncode)
    (v/'correctness.json').write_text(json.dumps(gates,indent=2))
    if result.returncode:
        state('FAILED',test=name);raise SystemExit(result.returncode)
for device in ('cpu','cuda'):
    directory=v/device;directory.mkdir(exist_ok=True)
    run('explicit_'+device,[repo/'bin/explicit_anchor_test',device],directory)
    run('explicit_python_'+device,[sys.executable,repo/'tests/anchor_selector/explicit_python_parity.py',directory,device])
    run('legacy27_'+device,[repo/'bin/anchor_attribute_test',device])
run('explicit_mapping',[repo/'bin/explicit_anchor_mapping_test','--parity'])
run('legacy27_mapping',[repo/'bin/anchor_attribute_mapping_test','--parity'])
run('legacy27_change_distill',[repo/'bin/anchor_change_distill_test','attributes'])
state('TRAINING_AND_EVALUATION')
with (v/'audit.log').open('w') as output:
    result=subprocess.run([sys.executable,str(repo/'tests/anchor_selector/run_explicit_audit.py'),str(repo),
        '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',str(repo/'tests/anchor_selector/explicit_validation/protocol.json')],
        cwd=repo,env=env,stdout=output,stderr=subprocess.STDOUT)
if result.returncode:
    state('FAILED',stage_detail='TRAINING_OR_EVALUATION',exit_code=result.returncode);raise SystemExit(result.returncode)
with (v/'summary.log').open('w') as output:
    result=subprocess.run([sys.executable,str(repo/'tests/anchor_selector/summarize_explicit_audit.py'),str(r)],
                          env=env,stdout=output,stderr=subprocess.STDOUT)
assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==sha for p,sha in manifest.items())
state('COMPLETED' if result.returncode==0 else 'FAILED',exit_code=result.returncode)

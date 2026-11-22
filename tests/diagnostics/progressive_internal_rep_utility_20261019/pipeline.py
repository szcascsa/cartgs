from pathlib import Path
import json,subprocess,sys,fcntl,traceback
ROOT=Path(__file__).resolve().parent
def save(v):(ROOT/'status.json').write_text(json.dumps(v,indent=2)+'\n')
lock=open(ROOT/'controller.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def run(stage,out=ROOT):
    save(dict(scene=out.name,stage=stage,state='running'))
    with open(out/(stage+'.log'),'w')as log:subprocess.run([sys.executable,str(out/(stage+'.py'))],cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('PASS',out.name,stage,flush=True)
try:
    run('prepare')
    for s in ['fr1_room','fr1_xyz','fr2_xyz']:run('baseline',ROOT/s)
    (ROOT/'baseline_all_passed.json').write_text(json.dumps(dict(all_passed=True,scenes=['fr1_room','fr1_xyz','fr2_xyz']))+'\n')
    for s in ['fr1_room','fr1_xyz','fr2_xyz']:
        for stage in ['construct','quality','diagnostics']:run(stage,ROOT/s)
    save(dict(state='complete',all_passed=True))
except BaseException:save(dict(state='failed',failure=traceback.format_exc()));raise

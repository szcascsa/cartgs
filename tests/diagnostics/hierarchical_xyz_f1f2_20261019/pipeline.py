from pathlib import Path
import subprocess,sys,json,fcntl,traceback
ROOT=Path(__file__).resolve().parent
def save(v):(ROOT/'status.json').write_text(json.dumps(v,indent=2)+'\n')
lock=open(ROOT/'controller.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
    with open(ROOT/'prepare.log','w')as log:subprocess.run([sys.executable,str(ROOT/'prepare.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    for scene in ['fr1_xyz','fr2_xyz']:
        out=ROOT/scene
        for stage in ['baseline','new_rankings','replay_checks','diagnostics','quality','analyze']:
            save(dict(scene=scene,stage=stage,state='running'))
            with open(out/(stage+'.log'),'w')as log:subprocess.run([sys.executable,str(out/(stage+'.py'))],cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('PASS',scene,stage,flush=True)
    save(dict(state='complete',all_passed=True))
except BaseException:save(dict(state='failed',failure=traceback.format_exc()));raise

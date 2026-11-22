from pathlib import Path
import subprocess,sys,time,json,fcntl,traceback
ROOT=Path(__file__).resolve().parent
def save(v):(ROOT/'status.json').write_text(json.dumps(v,indent=2)+'\n')
lock=open(ROOT/'controller.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);times=[]
def run(stage,out=ROOT):
    save(dict(state='running',scene=out.name,stage=stage));start=time.monotonic()
    with open(out/(stage+'.log'),'w')as log:subprocess.run([sys.executable,str(out/(stage+'.py'))],cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    times.append(dict(scene=out.name,stage=stage,wall_seconds=time.monotonic()-start));(ROOT/'runtime.json').write_text(json.dumps(times,indent=2)+'\n');print('PASS',out.name,stage,flush=True)
try:
    run('prepare');run('build')
    for s in ['fr1_room','fr1_xyz','fr2_xyz']:run('baseline',ROOT/s)
    (ROOT/'baseline_all_passed.json').write_text(json.dumps(dict(all_passed=True))+'\n')
    for s in ['fr1_room','fr1_xyz','fr2_xyz']:
        run('coverage',ROOT/s);run('fallback',ROOT/s)
    run('independent_check')
    for s in ['fr1_room','fr1_xyz','fr2_xyz']:run('quality',ROOT/s);run('diagnostics',ROOT/s)
    save(dict(state='complete',all_passed=True))
except BaseException:save(dict(state='failed',failure=traceback.format_exc()));raise

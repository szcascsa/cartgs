"""只收割本轮已暂停调度器的成功子任务，保留现场与所有重试记录。"""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time
import run as r
from prepare import R,read,write

s=read(R/'LOD_PIPELINE.json');old=read(R/'LOD_CONTROLLER.json')
assert old==r.p.identity(old['pid'])
assert Path(f"/proc/{old['pid']}/stat").read_text().split(') ',1)[1].split()[0]=='T'
active=[]
for j in s['jobs']:
    if j['state']=='running':active.append((j,j,False))
    for e in j['evaluations']:
        if e['state']=='running':active.append((j,e,True))
for job,item,is_eval in active:
    ident=item['process'];assert ident==r.p.identity(ident['pid'])
    fields=Path(f"/proc/{ident['pid']}/stat").read_text().split(') ',1)[1].split()
    assert fields[0]=='Z' and int(fields[-1])==0,(job['id'],fields[0],fields[-1])
write(R/'reports/cluster_transition/FINISHED_CHILDREN.json',active)
os.kill(old['pid'],signal.SIGKILL)
for _ in range(100):
    p=Path(f"/proc/{old['pid']}/stat")
    if not p.exists() or p.read_text().split(') ',1)[1].split()[0]=='Z':break
    time.sleep(.1)
else:raise RuntimeError('旧调度器未退出')
for job,item,is_eval in active:
    out=Path(item['attempt']);start=read(out/'launch.json')['started_unix']
    exit_info=dict(returncode=0,seconds=time.time()-start,recovered_after_verified_zombie=True)
    write(out/'exit.json',exit_info)
    item.setdefault('attempt_history',[]).append(dict(output=str(out),process=item['process'],oom=None,health=None,**exit_info))
    if is_eval:
        if not (out/'train.log').exists():(out/'train.log').symlink_to('evaluate.log')
        r.finish_evaluation(job,item,s,out)
    else:
        r.verify_training(out,job['target_steps'],job['reference_model'])
        with (out/'initialization_check.log').open('w') as log:
            p=subprocess.run([sys.executable,str(R/'verify_initialization.py'),str(out/'manifest.json')],env=r.environment(),stdout=log,stderr=subprocess.STDOUT)
        assert p.returncode==0
        job.update(state='trained',trained_model=str(out/'model'),evaluations=r.cases())
    r.save(s)
plan=read(R/'cluster/ASSIGNMENTS.json')['assignments']
for j in s['jobs']:j['assigned_worker']=plan[j['id']]
s.update(phase='cluster_preparing');r.save(s)
write(R/'reports/cluster_transition/RECOVERED.json',dict(passed=True,previous_controller=old,successfully_adopted=len(active),time=time.time()))
print('RECOVERED',len(active),flush=True)

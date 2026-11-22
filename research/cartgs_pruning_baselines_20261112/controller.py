"""独立串行评价队列；先验收，再完成32组；不训练、不写生产。"""
from pathlib import Path
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time

from common import ROOT, load, write, sha

SMOKE=['TUM_RGBD/freiburg1_xyz/mono','TUM_RGBD/freiburg1_xyz/rgbd',
       'VECtor/corner-slow/mono','VECtor/corner-slow/stereo',
       'Replica/office0/mono','Replica/office0/rgbd']


def run_child(group,smoke=False,attempt=1):
    log=ROOT/'logs'/group.replace('/','__')/f"{'smoke' if smoke else 'evaluation'}_{attempt:02}.log"
    log.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u',str(ROOT/'worker.py'),'--group',group]
    if smoke:command+=['--limit','2']
    started=time.time()
    with log.open('w') as output:
        child=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,
             env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                      MPLBACKEND='Agg'),start_new_session=True)
        fields=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()
        identity=dict(pid=child.pid,start_ticks=fields[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        write(ROOT/'ACTIVE.json',dict(group=group,smoke=smoke,process=identity,command=command,
                                    log=str(log),started_unix=started))
        code=child.wait()
    tail=log.read_text()[-10000:]
    oom=code!=0 and any(text in tail for text in ['CUDA out of memory','torch.OutOfMemoryError','CUDA error: out of memory'])
    return dict(group=group,smoke=smoke,code=code,oom=oom,seconds=time.time()-started,
                log=str(log),process=identity)


def main():
    (ROOT/'logs').mkdir(exist_ok=True)
    lock=(ROOT/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=load(ROOT/'EXPERIMENTS.json')
    fields=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()
    write(ROOT/'CONTROLLER.json',dict(pid=os.getpid(),start_ticks=fields[19],
          boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    for source,item in load(ROOT/'SOURCE_MANIFEST.json').items():
        assert sha(source)==sha(item['copy'])==item['sha256'],source
    if not (ROOT/'SMOKE_PASSED.json').exists():
        state['phase']='validating';write(ROOT/'EXPERIMENTS.json',state)
        tests=subprocess.run([sys.executable,str(ROOT/'test_selection.py')],cwd=ROOT,capture_output=True,text=True)
        (ROOT/'logs/selection_tests.log').write_text(tests.stdout+tests.stderr)
        assert tests.returncode==0
        records=[]
        for group in SMOKE:
            if (ROOT/'smoke'/group/'COMPLETE.json').exists():continue
            result=run_child(group,True);records.append(result)
            if result['code']!=0:
                state.update(phase='held',error=result);write(ROOT/'EXPERIMENTS.json',state);return
        write(ROOT/'SMOKE_PASSED.json',dict(passed=True,groups=SMOKE,records=records))
    for group in state['groups']:
        if (ROOT/'runs'/group['id']/'COMPLETE.json').exists():
            group['state']='complete';continue
        if shutil.disk_usage(ROOT).free<5*1024**3:
            state.update(phase='held',error='DISK_BELOW_5GIB');write(ROOT/'EXPERIMENTS.json',state);return
        group['state']='running';state.update(phase='evaluating',error=None)
        write(ROOT/'EXPERIMENTS.json',state)
        while True:
            record=run_child(group['id'],attempt=len(group['attempts'])+1)
            group['attempts'].append(record)
            if record['code']==0:
                assert (ROOT/'runs'/group['id']/'COMPLETE.json').exists()
                group['state']='complete';write(ROOT/'EXPERIMENTS.json',state)
                print('GROUP_COMPLETE',group['id'],flush=True);break
            if record['oom'] and sum(x['oom'] for x in group['attempts'])<3:
                write(ROOT/'EXPERIMENTS.json',state);continue
            group['state']='failed';state.update(phase='held',error=record)
            write(ROOT/'EXPERIMENTS.json',state);return
    after={name:sha(name) for name in load(ROOT/'PROTECTED_BEFORE.json')}
    write(ROOT/'PROTECTED_AFTER.json',after)
    assert after==load(ROOT/'PROTECTED_BEFORE.json')
    state['phase']='evaluations_complete';write(ROOT/'EXPERIMENTS.json',state)
    result=subprocess.run([sys.executable,str(ROOT/'report.py')],cwd=ROOT)
    assert result.returncode==0
    state['phase']='finished';write(ROOT/'EXPERIMENTS.json',state)
    print('全部32组、704项评价及核验完成。',flush=True)


if __name__=='__main__':main()

"""顺序完成32组局部ID并集统计；只启动本目录的只读推理进程。"""
from pathlib import Path
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time

from collect_union import ROOT,SOURCE,SOURCE_SHA,records,sha,write


def main():
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw=records();groups=list(dict.fromkeys(r['group'] for r in raw))
    assert len(groups)==32 and len(raw)==352
    identity=dict(pid=os.getpid(),start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    write(ROOT/'CONTROLLER.json',identity)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    finished=[];started=time.time()
    for group in groups:
        if shutil.disk_usage(ROOT).free<5*1024**3:raise RuntimeError('DISK_BELOW_5GIB')
        folder=ROOT/'runs'/group;folder.mkdir(parents=True,exist_ok=True)
        completion=folder/'COMPLETE.json'
        write(ROOT/'STATUS.json',dict(state='running',group=group,finished=len(finished),groups=32,
            started_unix=started,updated_unix=time.time(),controller=identity))
        if not completion.exists():
            command=[sys.executable,'-B',str(ROOT/'collect_union.py'),'--group',group]
            with (folder/'worker.log').open('a') as stream:
                child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
                write(folder/'PROCESS.json',dict(pid=child.pid,command=command,started_unix=time.time()))
                code=child.wait()
            if code:
                write(ROOT/'STATUS.json',dict(state='failed',group=group,finished=len(finished),groups=32,
                    returncode=code,log=str(folder/'worker.log'),updated_unix=time.time(),controller=identity))
                raise RuntimeError(f'UNION_WORKER_FAILED {group} {code}')
        result=json.loads(completion.read_text())
        assert result['complete'] and result['source_unchanged'] and len(result['ratios'])==11
        finished.append(result)
        print('GROUP_COMPLETE',len(finished),32,group,flush=True)
    all_rows=[r for result in finished for r in result['ratios']]
    with (ROOT/'LOCAL_UNION_COUNTS.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    assert sha(SOURCE)==SOURCE_SHA
    write(ROOT/'FINISHED.json',dict(complete=True,groups=32,ratios=352,
        views=sum(g['frames'] for g in finished),selection_checks=sum(g['frames']*11 for g in finished),
        all_view_hashes_match=True,source_unchanged=True,training_steps=0,images_saved=0,
        seconds=time.time()-started,completed_unix=time.time()))
    write(ROOT/'STATUS.json',dict(state='complete',finished=32,groups=32,updated_unix=time.time(),controller=identity))
    print('ALL_UNION_COUNTS_COMPLETE',len(all_rows),flush=True)


if __name__=='__main__':main()

"""第二个独立场景评价进程；只处理逆序待评价组，不写主队列状态。"""
from pathlib import Path
import fcntl
import os
import subprocess
import sys
import time
from common import ROOT,load,write


def main():
    lock=(ROOT/'parallel.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    groups=list(reversed(load(ROOT/'EXPERIMENTS.json')['groups']))
    for group in groups:
        folder=ROOT/'runs'/group['id']
        if (folder/'COMPLETE.json').exists():continue
        if load(ROOT/'EXPERIMENTS.json')['phase']=='held':return
        for attempt in range(3):
            log=ROOT/'logs'/group['id'].replace('/','__')/f'parallel_{attempt+1:02}.log'
            log.parent.mkdir(parents=True,exist_ok=True)
            started=time.time()
            with log.open('w') as stream:
                p=subprocess.Popen([sys.executable,'-u',str(ROOT/'worker.py'),'--group',group['id']],
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg'))
                fields=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()
                write(ROOT/'PARALLEL_ACTIVE.json',dict(group=group['id'],log=str(log),started_unix=started,
                     process=dict(pid=p.pid,start_ticks=fields[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())))
                code=p.wait()
            record=dict(group=group['id'],attempt=attempt+1,code=code,seconds=time.time()-started,log=str(log))
            write(folder/f'PARALLEL_ATTEMPT_{attempt+1:02}.json',record)
            if code==0:break
            if not any(x in log.read_text()[-10000:] for x in ['CUDA out of memory','torch.OutOfMemoryError','CUDA error: out of memory']):
                write(ROOT/'PARALLEL_ERROR.json',record);return
        else:
            write(ROOT/'PARALLEL_ERROR.json',record);return
    write(ROOT/'PARALLEL_FINISHED.json',dict(complete=True))


if __name__=='__main__':main()

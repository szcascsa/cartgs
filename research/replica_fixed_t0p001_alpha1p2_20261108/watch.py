"""每小时核对本实验的真实进程和文件进展，不控制其他队列。"""
from pathlib import Path
import fcntl
import os
import shutil
import subprocess
import time
from prepare_experiment import R,read,write


def live(info):
    try:
        stat=(Path('/proc')/str(info['pid'])/'stat').read_text().rsplit(')',1)[1].split()
        return stat[0]!='Z' and stat[19]==info['start_ticks'] and info['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except (KeyError,FileNotFoundError,ProcessLookupError):return False


def main():
    lock=(R/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    last_hour=0;progress={}
    while True:
        jobs=read(R/'QUEUE.json');evaluations=read(R/'TRAJECTORY_EVALUATION.json')
        active=[];alerts=[];current={}
        for kind,items in [('training',jobs),('evaluation',evaluations['jobs'])]:
            for job in items:
                if job['state']!='running':continue
                folder=Path(job.get('attempt',job.get('output','')))
                path=folder/('train.log' if kind=='training' else 'evaluate.log')
                size=path.stat().st_size if path.exists() else 0
                info=job['process'];pid=info['pid'];old=progress.get(pid)
                since=old[1] if old and old[0]==size else time.monotonic();current[pid]=(size,since)
                idle=time.monotonic()-since
                record=dict(id=job['id'],kind=kind,process=info,live_identity_match=live(info),
                            log_bytes=size,idle_seconds=idle,log=str(path))
                if not record['live_identity_match'] or idle>900:
                    wchan=Path('/proc')/str(pid)/'wchan'
                    record['wchan']=wchan.read_text() if wchan.exists() else 'exited';alerts.append(record)
                active.append(record)
        progress=current
        free=shutil.disk_usage(R).free/2**30
        if free<5 and not (R/'HOLD').exists():(R/'HOLD').write_text('磁盘不足5GiB，暂停发射，不清理模型\n')
        snapshot=dict(checked_unix=time.time(),active=active,free_gib=free,
                      coordinator_alive=live(read(R/'COORDINATOR.json')),alerts=alerts,
                      training_counts={state:sum(j['state']==state for j in jobs) for state in set(j['state'] for j in jobs)},
                      evaluation_counts=evaluations.get('counts',{}))
        if time.time()-last_hour>=3600 or alerts or (R/'FINISHED.json').exists():
            snapshot['gpu']=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
            snapshot['saved_images']=sum(p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp']
                                         for folder in ['jobs','evaluation'] for p in (R/folder).rglob('*') if p.is_file())
            write(R/'reports/HOURLY_CHECK.json',snapshot);last_hour=time.time()
        if alerts:write(R/'reports/RUNTIME_ALERT.json',snapshot)
        if (R/'FINISHED.json').exists():return
        if not snapshot['coordinator_alive']:
            write(R/'reports/RUNTIME_ALERT.json',dict(snapshot,error='本实验控制器已退出，保留现场，未盲目重跑'))
            return
        time.sleep(60)


if __name__=='__main__':main()

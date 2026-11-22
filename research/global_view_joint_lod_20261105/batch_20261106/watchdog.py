"""只监测本批子进程的实际日志/CSV进展；900秒停滞时保存现场并停止该子进程。"""
from pathlib import Path
import fcntl
import json
import os
import signal
import subprocess
import time
import pipeline as p

R=Path(__file__).resolve().parent


def main():
    lock=(R/'watchdog.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p.write(R/'WATCHDOG.json',p.identity(os.getpid()))
    previous=None;signature=None;last_progress=time.monotonic()
    while True:
        state=p.read(R/'LOD_PIPELINE.json')
        if state['phase'] in ['finished','blocked']:
            p.write(R/'reports/WATCHDOG_STATUS.json',dict(state='stopped',pipeline_phase=state['phase']))
            return
        candidates=[]
        if state.get('active_validation'):candidates.append(state['active_validation'])
        candidates.extend(state['jobs'])
        active=[]
        for job in candidates:
            if job.get('state')=='running' and p.alive(job.get('process',{})):active.append(job)
            for item in job.get('evaluations',[]):
                if item['state']=='running' and p.alive(item.get('process',{})):active.append(item)
        if active:
            assert len(active)==1,'同一批次出现多个活动GPU工作项'
            item=active[0];identity=item['process'];folder=Path(item['attempt'])
            folder.resolve().relative_to(R.resolve())
            paths=[folder/'train.log',folder/'evaluate.log',folder/'model/real_lod_steps.csv']
            sizes=tuple(path.stat().st_size if path.exists() else 0 for path in paths)
            key=(identity['pid'],identity['start_ticks'],identity['boot_id'])
            if key!=previous or sizes!=signature:
                previous=key;signature=sizes;last_progress=time.monotonic()
            idle=time.monotonic()-last_progress
            p.write(R/'reports/WATCHDOG_STATUS.json',dict(state='monitoring',process=identity,
                attempt=str(folder),log_csv_bytes=sizes,no_progress_seconds=idle,checked_unix=time.time()))
            if idle>900:
                pid=identity['pid'];proc=Path('/proc')/str(pid)
                evidence=dict(process=identity,attempt=str(folder),no_progress_seconds=idle,
                    ps=subprocess.run(['ps','-p',str(pid),'-o','pid,stat,etime,time,wchan:32,comm'],capture_output=True,text=True).stdout,
                    gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader'],capture_output=True,text=True).stdout,
                    status=(proc/'status').read_text() if proc.exists() else '')
                p.write(R/'reports/STALL_FAILURE.json',evidence)
                # 只停止身份三元组完全相同的本批直接工作进程，绝不触碰其他用户或Viewer。
                if p.alive(identity):os.kill(pid,signal.SIGTERM)
                return
        else:
            previous=None;signature=None;last_progress=time.monotonic()
        time.sleep(60)


if __name__=='__main__':main()

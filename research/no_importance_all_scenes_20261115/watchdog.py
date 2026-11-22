"""按用户指定每30分钟巡查；不修改训练、结果或其他进程。"""
from pathlib import Path
import json
import os
import subprocess
import time
from prepare import R,read,write
from pipeline import alive

INTERVAL=1800


def check():
    state=read(R/'LOD_PIPELINE.json');controller=read(R/'LOD_CONTROLLER.json');alerts=[];active=[]
    if state['phase'] not in ['finished','local_finished'] and not alive(controller):alerts.append('controller_exited_before_queue_finished')
    for j in state['jobs']:
        if j.get('assigned_worker','original')!=os.environ.get('CARTGS_CLUSTER_WORKER','original'):continue
        if j['state'] in ['failed','skipped_zero_selection','skipped_oom','partial_evaluation']:
            alerts.append(dict(id=j['id'],state=j['state'],error=j.get('error')))
        for scope,v in [('train',j)]+[(e['scope']+'_'+str(e['ratio']),e) for e in j['evaluations']]:
            if v['state']!='running':continue
            proc=v['process'];out=Path(v['attempt']);log=out/('train.log' if scope=='train' else 'evaluate.log')
            trace=out/'model/real_lod_steps.csv';progress=trace if trace.exists() else log
            identity_ok=alive(proc);age=time.time()-progress.stat().st_mtime
            row=dict(id=j['id'],scope=scope,identity_matches=identity_ok,progress_age_seconds=age,
                log_bytes=log.stat().st_size,process=proc,step=v.get('step'),target_steps=j['target_steps'])
            if not identity_ok or age>600:
                # Finished children can briefly precede the controller's saved state.
                row['log_tail']=log.read_text(errors='replace')[-4000:]
                for name in ['stat','wchan','status','io']:
                    p=Path('/proc')/str(proc['pid'])/name
                    if p.exists():row[name]=p.read_text()[-4000:]
                alerts.append(row)
            active.append(row)
    snapshot=dict(checked_unix=time.time(),interval_seconds=INTERVAL,phase=state['phase'],counts=state.get('counts'),
        active=active,evaluations_complete=sum(e['state']=='complete' for j in state['jobs'] for e in j['evaluations']),alerts=alerts)
    snapshot['gpu']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
    write(R/'reports/WATCHDOG_STATUS.json',snapshot)
    with (R/'logs/watchdog.jsonl').open('a') as f:f.write(json.dumps(snapshot,ensure_ascii=False)+'\n')
    if alerts:write(R/'reports/WATCHDOG_ALERT.json',snapshot)
    return snapshot


def main():
    import fcntl
    lock=(R/'watchdog.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(R/'reports/MONITORING_POLICY.json',dict(interval_seconds=INTERVAL,user_requested=True,
        normal_status_notifications=False,notify_on=['stage_complete','failure','required_user_action'],pid=os.getpid()))
    while True:
        # Short waits keep shutdown responsive; task inspection occurs only each 30 minutes.
        deadline=time.monotonic()+INTERVAL
        while time.monotonic()<deadline:time.sleep(min(30,deadline-time.monotonic()))
        snapshot=check()
        if snapshot['phase'] in ['finished','local_finished']:return


if __name__=='__main__':main()

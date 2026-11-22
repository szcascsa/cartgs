"""原服务器为唯一汇总节点；每半小时生成三节点状态，最后安全合并并出报告。"""
from pathlib import Path
import collections
import copy
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
import run
from prepare import R,read,write


def alive(ident):
    p=Path('/proc')/str(ident['pid'])/'stat'
    if not p.exists():return False
    fields=p.read_text().split(') ',1)[1].split()
    return fields[0]!='Z' and fields[19]==ident['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==ident['boot_id']


def check():
    state=read(R/'LOD_PIPELINE.json');merged=copy.deepcopy(state);nodes={};alerts=[]
    for node in ['cloud_a','cloud_b']:
        folder=R/'cluster'/node;remote=folder/'REMOTE_STATUS.json'
        if remote.exists():
            snapshot=read(remote);nodes[node]=dict(checked_unix=snapshot['checked_unix'],active=snapshot['active'],disk_free=snapshot['disk_free'])
            owned=[j for j in snapshot['pipeline']['jobs'] if j.get('assigned_worker')==node]
            collector=folder/'COLLECTOR_STATUS.json'
            collected=collector.exists() and read(collector).get('complete') and bool(owned) and all(
                j['state'] in run.TERMINAL and (folder/'imports'/(j['id']+'.json')).exists() for j in owned)
            nodes[node]['collection_complete']=bool(collected)
            session=snapshot.get('session') or {}
            if session.get('phase','').startswith('failed_'):alerts.append(dict(node=node,session=session,log_tail=snapshot.get('session_log_tail')))
            if session.get('phase')=='training' and not snapshot.get('controller_alive'):alerts.append(node+' 正式训练控制器身份不匹配或已退出')
            if not collected and time.time()-snapshot['checked_unix']>3900:alerts.append(node+' 巡查数据超过65分钟未更新')
            for row in snapshot['active']:
                if not row['identity_matches'] or row['progress_age']>600:alerts.append(dict(node=node,details=row))
            jobs={j['id']:j for j in snapshot['pipeline']['jobs'] if j.get('assigned_worker')==node}
            for j in merged['jobs']:
                if j.get('assigned_worker')==node and j['id'] in jobs:j.update(jobs[j['id']])
        else:nodes[node]=dict(state='awaiting_deployment_or_first_check')
    for job in merged['jobs']:
        if job['state'] in run.TERMINAL-{'complete'}:alerts.append(dict(id=job['id'],state=job['state'],error=job.get('error')))
    counts=collections.Counter(j['state'] for j in merged['jobs'])
    for node in ['original','cloud_a','cloud_b']:
        own=[j for j in merged['jobs'] if j.get('assigned_worker')==node]
        nodes.setdefault(node,{})['counts']=dict(collections.Counter(j['state'] for j in own))
        nodes[node]['evaluations_complete']=sum(e['state']=='complete' for j in own for e in j['evaluations'])
    snapshot=dict(checked_unix=time.time(),interval_seconds=1800,counts=dict(counts),nodes=nodes,alerts=alerts)
    write(R/'reports/CLUSTER_STATUS.json',snapshot);write(R/'reports/CLUSTER_PIPELINE.json',merged)
    with (R/'logs/cluster_watchdog.jsonl').open('a') as f:f.write(json.dumps(snapshot,ensure_ascii=False)+'\n')
    if alerts:write(R/'reports/CLUSTER_ALERT.json',snapshot)
    rows=[]
    for j in merged['jobs']:
        for e in j['evaluations']:
            if e['state']!='complete':continue
            a=next(v['result'] for v in j['reference_evaluations'] if (v['scope'],v['ratio'])==(e['scope'],e['ratio']));d=e['result']
            row=dict(id=j['id'],worker=j['assigned_worker'],scope=e['scope'],ratio=e['ratio'],frames=d['evaluated_frames'])
            for k in ['psnr','ssim','lpips','ate_rmse','actual_K_mean']:
                if a.get(k) is not None and d.get(k) is not None:row.update({k+'_A':a[k],k+'_D':d[k],'delta_'+k:d[k]-a[k]})
            rows.append(row)
    if rows:
        file=R/'reports/CLUSTER_A_VS_D_METRICS.csv';keys=list(dict.fromkeys(k for row in rows for k in row))
        with file.with_suffix('.tmp').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
        file.with_suffix('.tmp').replace(file)
    imports_ready=all((R/'cluster'/j['assigned_worker']/'imports'/(j['id']+'.json')).exists() for j in merged['jobs'] if j['assigned_worker']!='original')
    local_done=state['phase']=='local_finished' and not alive(read(R/'LOD_CONTROLLER.json'))
    if local_done and imports_ready and all(j['state'] in run.TERMINAL for j in merged['jobs']):
        with (R/'controller.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            merged['phase']='finished';run.save(merged)
            subprocess.run([sys.executable,str(R/'finish.py')],cwd=R,env=run.environment(),check=True)
        write(R/'reports/CLUSTER_COMPLETE.json',dict(passed=True,jobs=32,images_imported=0,finished_unix=time.time()))
        return True
    return False


if __name__=='__main__':
    lock=(R/'cluster_monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(R/'reports/CLUSTER_MONITOR_POLICY.json',dict(pid=os.getpid(),interval_seconds=1800,summary_node='original',image_import=False))
    while True:
        try:
            if check():break
        except Exception:
            write(R/'reports/CLUSTER_MONITOR_ERROR.json',dict(time=time.time(),traceback=traceback.format_exc()))
        time.sleep(1800)

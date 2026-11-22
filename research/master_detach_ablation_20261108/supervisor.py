"""本批专属监督器：每小时真实进展核查，健康重跑跨控制器重启保留。"""
from pathlib import Path
import collections
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import run_batch as b

R=b.R;p=b.p


def alive(info):
    try:
        fields=(Path('/proc')/str(info['pid'])/'stat').read_text().rsplit(')',1)[1].split()
        return fields[0]!='Z' and p.identity(info['pid'])=={k:info[k] for k in ['pid','start_ticks','boot_id']}
    except (FileNotFoundError,KeyError,ProcessLookupError):return False


def running(state):
    jobs=[*state['jobs']]+([state['active_validation']] if state.get('active_validation') else [])
    items=[]
    for job in jobs:
        if job['state']=='running':items.append((job,job,False))
        for item in job.get('evaluations',[]):
            if item['state']=='running':items.append((job,item,True))
    return items


def recover(state):
    """只恢复有明确健康/OOM/成功落盘证据的本批工作项，绝不覆盖活进程。"""
    if state.get('parallel_workers'):
        from parallel_queue import collect
        for lane,info in list(state['parallel_workers'].items()):
            if not alive(info['process']) and collect(state,lane):return False
        state.update(phase='recovered',error='');p.save(state);return True
    for job,item,evaluation in running(state):
        if alive(item.get('process',{})):return False
        out=Path(item['attempt']);out.resolve().relative_to(R.resolve())
        health=out/'model/lod_health_failure.json';exit_file=out/'exit.json'
        if health.exists():
            history=item.setdefault('attempt_history',[])
            if not any(v['output']==str(out) for v in history):
                history.append(dict(output=str(out),process=item['process'],health=p.read(health),oom=None,returncode=-1,seconds=0,recovered=True))
                item['health_failures']=item.get('health_failures',0)+1
            item.update(state='retry_pending' if b.health_decision(item,state['protocol']['max_health_retries']) else 'skipped_health',error=p.read(health)['error'])
            b.discard_initial_copies(out/'model')
        elif exit_file.exists() and p.read(exit_file)['returncode']==0:
            if evaluation:
                summary=p.read(out/'metric_summary.json')
                assert summary['evaluated_frames']==job['full_evaluation']['result']['evaluated_frames']
                item.update(state='complete',result=dict(summary,output=str(out),scope=item['scope'],saved_images=0))
            else:
                steps=6 if state.get('active_validation') is job else job['target_steps']
                b.verify_training(out,steps)
                if not (out/'model/initialization_verified.json').exists():b.initialization_check(out)
                b.discard_initial_copies(out/'model');job.update(state='trained',trained_model=str(out/'model'))
                if 'evaluations' not in job or not job['evaluations']:job['evaluations']=[] if steps==6 else b.cases()
        elif exit_file.exists():
            info=p.read(exit_file);oom=p.classify_oom(dict(output=str(out),process=item['process']),info)
            if not oom:return False
            item.setdefault('attempt_history',[]).append(dict(output=str(out),process=item['process'],oom=oom,health=None,**info))
            item['oom_failures']=item.get('oom_failures',0)+1
            item.update(state='retry_pending' if b.oom_decision(item) else 'skipped_oom')
        else:return False
    state.update(phase='recovered',error='');p.save(state);return True


def snapshot(state,controller):
    details=[]
    for job,item,evaluation in running(state):
        out=Path(item['attempt']);out.resolve().relative_to(R.resolve())
        paths=[out/('evaluate.log' if evaluation else 'train.log'),out/'model/real_lod_steps.csv']
        records={str(v.relative_to(R)):dict(bytes=v.stat().st_size,age_seconds=round(time.time()-v.stat().st_mtime,2)) for v in paths if v.exists()}
        progress=out/'model/real_lod_steps.csv';step=None
        if progress.exists():
            with progress.open('rb') as f:f.seek(max(0,progress.stat().st_size-4096));lines=f.read().decode(errors='replace').splitlines()
            step=next((int(s.split(',')[0]) for s in reversed(lines) if s.split(',')[0].isdigit()),None)
        details.append(dict(job=job['id'],kind='evaluation' if evaluation else 'training',process=item.get('process'),
            live_identity_match=alive(item.get('process',{})),attempt=str(out),files=records,step=step))
    images=[str(v) for folder in ['runs','evaluation','validation'] for v in (R/folder).rglob('*') if v.is_file() and v.suffix.lower() in ['.png','.jpg','.jpeg','.bmp']]
    result=dict(checked_unix=time.time(),phase=state['phase'],counts=dict(collections.Counter(j['state'] for j in state['jobs'])),
        controller=controller,controller_alive=alive(controller),active=details,free_GiB=shutil.disk_usage(R).free/2**30,
        rendered_image_count=len(images),images=images[:10],hourly_interval_seconds=3600,
        gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip())
    p.write(R/'reports/HOURLY_CHECK.json',result)
    with (R/'reports/HOURLY_CHECKS.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
    return result


def launch():
    with (R/'logs/worker.log').open('a') as log:
        script='parallel_queue.py' if (R/'PARALLEL_CONFIG.json').exists() and p.read(R/'PARALLEL_CONFIG.json')['enabled'] else 'run_batch.py'
        child=subprocess.Popen([sys.executable,'-u',str(R/script)],cwd=R,env=b.environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info=p.identity(child.pid);p.write(R/'LOD_CONTROLLER.json',info);return child,info


def main():
    assert not (R/'HOLD').exists() and p.read(R/'START_AUTHORIZED.json')['authorized']
    lock=(R/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p.write(R/'SUPERVISOR.json',p.identity(os.getpid()))
    bootstrap=p.read(R/'reports/BOOTSTRAP.json');assert bootstrap['phase']=='build_complete'
    for script,receipt in [('test_queue.py','reports/QUEUE_TESTS_PASSED.json'),('validate_startup.py','reports/STARTUP_VALIDATION.json')]:
        if (R/receipt).exists() and p.read(R/receipt).get('passed'):continue
        with (R/'logs'/('startup_'+script+'.log')).open('w') as log:
            code=subprocess.run([sys.executable,'-u',str(R/script)],cwd=R,env=b.environment(),stdout=log,stderr=subprocess.STDOUT).returncode
        assert code==0,script+'失败，保留日志，禁止进入正式队列'
        if script=='test_queue.py':p.write(R/receipt,dict(passed=True,tests=8))
    info=p.read(R/'LOD_CONTROLLER.json') if (R/'LOD_CONTROLLER.json').exists() else {}
    if not alive(info):child,info=launch()
    last_hour=0;progress={}
    while True:
        state=p.read(R/'LOD_PIPELINE.json')
        if time.time()-last_hour>=3600 or state['phase'] in ['finished','blocked','held']:
            snapshot(state,info);last_hour=time.time()
        if state['phase']=='finished':return
        if state['phase'] in ['blocked','held'] or (R/'HOLD').exists():return
        if not alive(info):
            if recover(state):child,info=launch()
            else:
                p.write(R/'reports/SUPERVISOR_ALERT.json',dict(error='控制器退出，缺少可自动恢复的明确证据；保留现场',checked_unix=time.time()))
                return
        active=running(state)
        stalled=[];current={}
        for _,item,evaluation in active:
            out=Path(item['attempt']);paths=[out/('evaluate.log' if evaluation else 'train.log'),out/'model/real_lod_steps.csv']
            pid=item['process']['pid'];signature=tuple(v.stat().st_size if v.exists() else 0 for v in paths)
            old=progress.get(pid);last=old[1] if old and old[0]==signature else time.monotonic()
            current[pid]=(signature,last)
            if time.monotonic()-last>900:stalled.append(dict(pid=pid,idle_seconds=time.monotonic()-last,attempt=str(out)))
        progress=current
        if stalled:p.write(R/'reports/STALL_ALERT.json',dict(checked_unix=time.time(),active=stalled))
        if shutil.disk_usage(R).free<5*2**30:
            (R/'HOLD').write_text('磁盘不足5GiB，暂停后续工作项，不擅自清理模型\n')
            snapshot(state,info);return
        time.sleep(60)


if __name__=='__main__':main()

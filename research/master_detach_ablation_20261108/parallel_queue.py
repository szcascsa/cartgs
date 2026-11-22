"""训练与评价各一个独立worker；只有控制器写主队列和指标总表。"""
from pathlib import Path
import collections
import copy
import fcntl
import json
import os
import resource
import shutil
import subprocess
import sys
import time
import run_batch as b
import supervisor

R=b.R;p=b.p
EXCLUSIVE='VECtor/corridors-dolly/stereo'


def worker(kind,path):
    state=p.read(path)
    def save(value):
        value['updated_at']=time.time();p.write(path,value)
    p.save=save
    job=state['jobs'][0]
    try:
        if kind in ['train','validate']:
            smoke=6 if kind=='validate' else 0
            folder=R/'validation'/(job['dataset']+'/'+job['mode']) if smoke else R/'runs'/job['id']
            out=Path(job['attempt']) if job['state']=='trained' else b.train(job,state,folder,smoke=smoke)
            assert out is not None or not smoke,'入口验收重试用尽'
            if out is not None and not smoke:job['evaluations']=b.cases()
        if kind in ['eval','validate']:
            assert job['state']=='trained' and Path(job['trained_model']).is_dir()
            if kind=='validate':
                for scope,ratio in [('full',1.),('global',.01),('global',.03),('global',.7),('view_ratio',.01)]:
                    if not any(e['scope']==scope and e['ratio']==ratio for e in job['evaluations']):
                        job['evaluations'].append(dict(scope=scope,ratio=ratio,kind='validation',state='pending'))
            for item in job['evaluations']:
                if item['state'] in ['complete','skipped_oom']:continue
                assert item['state'] in ['pending','retry_pending']
                root=(R/'validation'/(job['dataset']+'/'+job['mode']) if kind=='validate' else R/'evaluation'/job['id'])
                b.evaluate(job,item,state,root/('eval_'+item['scope']+'_'+str(item['ratio']) if kind=='validate'
                                             else item['scope']+'_'+str(item['ratio'])))
                if kind=='validate':assert item['state']=='complete'
            job['state']=('validated' if kind=='validate' else 'complete') if all(
                e['state']=='complete' for e in job['evaluations']) else 'partial_evaluation'
        state['worker_phase']='done';save(state)
    except BaseException as error:
        state.update(worker_phase='error',worker_error=repr(error));save(state);raise


def next_work(state,lane):
    active=state.get('parallel_workers',{})
    if any(v['job']==EXCLUSIVE for v in active.values()):return None
    busy={v['job'] for v in active.values()}
    for job in state['jobs']:
        if job['id'] in busy:continue
        if lane=='eval' and job['state']=='trained':kind='eval'
        elif lane=='train' and job['state'] in ['pending','retry_pending']:
            kind='train' if job['dataset']+'/'+job['mode'] in state['validated_modes'] else 'validate'
        else:continue
        if job['id']==EXCLUSIVE and active:return None
        return kind,job
    return None


def resources_available():
    available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()
                   if l.startswith('MemAvailable:'))
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
             '--format=csv,noheader,nounits'],text=True).splitlines()[0])
    return available>=8*2**30 and free>=6144


def launch(state,lane,kind,job):
    folder=R/'workers'/lane;folder.mkdir(parents=True,exist_ok=True)
    path=folder/(str(time.time_ns())+'.json')
    item=copy.deepcopy(state.get('active_validation') or job) if kind=='validate' else copy.deepcopy(job)
    if kind=='validate' and not state.get('active_validation'):
        for key in ['attempt','process','trained_model']:item.pop(key,None)
        item.update(state='pending',attempt_history=[],evaluations=[],health_failures=0,oom_failures=0)
    payload=dict(jobs=[item],protocol=copy.deepcopy(state['protocol']),active_job=item['id'],worker_phase='running')
    p.write(path,payload)
    with path.with_suffix('.log').open('w') as log:
        child=subprocess.Popen([sys.executable,'-u',str(R/'parallel_queue.py'),'--worker',kind,str(path)],
                 cwd=R,env=b.environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state.setdefault('parallel_workers',{})[lane]=dict(job=job['id'],kind=kind,path=str(path),
                        process=p.identity(child.pid),started_unix=time.time())
    p.save(state);print('LAUNCHED',lane,kind,job['id'],child.pid,flush=True)


def collect(state,lane):
    info=state['parallel_workers'][lane];result=p.read(Path(info['path']));job=result['jobs'][0]
    if info['kind']=='validate':state['active_validation']=job
    else:
        index=next(i for i,j in enumerate(state['jobs']) if j['id']==info['job']);state['jobs'][index]=job
    if supervisor.alive(info['process']):return None
    # worker异常退出时，任何仍活着的原生子进程都必须保留，不得重复发射。
    assert not any(supervisor.alive(item.get('process',{})) for item in [job,*job.get('evaluations',[])]), 'worker退出但子进程仍活着，保留现场'
    del state['parallel_workers'][lane]
    if result.get('worker_phase')!='done':return result.get('worker_error','worker退出且缺少完成凭据')
    if info['kind']=='validate':
        assert job['state']=='validated'
        kind=job['dataset']+'/'+job['mode'];state['validated_modes'].append(kind)
        state.setdefault('validation_history',[]).append(job);state.pop('active_validation',None)
    state.setdefault('parallel_history',[]).append(dict(info,finished_unix=time.time(),state=job['state']))
    print('FINISHED',lane,info['kind'],info['job'],job['state'],flush=True)
    return None


def main():
    assert not (R/'HOLD').exists() and p.read(R/'START_AUTHORIZED.json')['authorized']
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    lock=(R/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p.write(R/'LOD_CONTROLLER.json',p.identity(os.getpid()))
    state=p.read(R/'LOD_PIPELINE.json');assert len(state['jobs'])==32
    state.setdefault('parallel_workers',{});state.pop('error',None)
    error=None
    try:
        while True:
            for lane in list(state['parallel_workers']):
                failure=collect(state,lane)
                if failure:error=error or failure
            held=(R/'HOLD').exists()
            if shutil.disk_usage(R).free<5*2**30:
                (R/'HOLD').write_text('磁盘不足5GiB，暂停后续工作项，不清理模型\n');held=True
            if not held and not error:
                # 训练先预约；高内存双目任务等待现有评价完成后独占。
                for lane in ['train','eval']:
                    if lane not in state['parallel_workers']:
                        work=next_work(state,lane)
                        if work and resources_available():launch(state,lane,*work)
            state['active_training_job']=state['parallel_workers'].get('train',{}).get('job')
            state['active_evaluation_job']=state['parallel_workers'].get('eval',{}).get('job')
            state['active_job']=state['active_training_job'] or state['active_evaluation_job']
            state['phase']='parallel_draining' if (held or error) else 'parallel_running'
            p.save(state);b.report(state)
            if not state['parallel_workers']:
                if held or error:
                    state.update(phase='held' if held else 'blocked',error=error or 'HOLD存在')
                    p.save(state);return
                if all(j['state'] in b.TERMINAL for j in state['jobs']):break
                assert all(j['state'] in b.TERMINAL|{'pending','retry_pending','trained'} for j in state['jobs']), '无法接管不明running状态'
            time.sleep(2)
        before=p.read(R/'PROTECTED_BEFORE.json');after={k:{n:p.sha(Path(n)) for n in v} for k,v in before.items()}
        assert after==before,'只读Full或生产文件变动';p.write(R/'PROTECTED_AFTER.json',after)
        state.update(phase='finished',active_job=None);p.save(state);b.report(state)
        p.write(R/'FINISHED.json',dict(passed=all(j['state'] in ['complete','skipped_by_user'] for j in state['jobs']),counts=state['counts'],images_saved=0))
        audit_request=state.get('final_audit_request','EXTRA8_REQUEST.json')
        if (R/'reports'/audit_request).exists():
            audit_log=audit_request.replace('_REQUEST.json','_final_audit.log').lower()
            with (R/'logs'/audit_log).open('w') as log:
                subprocess.run([sys.executable,str(R/'finalize_rerun_20261111.py'),
                    '--request',audit_request],cwd=R,env=b.environment(),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
    except BaseException as exc:
        state.update(phase='blocked',error=repr(exc));p.save(state);raise


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2],Path(sys.argv[3]))
    else:main()

"""恢复关机中断的单个AutoDL队列，保留完成结果和中断现场。"""
from pathlib import Path
import copy
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT='/home/prmi/zwj/experiments/no_importance_all_scenes_20261115'
WORKER=os.environ.get('CARTGS_CLUSTER_WORKER','cloud_a')
assert WORKER in ['cloud_a','cloud_b']
os.environ.update(CARTGS_CLUSTER_ROOT=ROOT,CARTGS_CLUSTER_WORKER=WORKER,CARTGS_PIPELINE_OVERLAP='1',TORCH_HOME='/home/prmi/.cache/torch')
import run as r
from prepare import R,read,write,sha


def identity_alive(value):
    p=Path('/proc')/str(value['pid'])/'stat'
    if not p.exists():return False
    fields=p.read_text().split(') ',1)[1].split()
    return fields[0]!='Z' and fields[19]==value['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==value['boot_id']


def main():
    apply='--apply' in sys.argv
    locks=[]
    for name in ['controller.lock','cloud_session.lock','shutdown_recovery.lock']:
        lock=(R/name).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    old=read(R/'LOD_CONTROLLER.json');assert not identity_alive(old),'旧控制器仍存活，禁止重复启动'
    assert read(R/'reports/CLOUD_LAUNCH.json')['worker']==WORKER
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name)==os.getpid():continue
        try:args=(d/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        assert not any(str(R/n).encode() in args for n in ['run.py','cluster_start.py','cloud_pipeline_handoff.py','runtime/bin/lod_from_full','eval_joint_entry.py']),'存在本队列子进程，禁止重复恢复'
    assert read(R/'reports/CLOUD_VALIDATION.json')['passed']
    assert __import__('shutil').disk_usage(R).free>5*2**30
    before=read(R/'LOD_PIPELINE.json');state=copy.deepcopy(before);actions=[];unchanged=[]
    for job in state['jobs']:
        if job.get('assigned_worker')!=WORKER:continue
        if job['state'] in ['trained','complete']:
            model=Path(job['trained_model']);assert read(model.parent/'D_VERIFIED.json')['passed']
            checkpoint,=model.glob('*_shutdown/ply/selector.pt');assert checkpoint.stat().st_size>0
        for scope,item in [('train',job)]+[(e['scope']+'_'+str(e['ratio']),e) for e in job['evaluations']]:
            if item['state']!='running':continue
            assert not identity_alive(item['process']),'子进程仍存活，不能复用其任务'
            out=Path(item['attempt']);assert out.exists() and not (out/'exit.json').exists()
            assert not any(a['output']==str(out) for a in item.get('attempt_history',[]))
            assert not (out/'model/lod_health_failure.json').exists(),'存在健康失败，应先定位，不能归因关机'
            if scope=='train':
                assert not list(out.glob('model/*_shutdown/ply/selector.pt')),'发现最终checkpoint，需核验后收割'
            complete_outputs=scope!='train' and (out/'metric_summary.json').exists()
            if complete_outputs:
                import csv
                import math
                summary=read(out/'metric_summary.json')
                with (out/'per_frame_metrics.csv').open() as f:rows=list(csv.DictReader(f))
                assert len(rows)==job['full_evaluation']['result']['evaluated_frames']==summary['evaluated_frames']
                for name in ['psnr','ssim','lpips','render_ms']:
                    values=[float(row[name]) for row in rows];assert all(math.isfinite(v) for v in values)
                    assert sum(values)/len(values)==summary[name]
                old_save=r.save;r.save=lambda _:None
                try:r.finish_evaluation(job,item,state,out)
                finally:r.save=old_save
            log=out/('train.log' if scope=='train' else 'evaluate.log');trace=out/'model/real_lod_steps.csv'
            progress=trace if trace.exists() else log
            step=None
            if trace.exists():
                with trace.open('rb') as stream:
                    stream.seek(max(0,trace.stat().st_size-2048));lines=stream.read().decode(errors='replace').splitlines()
                step=next((int(line.split(',')[0]) for line in reversed(lines) if line.split(',')[0].isdigit()),None)
            info=dict(output=str(out),process=item['process'],returncode=None,
                seconds=max(0,progress.stat().st_mtime-read(out/'launch.json')['started_unix']),
                state='recovered_complete_outputs' if complete_outputs else 'interrupted_by_shutdown',
                reason='完整评价文件核验通过；退出码未知' if complete_outputs else '用户确认关机；旧进程消失；输出未完成',oom=None,health=None,
                completed_steps=step,progress_bytes=progress.stat().st_size,progress_sha256=sha(progress),
                source_restart='none' if complete_outputs else ('same_original_full_checkpoint' if scope=='train' else 'same_trained_model_same_ratio'))
            item.setdefault('attempt_history',[]).append(info)
            if not complete_outputs:
                item.update(state='pending',shutdown_restarts=item.get('shutdown_restarts',0)+1)
                for key in ['step','trace_bytes','log_bytes','no_progress_seconds','waiting_gpu_memory_mib']:item.pop(key,None)
                base=out.parent;assert not (base/f"attempt_{len(item['attempt_history'])+1:02d}").exists()
            actions.append(dict(id=job['id'],scope=scope,previous=info,oom_failures=item.get('oom_failures',0)))
        previous=next(j for j in before['jobs'] if j['id']==job['id'])
        if previous==job:unchanged.append(job['id'])
        for e in previous['evaluations']:
            if e['state']=='complete':assert e==next(x for x in job['evaluations'] if (x['scope'],x['ratio'])==(e['scope'],e['ratio']))
    assert len(actions)==4 and sum(a['scope']=='train' for a in actions)==2,'关机现场已变化，需重新核查'
    result=dict(passed=True,apply=apply,worker=WORKER,previous_controller=old,actions=actions,unchanged_jobs=unchanged,
        complete_jobs_preserved=sum(j['state']=='complete' for j in state['jobs'] if j.get('assigned_worker')==WORKER),
        completed_evaluations_preserved=sum(e['state']=='complete' for j in state['jobs'] if j.get('assigned_worker')==WORKER for e in j['evaluations']),
        completed_trainings_preserved=sum(j['state'] in ['trained','complete'] for j in state['jobs'] if j.get('assigned_worker')==WORKER),
        target_steps_unchanged=all(a['target_steps']==b['target_steps'] for a,b in zip(before['jobs'],state['jobs'])),
        inspected_unix=time.time())
    if not apply:
        write(R/'reports/SHUTDOWN_RECOVERY_PLAN.json',result);print(json.dumps(result,ensure_ascii=False));return
    audit=R/'reports'/('shutdown_recovery_'+str(time.time_ns()));audit.mkdir()
    write(audit/'PIPELINE_BEFORE.json',before);write(audit/'RECOVERY_PLAN.json',result)
    for action in actions:
        out=Path(action['previous']['output']);write(out/'shutdown_interruption.json',action['previous'])
        if action['previous']['state']=='recovered_complete_outputs':
            write(out/'exit.json',dict(returncode=None,recovered_complete_outputs=True,seconds=action['previous']['seconds']))
            if not (out/'train.log').exists():(out/'train.log').symlink_to('evaluate.log')
    state.update(phase='recovering_after_shutdown');r.save(state)
    for lock in locks:lock.close()
    with (R/'logs/cloud_shutdown_recovery.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u',str(R/'run.py')],cwd=R,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    ident=r.p.identity(child.pid)
    for _ in range(100):
        if read(R/'LOD_CONTROLLER.json')==ident:break
        assert child.poll() is None,'恢复调度器退出；保留现场'
        time.sleep(.1)
    else:raise RuntimeError('恢复调度器未获得控制权')
    write(R/'reports/CLOUD_LAUNCH.json',dict(worker=WORKER,identity=ident,started_unix=time.time(),scheduling='2_train_plus_2_evaluate',recovery_audit=str(audit)))
    result.update(new_controller=ident,audit=str(audit),started_unix=time.time());write(R/'reports/SHUTDOWN_RECOVERY.json',result)
    print('SHUTDOWN_RECOVERY_STARTED',json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

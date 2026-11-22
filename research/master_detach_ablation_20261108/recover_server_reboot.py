"""有boot_id变化证据时恢复本批中断任务，保留失败现场与原重试计数。"""
from pathlib import Path
import argparse
import copy
import fcntl
import json
import time
import run_batch as b
import supervisor

R=b.R;p=b.p


def recover_item(item,old_boot,current_boot):
    assert old_boot!=current_boot
    assert item['process']['boot_id']==old_boot
    assert not supervisor.alive(item['process']), '原任务仍活着，拒绝重跑'
    out=Path(item['attempt']);out.resolve().relative_to(R.resolve())
    assert not (out/'exit.json').exists(), '有退出凭据，应按原退出原因恢复'
    assert not (out/'model/lod_health_failure.json').exists(), '有健康失败凭据，不能作为重启中断'
    assert not (out/'memory_reserve_stop.json').exists(), '有内存保护凭据，不能作为重启中断'
    before={key:item.get(key,0) for key in ['health_failures','oom_failures']}
    record=dict(output=str(out),process=copy.deepcopy(item['process']),returncode=None,
                seconds=None,health=None,oom=None,interrupted_server_reboot=True,
                old_boot_id=old_boot,new_boot_id=current_boot,recovered_unix=time.time())
    history=item.setdefault('attempt_history',[])
    assert not any(h['output']==str(out) for h in history), '该次尝试已归档'
    history.append(record)
    item['reboot_interruptions']=item.get('reboot_interruptions',0)+1
    item.update(state='retry_pending',error='服务器重启中断；新开attempt，健康和OOM计数不变')
    assert before=={key:item.get(key,0) for key in before}
    return record


def recover_workers(state,current_boot):
    records=[]
    for lane,info in list(state.get('parallel_workers',{}).items()):
        assert not supervisor.alive(info['process'])
        path=Path(info['path']);payload=p.read(path);job=payload['jobs'][0]
        assert job['id']==info['job']
        assert info['process']['boot_id']!=current_boot
        assert payload['worker_phase']=='running', '非重启运行状态需独立核查'
        for item in [job,*job.get('evaluations',[])]:
            assert not supervisor.alive(item.get('process',{}))
        if info['kind']=='train':
            assert job['state']=='running' and not job.get('trained_model')
            records.append(recover_item(job,info['process']['boot_id'],current_boot))
        elif info['kind']=='eval':
            assert job['state']=='trained' and Path(job['trained_model']).is_dir()
            for item in job['evaluations']:
                if item['state']=='running':
                    records.append(recover_item(item,info['process']['boot_id'],current_boot))
        else:
            raise AssertionError('入口验收中断需独立验证，不能按正式任务恢复')
        index=next(i for i,j in enumerate(state['jobs']) if j['id']==job['id'])
        state['jobs'][index]=job
        state.setdefault('reboot_worker_history',[]).append(dict(info,recovered_unix=time.time()))
        del state['parallel_workers'][lane]
    state.update(phase='recovered_server_reboot',error='',active_job=None,
                 active_training_job=None,active_evaluation_job=None)
    return records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    lock=(R/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ['LOD_CONTROLLER.json','SUPERVISOR.json','reports/PAUSE_WAITER.json']:
        assert not supervisor.alive(p.read(R/name) if (R/name).exists() else {})
    original=p.read(R/'LOD_PIPELINE.json');state=copy.deepcopy(original)
    current=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    records=recover_workers(state,current)
    assert len(records)==2 and not supervisor.running(state)
    # 已完成模型和数值必须能重新核验，不根据中断现场的部分结果宣告成功。
    models=0;evaluations=0
    for job in state['jobs']:
        if job.get('trained_model'):
            b.verify_training(Path(job['trained_model']).parent,job['target_steps']);models+=1
        for item in job.get('evaluations',[]):
            if item['state']=='complete':
                summary=p.read(Path(item['result']['output'])/'metric_summary.json')
                assert summary['evaluated_frames']==job['full_evaluation']['result']['evaluated_frames']
                assert all(summary[k]==item['result'][k] for k in ['psnr','ssim','lpips','ate_rmse'])
                evaluations+=1
    for name,h in p.read(R/'reports/BUILD_HASHES.json').items():assert p.sha(R/name)==h
    protected=p.read(R/'PROTECTED_BEFORE.json')
    assert all(p.sha(Path(name))==h for values in protected.values() for name,h in values.items())
    receipt=dict(passed=True,apply=args.apply,new_boot_id=current,records=records,
                 verified_models=models,verified_evaluations=evaluations,
                 protected_files_unchanged=sum(map(len,protected.values())),
                 checked_unix=time.time(),no_oom_or_health_counter_increment=True)
    if args.apply:
        backup=R/'reports'/('reboot_backup_'+str(time.time_ns()));backup.mkdir()
        p.write(backup/'LOD_PIPELINE.json',original)
        for lane,info in original['parallel_workers'].items():
            p.write(backup/(lane+'_worker.json'),p.read(Path(info['path'])))
        p.save(state);b.report(state)
        p.write(R/'reports/SERVER_REBOOT_RECOVERY.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

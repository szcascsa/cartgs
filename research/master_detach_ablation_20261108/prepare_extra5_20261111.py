"""最终失败三组各新增5次机会；保留前15次历史和已完成结果。"""
from pathlib import Path
import copy
import fcntl
import json
import time
import run_batch as b
import supervisor
import memory_guard

R=b.R;p=b.p
TARGETS={'TUM_RGBD/freiburg1_room/mono','VECtor/corridors-dolly/mono',
         'VECtor/corridors-dolly/stereo'}


def requeue(state):
    assert state['phase']=='finished' and not state['parallel_workers']
    assert {j['id'] for j in state['jobs'] if j['state']!='complete'}==TARGETS
    restored=[]
    for job in state['jobs']:
        if job['id'] not in TARGETS:continue
        assert job['state']=='skipped_health' and job['health_failures']==15
        assert len(job['attempt_history'])==15 and not job.get('trained_model') and not job['evaluations']
        job.update(state='retry_pending',max_training_attempts=20,max_health_retries=19)
        job.setdefault('manual_rerun_history',[]).append(dict(authorized_unix=time.time(),
            prior_health_failures=15,prior_attempts=15,first_attempt=16,last_attempt=20,
            additional_attempts=5,stop_after_first_success=True,reason='用户授权每个失败场景再次额外5次'))
        restored.append(job['id'])
    state.update(phase='extra5_prepared',error='',final_audit_request='EXTRA5_REQUEST.json')
    return restored


def main():
    locks=[]
    for name in ['controller.lock','supervisor.lock']:
        lock=(R/name).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    assert not (R/'HOLD').exists()
    for name in ['SUPERVISOR.json','LOD_CONTROLLER.json']:assert not supervisor.alive(p.read(R/name))
    original=p.read(R/'LOD_PIPELINE.json');assert not supervisor.running(original)
    state=copy.deepcopy(original);targets=requeue(state);assert state['protocol']==original['protocol']
    assert state['protocol']['max_health_retries']==5 and state['protocol']['max_oom_retries']==2
    assert all(p.sha(R/n)==h for n,h in p.read(R/'reports/BUILD_HASHES.json').items())
    memory_guard.snapshot(p.read(R/'configs/corridors_stereo_memory.json'))
    assert not Path(p.read(R/'configs/corridors_stereo_memory.json')['cgroup'],'cgroup.procs').read_text().strip()
    backup=R/'reports'/('extra5_backup_'+str(time.time_ns()));backup.mkdir()
    for name in ['LOD_PIPELINE.json','FINISHED.json','reports/FINAL_AUDIT.json',
                 'reports/MASTER_DETACH_ABLATION_FINAL_REPORT.md','reports/LOD_METRICS.csv']:
        path=R/name
        if path.exists():(backup/Path(name).name).write_bytes(path.read_bytes())
    models={str(path):p.sha(path) for job in original['jobs'] if job['state']=='complete'
            for path in Path(job['trained_model']).rglob('*') if path.is_file()}
    p.write(backup/'COMPLETED_MODEL_HASHES.json',models)
    request=dict(prepared_unix=time.time(),targets=targets,first_attempt=16,last_attempt=20,
        additional_attempts_each=5,total_cap_includes_oom=True,stop_after_first_success=True,
        old_health_failures_preserved=True,global_protocol_unchanged=True,
        max_oom_retries=2,completed_groups_preserved=29,completed_evaluations_preserved=667,
        backup=str(backup),checkpoint_hashes=str(backup/'COMPLETED_MODEL_HASHES.json'),
        evaluation_cases_per_success=23,save_images=False)
    p.write(R/'reports/EXTRA5_REQUEST.json',request)
    p.save(state);b.report(state)
    p.write(R/'FINISHED.json',dict(passed=False,queue_finished=False,rerun_in_progress=True,
        previous_finished_snapshot=str(backup/'FINISHED.json')))
    p.write(R/'reports/FINAL_AUDIT.json',dict(audit_pending=True,rerun_in_progress=True,
        previous_verified_audit=str(backup/'FINAL_AUDIT.json')))
    (R/'reports/MASTER_DETACH_ABLATION_FINAL_REPORT.md').write_text(
        '# 最终失败三组：各额外5次重跑进行中\n\n'
        '仅fr1_room Mono、corridors-dolly Mono和双目，各授权attempt_16至attempt_20。'
        '首次成功即停止重训并执行23项全轨迹评价，不保存图片。原15次失败和其他29组667项评价保留。'
        'OOM保留原额外2次上限，并计入本轮5次总机会。训练配置和健康阈值不变。\n\n'
        '前轮完整结果：'+str(backup)+'\n')
    with (R/'reports/ABLATION_PROTOCOL.md').open('a') as out:
        out.write('\n\n## 2026-11-11：失败三组再次各额外5次机会\n\n'
            '每组attempt_16至attempt_20，保留前15次失败；最多新增5次训练尝试，首次成功即转23项评价。'
            '使用任务私有max_training_attempts=20、max_health_retries=19，全局上限和其他29组不变。'
            'OOM仍最多额外2次且计入总尝试数。不得放宽健康阈值，不保存图片。\n')
    print(json.dumps(request,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

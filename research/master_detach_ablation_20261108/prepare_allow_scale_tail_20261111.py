"""允许Scale长尾后，三组各新增5次机会；保留前20次历史。"""
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
        assert job['state']=='skipped_health' and job['health_failures']==20
        assert len(job['attempt_history'])==20 and not job.get('trained_model') and not job['evaluations']
        job.update(state='retry_pending',max_training_attempts=25,max_health_retries=24)
        job.setdefault('manual_rerun_history',[]).append(dict(authorized_unix=time.time(),
            prior_health_failures=20,prior_attempts=20,first_attempt=21,last_attempt=25,
            additional_attempts=5,stop_after_first_success=True,reason='用户允许Scale长尾后重跑，沿用每组额外5次额度'))
        restored.append(job['id'])
    state.update(phase='allow_scale_tail_prepared',error='',final_audit_request='ALLOW_SCALE_TAIL_REQUEST.json')
    return restored


def main():
    locks=[]
    for name in ['controller.lock','supervisor.lock']:
        lock=(R/name).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    assert not (R/'HOLD').exists()
    assert p.read(R/'reports/ALLOW_SCALE_TAIL_BUILD.json')['passed']
    assert 'LoDHealth.scale_factor_limit: 0.' in p.config().read_text()
    for name in ['SUPERVISOR.json','LOD_CONTROLLER.json']:assert not supervisor.alive(p.read(R/name))
    original=p.read(R/'LOD_PIPELINE.json');assert not supervisor.running(original)
    state=copy.deepcopy(original);targets=requeue(state);assert state['protocol']==original['protocol']
    assert state['protocol']['max_health_retries']==5 and state['protocol']['max_oom_retries']==2
    assert all(p.sha(R/n)==h for n,h in p.read(R/'reports/BUILD_HASHES.json').items())
    memory_guard.snapshot(p.read(R/'configs/corridors_stereo_memory.json'))
    assert not Path(p.read(R/'configs/corridors_stereo_memory.json')['cgroup'],'cgroup.procs').read_text().strip()
    backup=R/'reports'/('allow_scale_tail_backup_'+str(time.time_ns()));backup.mkdir()
    for name in ['LOD_PIPELINE.json','FINISHED.json','reports/FINAL_AUDIT.json',
                 'reports/MASTER_DETACH_ABLATION_FINAL_REPORT.md','reports/LOD_METRICS.csv']:
        path=R/name
        if path.exists():(backup/Path(name).name).write_bytes(path.read_bytes())
    models={str(path):p.sha(path) for job in original['jobs'] if job['state']=='complete'
            for path in Path(job['trained_model']).rglob('*') if path.is_file()}
    p.write(backup/'COMPLETED_MODEL_HASHES.json',models)
    request=dict(prepared_unix=time.time(),targets=targets,first_attempt=21,last_attempt=25,
        additional_attempts_each=5,total_cap_includes_oom=True,stop_after_first_success=True,
        old_health_failures_preserved=True,global_protocol_unchanged=True,
        max_oom_retries=2,completed_groups_preserved=29,completed_evaluations_preserved=667,
        backup=str(backup),checkpoint_hashes=str(backup/'COMPLETED_MODEL_HASHES.json'),
        evaluation_cases_per_success=23,save_images=False,
        scale_tail_stop_enabled=False,scale_diagnostic_limit=100,
        health_policy='Scale长尾只记录不停止；保留原连续零选点及其他运行保护')
    p.write(R/'reports/ALLOW_SCALE_TAIL_REQUEST.json',request)
    p.save(state);b.report(state)
    p.write(R/'FINISHED.json',dict(passed=False,queue_finished=False,rerun_in_progress=True,
        previous_finished_snapshot=str(backup/'FINISHED.json')))
    p.write(R/'reports/FINAL_AUDIT.json',dict(audit_pending=True,rerun_in_progress=True,
        previous_verified_audit=str(backup/'FINAL_AUDIT.json')))
    (R/'reports/MASTER_DETACH_ABLATION_FINAL_REPORT.md').write_text(
        '# 最终失败三组：允许Scale长尾后，各额外5次重跑进行中\n\n'
        '仅fr1_room Mono、corridors-dolly Mono和双目，各授权attempt_21至attempt_25。'
        '首次成功即停止重训并执行23项全轨迹评价，不保存图片。原20次失败和其他29组667项评价保留。'
        'OOM保留原额外2次上限，并计入本轮5次总机会。仅关闭Scale长尾停止条件，仍按100倍记录诊断；训练配置和其他保护不变。\n\n'
        '前轮完整结果：'+str(backup)+'\n')
    with (R/'reports/ABLATION_PROTOCOL.md').open('a') as out:
        out.write('\n\n## 2026-11-11：允许Scale长尾，失败三组再次各额外5次机会\n\n'
            '每组attempt_21至attempt_25，保留前20次失败；最多新增5次训练尝试，首次成功即转23项评价。'
            '使用任务私有max_training_attempts=25、max_health_retries=24，全局上限和其他29组不变。'
            'OOM仍最多额外2次且计入总尝试数。关闭Scale长尾停止但保留统计，连续零选点和其他运行保护不变，不保存图片。\n')
    print(json.dumps(request,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

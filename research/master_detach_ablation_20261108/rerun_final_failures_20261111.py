"""用户授权：最终失败三组各新增一次尝试，成功后由原队列自动评价。"""
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
    failed={j['id'] for j in state['jobs'] if j['state']!='complete'}
    assert failed==TARGETS,failed
    changed=[]
    for job in state['jobs']:
        if job['id'] not in TARGETS:continue
        assert job['state']=='skipped_health' and job['health_failures']==6
        assert len(job['attempt_history'])==6 and not job.get('trained_model')
        assert not job['evaluations']
        job.setdefault('manual_rerun_history',[]).append(dict(authorized_unix=time.time(),
            prior_state=job['state'],prior_health_failures=job['health_failures'],
            prior_oom_failures=job.get('oom_failures',0),next_attempt=7,
            additional_health_attempts=1,reason='用户要求最终失败组重跑，成功后重新渲染'))
        job['state']='retry_pending'
        changed.append(job['id'])
    state.update(phase='rerun_final_failures_prepared',error='')
    return changed


def main():
    locks=[]
    for name in ['controller.lock','supervisor.lock']:
        handle=(R/name).open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
    for name in ['SUPERVISOR.json','LOD_CONTROLLER.json']:
        assert not supervisor.alive(p.read(R/name))
    assert not (R/'HOLD').exists()
    original=p.read(R/'LOD_PIPELINE.json');assert not supervisor.running(original)
    state=copy.deepcopy(original);changed=requeue(state)
    assert state['protocol']==original['protocol']
    assert state['protocol']['max_health_retries']==5 and state['protocol']['max_oom_retries']==2
    for name,h in p.read(R/'reports/BUILD_HASHES.json').items():assert p.sha(R/name)==h
    memory_guard.snapshot(p.read(R/'configs/corridors_stereo_memory.json'))
    assert not Path(p.read(R/'configs/corridors_stereo_memory.json')['cgroup'],'cgroup.procs').read_text().strip()
    completed={j['id']:j for j in original['jobs'] if j['state']=='complete'}
    checkpoint_hashes={}
    for job in completed.values():
        for path in Path(job['trained_model']).rglob('*'):
            if path.is_file():checkpoint_hashes[str(path)]=p.sha(path)
    backup=R/'reports'/('final_failures_rerun_'+str(time.time_ns()));backup.mkdir()
    for name in ['LOD_PIPELINE.json','FINISHED.json','reports/FINAL_AUDIT.json',
                 'reports/MASTER_DETACH_ABLATION_FINAL_REPORT.md','reports/LOD_METRICS.csv']:
        path=R/name
        if path.exists():(backup/Path(name).name).write_bytes(path.read_bytes())
    p.write(backup/'COMPLETED_MODEL_HASHES.json',checkpoint_hashes)
    receipt=dict(prepared_unix=time.time(),targets=changed,next_attempt=7,
        additional_health_attempts_each=1,global_retry_limit_unchanged=True,
        completed_groups_preserved=29,completed_evaluations_preserved=667,
        backup=str(backup),checkpoint_hashes=str(backup/'COMPLETED_MODEL_HASHES.json'),
        evaluation_cases_per_success=23,save_images=False)
    p.write(R/'reports/FINAL_FAILURES_RERUN_REQUEST.json',receipt)
    p.save(state);b.report(state)
    # 原终态凭据已完整归档，当前根目录显式标记重跑进行中。
    p.write(R/'FINISHED.json',dict(passed=False,queue_finished=False,rerun_in_progress=True,
                                 previous_finished_snapshot=str(backup/'FINISHED.json')))
    p.write(R/'reports/FINAL_AUDIT.json',dict(audit_passed=False,audit_pending=True,
                rerun_in_progress=True,previous_verified_audit=str(backup/'FINAL_AUDIT.json')))
    report=R/'reports/MASTER_DETACH_ABLATION_FINAL_REPORT.md'
    report.write_text('# 最终失败三组重新运行\n\n此前29组667项评价已完成且核验通过。'
        '本次只对fr1_room Mono、corridors-dolly Mono和双目各新增第7次尝试；原6次失败计数保留，'
        '全局重试上限及健康阈值不变。成功模型自动进行原23项全轨迹评价，不保存图片。'
        '\n\n当前状态见LOD_PIPELINE.json；前一轮完整报告、审计、指标与队列已归档到：\n\n'+str(backup)+'\n')
    protocol=R/'reports/ABLATION_PROTOCOL.md'
    with protocol.open('a') as out:
        out.write('\n\n## 2026-11-11用户授权重跑最终失败组\n\n'
            '仅3个最终失败组各新增第7次尝试；原健康失败6次及所有attempt保留，不修改全局健康重试上限。'
            '本次再次健康失败即停止该组；OOM仍按既有额外2次上限。成功后完成全局11档、局部11档与Full，'
            '共23项全轨迹评价。其他29组模型及667项评价不重跑。\n')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

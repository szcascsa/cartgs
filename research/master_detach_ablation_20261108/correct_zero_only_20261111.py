"""修正遗漏的XYZ/概率饱和停止；已完成模型不重训。"""
from pathlib import Path
import copy
import fcntl
import time
import run_batch as b
import supervisor

R=b.R;p=b.p
MONO='VECtor/corridors-dolly/mono'
STEREO='VECtor/corridors-dolly/stereo'


def correct(state):
    assert state['phase']=='held' and not state['parallel_workers']
    jobs={j['id']:j for j in state['jobs']}
    mono=jobs[MONO];stereo=jobs[STEREO]
    assert stereo['state']=='retry_pending' and len(stereo['attempt_history'])==20
    replaced=[]
    if mono['state']=='skipped_health':
        assert mono['health_failures']==25 and len(mono['attempt_history'])==25
        assert all(h.get('health',{}).get('error')=='view_budget_compensation_xyz_tail'
                   for h in mono['attempt_history'][20:])
        mono.update(state='retry_pending',max_training_attempts=30,max_health_retries=29)
        mono.setdefault('manual_rerun_history',[]).append(dict(corrected_unix=time.time(),
            replaced_attempts=[21,22,23,24,25],first_attempt=26,last_attempt=30,
            reason='前5次被遗漏的XYZ停止条件中断；修正后重新执行已授权任务',stop_after_first_success=True))
        replaced=[21,22,23,24,25]
    else:assert mono['state'] in ['trained','complete']
    state.update(phase='zero_only_corrected',error='')
    return replaced


def main():
    locks=[]
    for n in ['controller.lock','supervisor.lock']:
        lock=(R/n).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    assert (R/'HOLD').read_text().startswith('策略修正临时排空')
    for n in ['SUPERVISOR.json','LOD_CONTROLLER.json']:assert not supervisor.alive(p.read(R/n))
    assert p.read(R/'reports/ZERO_ONLY_BUILD.json')['passed']
    assert all(p.sha(R/n)==h for n,h in p.read(R/'reports/BUILD_HASHES.json').items())
    config=p.config().read_text()
    assert all(line in config for line in ['LoDHealth.scale_factor_limit: 0.',
        'LoDHealth.xyz_anchor_limit: 0.','LoDHealth.saturation_patience: 0'])
    original=p.read(R/'LOD_PIPELINE.json');assert not supervisor.running(original)
    state=copy.deepcopy(original);replaced=correct(state)
    assert state['protocol']==original['protocol']
    before={j['id']:j for j in original['jobs']}
    assert all(j==before[j['id']] for j in state['jobs'] if j['state']=='complete')
    backup=R/'reports'/('zero_only_correction_backup_'+str(time.time_ns()));backup.mkdir()
    p.write(backup/'LOD_PIPELINE.json',original)
    hashes={str(f):p.sha(f) for j in original['jobs'] if j['state']=='complete'
            for f in Path(j['trained_model']).rglob('*') if f.is_file()}
    p.write(backup/'COMPLETED_MODEL_HASHES.json',hashes)
    request=p.read(R/'reports/ALLOW_SCALE_TAIL_REQUEST.json')
    p.write(backup/'ALLOW_SCALE_TAIL_REQUEST.json',request)
    request.update(health_policy='Scale/XYZ长尾和概率饱和只记录；仅连续零选点8次或非有限值停止，保留OOM/磁盘保护。'
        '此前corridors单目被遗漏XYZ保护停止的尝试保留，修正后重新执行原任务额度。',
        xyz_tail_stop_enabled=False,saturation_stop_enabled=False,zero_patience=8,
        correction_backup=str(backup),correction_protected_model_hashes=str(backup/'COMPLETED_MODEL_HASHES.json'))
    if replaced:request['attempt_ranges']={MONO:dict(first=21,last=30)}
    state['health_policy_override']=dict(scale_tail_stop=False,xyz_tail_stop=False,
        saturation_stop=False,zero_patience=8,finite_checks=True,selector_config=str(p.config()))
    p.write(R/'reports/ALLOW_SCALE_TAIL_REQUEST.json',request)
    p.save(state);b.report(state)
    receipt=p.read(R/'reports/ZERO_ONLY_CORRECTION.json')
    receipt.update(prepared_unix=time.time(),backup=str(backup),replaced_mono_attempts=replaced,
        completed_groups_preserved=sum(j['state']=='complete' for j in original['jobs']),
        completed_model_files_hashed=len(hashes),policy=state['health_policy_override'])
    p.write(R/'reports/ZERO_ONLY_CORRECTION.json',receipt)
    with (R/'reports/ABLATION_PROTOCOL.md').open('a') as out:
        out.write('\n\n## 2026-11-11：修正仅保留零选点停止的策略\n\n'
            '上一轮只关闭Scale，遗漏XYZ停止，不符合用户“只要不出现0选点”的要求。'
            '现Scale/XYZ长尾与概率饱和只记录，连续零选点保持原8次，非有限值、OOM和磁盘保护保留。'
            'fr1_room Mono成功模型和23项评价保留；corridors单目如前5次均因XYZ停止，则补做26至30次；'
            '双目仍使用原21至25次额度。首次成功即评价，不保存图片。\n')
    print('ZERO_ONLY_CORRECTION_READY',receipt,flush=True)


if __name__=='__main__':main()

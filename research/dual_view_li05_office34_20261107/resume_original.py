"""额外对照验收和汇报后恢复原队列，避免重复控制器和覆盖运行任务。"""
from pathlib import Path
import datetime
import fcntl
import json
import os
import subprocess
import sys

R = Path(__file__).resolve().parent
B = R.parent/'dual_view_li_best_full_20261107'


def main():
    sys.path.insert(0, str(B))
    import run_batch as b
    p = b.p
    extra = p.read(R/'LOD_PIPELINE.json')
    assert extra['phase'] == 'ready_for_comparison'
    assert all(j['state'] in b.TERMINAL for j in extra['jobs'])
    assert p.read(R/'reports/COMPARISON_READY.json')['common_frame_checks']
    assert (R/'reports/RESULTS_REPORTED.json').is_file(), '先汇报额外对照结果'
    def alive(info):
        path = Path('/proc')/str(info['pid'])
        return path.exists() and (path/'stat').read_text().rsplit(') ', 1)[1].split()[0] != 'Z' and p.alive(info)
    for name in ['SUPERVISOR.json', 'LOD_CONTROLLER.json']:
        assert not alive(p.read(B/name)), name+'仍在运行，不能重复启动'
    with (B/'controller.lock').open('a') as controller_lock, (B/'supervisor.lock').open('a') as supervisor_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(supervisor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = p.read(B/'LOD_PIPELINE.json')
        assert state['phase'] == 'held_for_li05_controls' and (B/'HOLD').exists()
        assert len(state['jobs']) == 26 and all(j['dataset'] != 'VECtor' for j in state['jobs'])
        assert all(j['state'] != 'running' and all(e['state'] != 'running' for e in j.get('evaluations', [])) for j in state['jobs'])
        p.write(B/'reports/STATE_BEFORE_LI05_RESUME.json', state)
        p.write(B/'reports/LI05_CONTROLS_COMPLETED.json', dict(
            extra_directory=str(R), summary=str(R/'reports/LI05_COMPARISON_REPORT.md'),
            changed_default_configuration=False, resume_authorization='用户明确要求额外对照完成后继续原队列'))
        (B/'HOLD').unlink()
        state.update(phase='resuming_after_li05_controls', error='', active_job=None)
        p.save(state)
    with (B/'logs/supervisor.log').open('a') as log:
        child = subprocess.Popen([sys.executable, '-u', str(B/'supervisor.py')], cwd=B,
            env=b.environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(resumed_at_hk=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                   original_batch=str(B), supervisor_process=p.identity(child.pid),
                   completed_jobs_before_resume=sum(j['state'] == 'complete' for j in state['jobs']),
                   pending_jobs_before_resume=sum(j['state'] == 'pending' for j in state['jobs']),
                   vector_deferred=True, li_weight_original=.1)
    p.write(R/'reports/ORIGINAL_QUEUE_RESUMED.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()

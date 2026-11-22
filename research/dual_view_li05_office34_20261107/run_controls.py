"""原评价完成后顺序运行两组LI=0.5，留待汇报后恢复原队列。"""
from pathlib import Path
import copy
import fcntl
import os
import resource
import time
import run_batch as b

R = b.R
p = b.p
B = R.parent/'dual_view_li_best_full_20261107'


def verify_li05(out, target):
    ply = b.original_verify(out, target)
    stage = p.read(ply/'real_lod_state.json')
    assert stage['view_budget_conditioning'] == 2 and abs(stage['view_budget_li_weight']-.5) < 1e-6
    assert stage['view_budget_enabled'] and not stage['view_budget_view_residual']
    assert stage['view_budget_li_full_renders'] == target
    assert stage['view_budget_li_backwards'] <= stage['view_budget_training_backwards'] <= target
    assert stage['total_training_renders'] == sum(stage[n] for n in [
        'full_training_renders', 'lod_training_renders', 'view_budget_training_renders'])
    assert not stage['master_frozen_in_lod'] and not stage['maturity_gate'] and not stage['maturity_loss']
    p.write(out/'li05_verified.json', dict(passed=True, steps=target, li_weight=stage['view_budget_li_weight'],
                                        conditioning=stage['view_budget_conditioning']))
    return ply


def baseline_ready():
    original = p.read(B/'LOD_PIPELINE.json')
    job = next(j for j in original['jobs'] if j['id'] == 'Replica/office4/rgbd')
    if job['state'] == 'complete':
        assert (B/'HOLD').exists() and original['phase'] == 'held_for_li05_controls'
        return True
    controller = p.read(R/'EVALUATION_RESUME_CONTROLLER.json')
    assert p.alive(controller), '补评价控制器已退出，须检查日志，不得跳过'
    return False


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with (R/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p.write(R/'LOD_CONTROLLER.json', p.identity(os.getpid()))
        state = p.read(R/'LOD_PIPELINE.json')
        b.verify_training = verify_li05
        try:
            state['phase'] = 'waiting_for_office4_evaluation'
            p.save(state)
            while not baseline_ready():
                assert not (R/'HOLD').exists(), '用户暂停额外对照'
                time.sleep(15)
            if not state.get('validated_li05'):
                validation = state.get('active_validation') or copy.deepcopy(state['jobs'][0])
                state.update(phase='validating_li05', active_validation=validation)
                p.save(state)
                out = b.train(validation, state, R/'validation/li05', smoke=6)
                assert out is not None, 'LI=0.5六步验收失败，保留现场'
                state.update(validated_li05=True)
                state.pop('active_validation')
                p.write(R/'reports/LI05_SMOKE_PASSED.json', dict(passed=True, steps=6, output=str(out)))
                p.save(state)
            for job in state['jobs']:
                if job['state'] in b.TERMINAL:
                    continue
                assert job['state'] in ['pending', 'retry_pending', 'trained'], '不能覆盖中断现场'
                if job['state'] in ['pending', 'retry_pending']:
                    state.update(phase='training', active_job=job['id'])
                    p.save(state)
                    out = b.train(job, state, R/'runs'/job['id'])
                    if out is None:
                        continue
                    job['evaluations'] = b.cases()
                    p.save(state)
                    print('TRAINED', job['id'], flush=True)
                state.update(phase='evaluating', active_job=job['id'])
                p.save(state)
                for item in job['evaluations']:
                    if item['state'] in ['complete', 'skipped_oom']:
                        continue
                    assert item['state'] in ['pending', 'retry_pending']
                    b.evaluate(job, item, state, R/'evaluation'/job['id']/(item['scope']+'_'+str(item['ratio'])))
                job['state'] = 'complete' if all(e['state'] == 'complete' for e in job['evaluations']) else 'partial_evaluation'
                p.save(state)
                print('COMPLETE', job['id'], job['state'], flush=True)
            before = p.read(R/'PROTECTED_BEFORE.json')
            changed = [name for name, digest in before.items() if p.sha(Path(name)) != digest]
            assert not changed, changed
            images = [str(f) for folder in ['runs', 'evaluation'] for f in (R/folder).rglob('*')
                      if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']]
            assert not images, images[:10]
            state.update(phase='ready_for_comparison', active_job=None)
            p.save(state)
            p.write(R/'reports/CONTROLS_FINISHED.json', dict(
                completed_at=time.time(), counts=state['counts'], protected_files_unchanged=len(before), saved_render_images=0))
            print('READY_FOR_COMPARISON; 原队列仍保持HOLD', flush=True)
        except Exception as error:
            state.update(phase='held' if (R/'HOLD').exists() else 'blocked', error=repr(error))
            p.save(state)
            raise


if __name__ == '__main__':
    main()

"""32组最佳Full的无视锥Fine与全轨迹评价；保留逐步健康检查和OOM边界。"""
from pathlib import Path
import fcntl
import math
import os
import resource
import subprocess
import sys

import pipeline as p

R = Path(__file__).resolve().parent
p.config = lambda: R / 'configs/train_no_frustum.yaml'
original_environment = p.environment


def environment():
    env = original_environment()
    for key in list(env):
        if key.startswith('CARTGS_LOD_DIAGNOSTIC_'):
            env.pop(key)
    env['CARTGS_AUDIT_SEED'] = '0'
    return env


p.environment = environment


def manifest(job, output):
    full = Path(job['full_job']['result']['checkpoint'])
    cloud, = (full / 'point_cloud').glob('iteration_*/point_cloud.ply')
    value = dict(full_ply=str(full), point_cloud=str(cloud),
        anchor_state=str(cloud.parent / 'anchor_selector_state.json'),
        camera_config=job['camera_config'], mapping_config=job['mapping_config'],
        selector_config=str(p.config()), initial_iteration=int(full.parent.name.split('_')[0]),
        output=str(output), mode=job['mode'], dataset=job['dataset'],
        dataset_root=job['full_evaluation']['data'],
        optimizer_protocol='fresh Adam/RNG; not complete SLAM/optimizer resume')
    paths = [cloud, cloud.parent / 'anchor_selector_state.json', cloud.parent / 'anchor_pruning_state.pt',
             full / 'selector.pt', full / 'real_lod_state.json', full / 'cameras.json',
             Path(value['mapping_config']), Path(value['camera_config']), p.config()]
    value['input_sha256'] = {str(path): p.sha(path) for path in paths}
    return value


p.manifest = manifest


def evaluate(job, item, state, root):
    model = Path(job['trained_model'])
    ply, = model.glob('*_shutdown/ply')
    def command(output):
        argv = [sys.executable, str(R / 'eval_entry.py'), str(R / 'source/python/run.py'),
            str(model), job['full_evaluation']['data'], '--shutdown-model', ply.parent.name,
            '--output_path', str(output), '--no-save-images']
        if item['ratio'] != 1:
            argv += ['--selector', str(ply / 'selector.pt'), '--selector-ratio', str(item['ratio']),
                     '--selector-config', str(p.config())]
        if job['mode'] == 'mono':
            argv.append('--correct_scale')
        elif job['mode'] == 'stereo':
            argv += ['--stereo-camera-config', job['camera_config'],
                     '--stereo-native-maps', str(R / 'configs/vector_stereo_native_maps.yml.gz')]
        return argv
    out = p.run_process(command, root, item, state, evaluation=True)
    if out is None:
        return
    result = p.read(out / 'metric_summary.json')
    assert result['evaluated_frames'] == job['full_evaluation']['result']['evaluated_frames']
    assert result['evaluated_frames'] > 0
    assert all(math.isfinite(result[k]) for k in ['psnr', 'ssim', 'lpips', 'ate_rmse'])
    assert not any(path.suffix.lower() in ['.png', '.jpg', '.jpeg'] for path in out.rglob('*'))
    result.update(output=str(out), saved_images=0, evaluation_policy='no_frustum')
    item.update(state='complete', result=result)
    p.save(state)


def verify_protected():
    before = p.read(R / 'PROTECTED_BEFORE.json')
    after = {group: {path: p.sha(Path(path)) for path in hashes}
             for group, hashes in before.items()}
    assert after == before, '原始模型/生产文件发生变化，不能无条件通过'
    p.write(R / 'PROTECTED_AFTER.json', after)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    lock = (R / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p.write(R / 'LOD_CONTROLLER.json', p.identity(os.getpid()))
    assert p.read(R / 'PREPARED.json')['passed']
    state = p.read(R / 'LOD_PIPELINE.json')
    try:
        validated = set(state.get('validated_modes', []))
        for job in state['jobs']:
            if job['state'] in ['complete', 'quality_failed', 'skipped_oom', 'partial_evaluation']:
                continue
            assert job['state'] in ['pending', 'retry_pending', 'trained'], '中断现场须先核验，禁止重复覆盖'
            kind = job['dataset'] + '/' + job['mode']
            if kind not in validated:
                state['phase'] = 'validating_' + kind
                validation = dict(job, state='pending', attempt_history=[], evaluations=[])
                state['active_validation'] = validation
                p.save(state)
                root = R / 'validation' / kind
                out = p.launch_training(validation, state, root, smoke=6)
                assert out is not None, '验收OOM，不能未经验证开始正式训练'
                p.verify_training(out, 6)
                check = subprocess.run([sys.executable, str(R / 'verify_initialization.py'),
                    str(out / 'manifest.json')], env=environment(), text=True, capture_output=True)
                (out / 'initialization_check.log').write_text(check.stdout + check.stderr)
                assert check.returncode == 0, '初始PLY/网络逐位复制校验失败'
                validation['trained_model'] = str(out / 'model')
                for ratio in [1., .01, .03, .7]:
                    item = dict(ratio=ratio, kind='validation', state='pending')
                    validation['evaluations'].append(item)
                    p.save(state)
                    evaluate(validation, item, state, root / ('eval_' + str(ratio)))
                    assert item['state'] == 'complete'
                validated.add(kind)
                state['validated_modes'] = sorted(validated)
                state.setdefault('validation_history', []).append(validation)
                state.pop('active_validation')
                p.save(state)
                print('VALIDATED', kind, flush=True)
            if job['state'] in ['pending', 'retry_pending']:
                state['phase'] = 'training'
                p.save(state)
                try:
                    out = p.launch_training(job, state, R / 'runs' / job['id'])
                except RuntimeError:
                    failure = Path(job['attempt']) / 'model/lod_health_failure.json'
                    if not failure.is_file():
                        raise
                    job.update(state='quality_failed', failure=p.read(failure))
                    p.save(state)
                    print('HEALTH_STOP', job['id'], job['failure'], flush=True)
                    continue
                if out is None:
                    continue
                p.verify_training(out, job['target_steps'])
                job.update(state='trained', trained_model=str(out / 'model'))
                job['evaluations'] = [dict(ratio=ratio, kind='full' if ratio == 1 else
                    'train' if ratio in p.TRAIN else 'unseen', state='pending')
                    for ratio in sorted(p.TRAIN + p.UNSEEN + [1.])]
                p.save(state)
                print('TRAINED', job['id'], job['target_steps'], flush=True)
            assert job['state'] == 'trained'
            state['phase'] = 'evaluating'
            p.save(state)
            for item in job['evaluations']:
                if item['state'] in ['complete', 'skipped_oom']:
                    continue
                assert item['state'] in ['pending', 'retry_pending'], '评价中断现场须先核验'
                evaluate(job, item, state, R / 'evaluation' / job['id'] / ('ratio_' + str(item['ratio'])))
            job['state'] = 'complete' if all(item['state'] == 'complete' for item in job['evaluations']) else 'partial_evaluation'
            p.save(state)
            print('COMPLETE', job['id'], flush=True)
        verify_protected()
        state['phase'] = 'finished'
        p.save(state)
        p.final_report(state)
        p.write(R / 'FINISHED.json', dict(passed=all(j['state'] == 'complete' for j in state['jobs']),
            counts=state['counts'], protected_files_unchanged=True,
            metric_rows=sum(len(j['evaluations']) for j in state['jobs'])))
        print('ALL_DONE', state['counts'], flush=True)
    except Exception as error:
        state.update(phase='blocked', error=repr(error))
        p.save(state)
        raise


if __name__ == '__main__':
    main()

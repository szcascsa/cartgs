"""等待Full批次结束，按场景/模式选最佳Full，独立执行原生Fine和全轨迹评价。"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import time

from oom_retry import classify_oom

R = Path(__file__).resolve().parent
B = R.parent / 'multi_dataset_full_tail_prune1_seed0_20261103'
TRAIN = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7]
UNSEEN = [0.03, 0.075, 0.2, 0.4, 0.6]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def identity(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=fields[19], boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def alive(info):
    try:
        return identity(info['pid']) == {k: info[k] for k in ['pid', 'start_ticks', 'boot_id']}
    except (FileNotFoundError, ProcessLookupError):
        return False


def save(state):
    state['updated_at'] = time.time()
    state['counts'] = dict(Counter(j['state'] for j in state.get('jobs', [])))
    write(R / 'LOD_PIPELINE.json', state)
    rows = []
    for job in state.get('jobs', []):
        base = {k: job[k] for k in ['id', 'dataset', 'scene', 'mode', 'winner', 'full_psnr', 'state']}
        for item in job.get('evaluations', []):
            rows.append(dict(base, ratio=item['ratio'], ratio_kind=item['kind'], evaluation_state=item['state'],
                             **item.get('result', {})))
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        path = R / 'reports/LOD_METRICS.csv'; path.parent.mkdir(exist_ok=True)
        with path.with_suffix('.tmp').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, keys); writer.writeheader(); writer.writerows(rows)
        os.replace(path.with_suffix('.tmp'), path)


def select_winners(training, evaluation):
    lookup = {x['id']: x for x in evaluation['jobs']}
    groups = defaultdict(list)
    for job in training:
        assert job['seed'] == 0 and job['prune'] == 1
        groups[(job['dataset'], job['scene'], job['mode'])].append(job)
    assert len(groups) == 22 and len(training) == 66
    winners = []
    for (dataset, scene, mode), candidates in sorted(groups.items()):
        assert sorted(x['repeat'] for x in candidates) == [1, 2, 3]
        valid = [x for x in candidates if x['state'] == 'complete' and lookup[x['id']]['state'] == 'complete']
        if not valid:
            winners.append(dict(id=f'{dataset}/{scene}/{mode}', dataset=dataset, scene=scene, mode=mode,
                                winner=None, full_psnr=None, state='skipped_no_valid_full', candidates=candidates))
            continue
        assert all(math.isfinite(lookup[x['id']]['result']['trajectory_psnr']) for x in valid)
        best = max(valid, key=lambda x: (lookup[x['id']]['result']['trajectory_psnr'], -x['repeat']))
        winners.append(dict(id=f'{dataset}/{scene}/{mode}', dataset=dataset, scene=scene, mode=mode,
            winner=best['id'], full_psnr=lookup[best['id']]['result']['trajectory_psnr'], state='pending',
            full_job=best, full_evaluation=lookup[best['id']],
            candidates=[dict(id=x['id'], train_state=x['state'], eval_state=lookup[x['id']]['state'],
                             result=lookup[x['id']].get('result')) for x in candidates]))
    return winners


def wait_full(state):
    while True:
        if (R / 'HOLD').exists():
            state['phase'] = 'held'; save(state); return False
        training = read(B / 'QUEUE.json'); evaluation = read(B / 'TRAJECTORY_EVALUATION.json')
        controllers = [read(B / p) for p in ['COORDINATOR.json', 'QUEUE_CONTROLLER.json']]
        eval_pid = B / 'TRAJECTORY_EVALUATION_CONTROLLER.json'
        if eval_pid.exists(): controllers.append(read(eval_pid))
        state['full_counts'] = dict(Counter(x['state'] for x in training))
        state['full_evaluation_counts'] = dict(Counter(x['state'] for x in evaluation['jobs']))
        train_terminal = all(x['state'] in ['complete', 'skipped_oom', 'failed'] for x in training)
        eval_terminal = all(x['state'] in ['complete', 'failed', 'skipped_training_oom', 'blocked_training_failed']
                            for x in evaluation['jobs'])
        if train_terminal and eval_terminal and not any(alive(x) for x in controllers):
            assert all(x['state'] != 'failed' for x in training), 'Full非OOM故障须修复后再选择最佳'
            assert all(x['state'] != 'failed' for x in evaluation['jobs']), 'Full评价故障须修复，不能忽略失败候选'
            return True
        if not any(alive(x) for x in controllers):
            raise RuntimeError('Full批次尚未完成但控制器已退出；需定位，不得跳过')
        state['phase'] = 'waiting_for_full_and_evaluation'; save(state)
        time.sleep(30)


def config():
    text = (B / 'configs/selector_prune1.yaml').read_text()
    text, count = re.subn(r'^Selector.target_ratios:.*$', 'Selector.target_ratios: '+json.dumps(TRAIN), text, flags=re.M)
    assert count == 1
    # Full+LoD保持现有生产Fine目标；不修改其它网络、Loss与优化配置。
    path = R / 'configs/selector_six_ratios.yaml'; path.parent.mkdir(exist_ok=True)
    if path.exists(): assert path.read_text() == text
    else: path.write_text(text)
    return path


def manifest(job, output):
    full = Path(job['full_job']['result']['checkpoint'])
    cloud, = (full / 'point_cloud').glob('iteration_*/point_cloud.ply')
    anchor = cloud.parent / 'anchor_selector_state.json'
    inventory = next(x for x in read(B / 'SCENE_INVENTORY.json') if x['scene'] == job['scene'])
    value = dict(full_ply=str(full), point_cloud=str(cloud), anchor_state=str(anchor),
        camera_config=inventory['modes'][job['mode']]['camera'], mapping_config=inventory['modes'][job['mode']]['mapping'],
        selector_config=str(config()), initial_iteration=int(full.parent.name.split('_')[0]),
        output=str(output), mode=job['mode'], dataset=job['dataset'])
    paths = [cloud, anchor, full / 'selector.pt', full / 'real_lod_state.json', full / 'cameras.json',
             Path(value['mapping_config']), Path(value['camera_config']), Path(value['selector_config'])]
    value['input_sha256'] = {str(p): sha(p) for p in paths}
    value['optimizer_protocol'] = 'fresh Adam; not complete SLAM/optimizer/RNG resume'
    return value


def environment():
    env = dict(os.environ, LD_LIBRARY_PATH=str(R / 'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLBACKEND='Agg', QT_QPA_PLATFORM='offscreen',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    env.pop('LD_PRELOAD', None); env.pop('CARTGS_BATCH_FULL_TAIL_ONLY', None)
    return env


def run_process(argv, directory, item, state, evaluation=False):
    """OOM限额作用于当前工作项；持久化次数，不因重启清零。"""
    for attempt in range(len(item.get('attempt_history', []))+1, 4):
        if (R / 'HOLD').exists(): raise RuntimeError('LoD HOLD存在；保留状态')
        assert shutil.disk_usage(R).free > 5*1024**3, '剩余磁盘不足5GiB，暂停而非继续写满'
        out = directory / ('attempt_%02d'%attempt); out.mkdir(parents=True, exist_ok=False)
        command = argv(out)
        write(out / 'launch.json', dict(command=command))
        start = time.monotonic()
        log_name = 'evaluate.log' if evaluation else 'train.log'
        with (out / log_name).open('w') as stream:
            child = subprocess.Popen(command, cwd=R / 'source', env=environment(), stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            item.update(state='running', process=identity(child.pid), attempt=str(out)); save(state)
            while True:
                try:
                    code = child.wait(timeout=30); break
                except subprocess.TimeoutExpired:
                    item['observed_at'] = time.time()
                    item['log_bytes'] = (out/log_name).stat().st_size
                    trace = out/'model/real_lod_steps.csv'
                    if trace.exists(): item['trace_bytes'] = trace.stat().st_size
                    save(state)
        exit_info = dict(returncode=code, seconds=time.monotonic()-start)
        write(out / 'exit.json', exit_info)
        # 现有OOM分类器读取train.log；评价使用同一只读日志证据。
        if evaluation: (out / 'train.log').symlink_to(log_name)
        oom = classify_oom(dict(output=str(out), process=item['process']), exit_info)
        item.setdefault('attempt_history', []).append(dict(output=str(out), oom=oom, **exit_info)); save(state)
        if code == 0: return out
        if not oom:
            item.update(state='failed', error=f'非OOM退出{code}，见{out/log_name}'); save(state)
            raise RuntimeError(item['error'])
        item['oom_retry_count'] = attempt
        if attempt == 3:
            item.update(state='skipped_oom', error='原始运行及两次重试均OOM，跳过'); save(state); return None
        item['state'] = 'retry_pending'; save(state)
    raise RuntimeError('中断现场需要显式核查，不自动无限重训')


def launch_training(job, state, root, smoke=0):
    def command(attempt):
        model = attempt / 'model'; value = manifest(job, model)
        write(attempt / 'manifest.json', value)
        # 离线评价沿用所选Full的最终估计轨迹与耗时，不重新跑Tracking。
        model.mkdir()
        original_model = Path(job['full_evaluation']['model'])
        for name in ['CameraTrajectory_TUM.txt', 'TrackingTime.txt']:
            shutil.copy2(original_model / name, model / name)
        return [str(R / 'runtime/bin/lod_from_full'), str(attempt / 'manifest.json')] + ([str(smoke)] if smoke else [])
    return run_process(command, root, job, state)


def verify_training(out, target=None):
    model = out / 'model'; result = read(model / 'training_verified.json')
    assert result['passed'] and result['topology_unchanged'] and result['master_xyz_changed']
    if target is not None: assert result['steps'] == target
    ply, = model.glob('*_shutdown/ply')
    stage = read(ply / 'real_lod_state.json')
    n = result['steps']
    for key in ['lod_steps', 'master_updates', 'selector_updates', 'full_training_renders',
                'full_training_backwards', 'lod_training_renders', 'lod_training_backwards', 'optimizer_commits']:
        assert stage[key] == n, (key, stage[key], n)
    assert stage['lod_nonfinite'] == 0 and sum(stage['ratio_steps']) == n
    assert max(stage['ratio_steps'])-min(stage['ratio_steps']) <= 1
    assert len(stage['ratios']) == 6
    assert all(abs(a-b)<1e-6 for a,b in zip(stage['ratios'], TRAIN))
    assert all(sha(Path(p)) == h for p,h in read(out/'manifest.json')['input_sha256'].items())
    assert not any(p.suffix.lower() in ['.png','.jpg','.jpeg'] for p in model.rglob('*'))
    write(out/'verified.json', dict(passed=True, **stage))
    return ply


def evaluate(job, item, state, root):
    model = Path(job['trained_model']); ply, = model.glob('*_shutdown/ply')
    def command(out):
        argv = [sys.executable, str(R/'lod_eval_entry.py'), str(R/'source/python/run.py'), str(model),
            job['full_evaluation']['data'], '--shutdown-model', ply.parent.name,
            '--output_path', str(out), '--no-save-images']
        if item['ratio'] != 1.0:
            argv += ['--selector', str(ply/'selector.pt'), '--selector-ratio', str(item['ratio']),
                     '--selector-config', str(config())]
        if job['mode']=='mono': argv.append('--correct_scale')
        return argv
    out = run_process(command, root, item, state, evaluation=True)
    if out is None: return
    result = read(out/'metric_summary.json')
    assert result['evaluated_frames'] == job['full_evaluation']['result']['evaluated_frames']
    assert not any(p.suffix.lower() in ['.png','.jpg','.jpeg'] for p in out.rglob('*'))
    assert all(math.isfinite(result[k]) for k in ['psnr','ssim','lpips','ate_rmse'])
    result.update(output=str(out), saved_images=0)
    item.update(state='complete', result=result); save(state)


def final_report(state):
    lines=['# 最佳Full模型LoD实验结果', '',
        '每个数据集/场景/模式从seed=0三次运行中按Full全轨迹平均PSNR选一次，其他指标不另选。', '',
        '训练ratio：'+str(TRAIN)+'；未见ratio：'+str(UNSEEN)+'；另评Fine后的Full。', '',
        'Fine复用原生Full+轮转LoD，每步累积后一次Adam；Master可更新，拓扑固定。',
        'Adam与RNG重新初始化；这不是完整在线SLAM恢复。XYZ学习率尺度由最终保存关键帧中心半径重建。', '',
        '| 数据集/场景/模式 | 选中Full运行 | Full起点PSNR | 状态 |',
        '|---|---|---:|---|']
    for job in state['jobs']:
        lines.append(f"| {job['id']} | {job['winner']} | {job['full_psnr']} | {job['state']} |")
    lines+=['', '逐ratio PSNR/SSIM/LPIPS、ATE、实际K与模型规模见LOD_METRICS.csv；逐帧数据和选点身份保存在evaluation下。',
            '未见ratio只评价不训练；不保存渲染或GT图片。任何失败、OOM及重试保留原日志。',
            'Full的最高PSNR用于选择，LoD指标不用于回选；三次轨迹覆盖可能不同，不将最高均值视为等共同帧的统计检验。']
    (R/'reports/BEST_FULL_LOD_REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    lock = (R/'pipeline.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    resource.setrlimit(resource.RLIMIT_CORE, (0,0))
    state = read(R/'LOD_PIPELINE.json') if (R/'LOD_PIPELINE.json').exists() else dict(jobs=[])
    write(R/'LOD_CONTROLLER.json', identity(os.getpid()))
    try:
        assert read(R/'BUILD_READY.json')['passed']
        if not wait_full(state): return
        if not state['jobs']:
            state['jobs'] = select_winners(read(B/'QUEUE.json'), read(B/'TRAJECTORY_EVALUATION.json'))
            write(R/'SELECTED_BEST_FULL.json', state['jobs']); save(state)
        # 每种入口至少一次6步原生训练/权重转换/首尾ratio与插值评价验收。
        validated = set(state.get('validated_modes', []))
        for job in state['jobs']:
            if job['state'] in ['complete','skipped_no_valid_full','skipped_oom']: continue
            kind = job['dataset']+'/'+job['mode']
            if kind not in validated:
                state['phase']='validating_'+kind; save(state)
                validation = dict(job, state='pending', attempt_history=[])
                state['active_validation']=validation; save(state)
                revision = sha(R/'runtime/bin/lod_from_full')[:12]
                validation_root = R/'validation'/kind/('revision_'+revision)
                out = launch_training(validation, state, validation_root, smoke=6)
                assert out is not None, '验证OOM后暂停，不能未经验证启动正式任务'
                verify_training(out, 6)
                check = subprocess.run([sys.executable, str(R/'verify_initialization.py'), str(out/'manifest.json')],
                                       env=environment(), text=True, capture_output=True)
                (out/'initialization_check.log').write_text(check.stdout+check.stderr)
                assert check.returncode==0, '初始模型转换校验失败'
                validation['trained_model']=str(out/'model')
                for ratio in [1.0, 0.01, 0.03, 0.7]:
                    item=dict(ratio=ratio,kind='validation',state='pending')
                    validation.setdefault('evaluations',[]).append(item); save(state)
                    evaluate(validation,item,state,validation_root/('eval_'+str(ratio)))
                    assert item['state']=='complete', '验证评价未通过'
                validated.add(kind); state['validated_modes']=sorted(validated)
                state.pop('active_validation',None); save(state)
            if job['state'] in ['pending','retry_pending']:
                state['phase']='training'; save(state)
                out = launch_training(job,state,R/'runs'/job['id'])
                if out is None: continue
                verify_training(out,job['full_job']['result']['online_steps'])
                job.update(state='trained', trained_model=str(out/'model'))
                job['evaluations']=[dict(ratio=r,kind='full' if r==1 else 'train' if r in TRAIN else 'unseen',state='pending')
                                    for r in sorted(TRAIN+UNSEEN)+[1.0]]; save(state)
            assert job['state']=='trained', '中断或失败现场需要核查，不能直接覆盖重训'
            state['phase']='evaluating'; save(state)
            for item in job['evaluations']:
                if item['state']=='complete' or item['state']=='skipped_oom': continue
                assert item['state'] in ['pending','retry_pending'], '评价中断需核查现场'
                evaluate(job,item,state,R/'evaluation'/job['id']/('ratio_'+str(item['ratio'])))
            job['state']='complete' if all(x['state']=='complete' for x in job['evaluations']) else 'partial_evaluation'
            save(state)
        state['phase']='finished'; save(state); final_report(state)
    except Exception as error:
        state.update(phase='blocked', error=repr(error)); save(state); raise


if __name__=='__main__': main()

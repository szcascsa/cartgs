"""以CaRtGS未校正原图重评双目；保留旧现场，全部验证后更新批次汇总。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
BATCH = ROOT.parent/'dual_view_li_vector_best_full_20261107'
BACKUP = ROOT/'before_original_rgb'
METRICS = ['psnr', 'ssim', 'lpips']


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def environment():
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLBACKEND='Agg',
               QT_QPA_PLATFORM='offscreen', PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', CARTGS_AUDIT_SEED='0')
    for key in list(env):
        if key.startswith('CARTGS_LOD_DIAGNOSTIC_') or key in [
                'GEOMETRY_HEAD_VARIANT', 'LD_PRELOAD', 'CARTGS_BATCH_FULL_TAIL_ONLY']:
            env.pop(key)
    env['LD_LIBRARY_PATH'] = ':'.join([
        str(BATCH/'runtime/lib'),
        '/home/prmi/zwj/experiments/real_lod_full_joint_20261103/source/third_party/ORB-SLAM3/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib',
        '/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib',
        '/usr/local/cuda-11.8/lib64'])
    return env


def evaluate(job, item):
    import numpy as np

    old = Path(item['result']['output'])
    out = BATCH/'evaluation_original_rgb_20261108'/job['id']/old.parent.name/'attempt_01'
    receipt = out/'verified.json'
    if receipt.exists():
        return read(receipt)
    assert shutil.disk_usage(ROOT).free > 5*1024**3
    command = read(old/'launch.json')['command']
    command = [str(out) if arg == str(old) else arg for arg in command]
    for flag in ['--stereo-camera-config', '--stereo-native-maps']:
        index = command.index(flag)
        del command[index:index+2]
    assert '--no-save-images' in command and '--correct_scale' not in command
    if out.exists():
        assert read(out/'exit.json')['returncode'] == 0, f'未完成现场需核查：{out}'
        assert read(out/'launch.json')['command'] == command
        seconds = read(out/'exit.json')['seconds']
    else:
        out.mkdir(parents=True)
        write(out/'launch.json', dict(command=command, started_unix=time.time(),
                                     changed_only='移除双目GT图像校正参数并改用独立输出目录',
                                     previous_evaluation=str(old)))
        started = time.monotonic()
        with (out/'evaluate.log').open('w') as stream:
            process = subprocess.Popen(command, cwd=BATCH/'source', env=environment(),
                                       stdout=stream, stderr=subprocess.STDOUT)
            write(out/'process.json', dict(pid=process.pid,
                  start_ticks=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19],
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
            returncode = process.wait()
        seconds = time.monotonic()-started
        write(out/'exit.json', dict(returncode=returncode, seconds=seconds))
        assert returncode == 0, f'评价失败：{out}/evaluate.log'
    summary = read(out/'metric_summary.json')
    previous = item['result']
    assert read(out/'image_preprocessing.json') == {'protocol': 'original_rgb'}
    assert summary['evaluated_frames'] == previous['evaluated_frames']
    for key in ['gaussian_count', 'actual_K_mean', 'actual_K_min', 'actual_K_max', 'target_K']:
        assert summary[key] == previous[key], (job['id'], item['scope'], key)
    with (out/'per_frame_metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == previous['evaluated_frames']
    with (old/'per_frame_metrics.csv').open() as stream:
        old_rows = list(csv.DictReader(stream))
    for row, before in zip(rows, old_rows):
        for key in ['synchronized_trajectory_index', 'rgb_index', 'trajectory_timestamp', 'rgb_timestamp', 'rgb_path']:
            assert row[key] == before[key], (job['id'], key)
    for name in METRICS:
        assert math.isfinite(summary[name])
        assert abs(sum(float(row[name]) for row in rows)/len(rows)-summary[name]) < 1e-9
    selected_identical = None
    if item['scope'] == 'global':
        before = np.load(old/'selector_selected_ids.npy')
        after = np.load(out/'selector_selected_ids.npy')
        assert np.array_equal(before, after)
        selected_identical = True
    elif item['scope'] == 'view_ratio':
        assert read(old/'selection_hashes.json') == read(out/'selection_hashes.json')
        selected_identical = True
    assert not any(p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp'] for p in out.rglob('*'))
    summary.update(output=str(out), scope=item['scope'], anchor_count=previous.get('anchor_count'),
                   evaluation_protocol='cartgs_original_rgb_full_trajectory')
    write(out/'metric_summary.json', summary)
    result = dict(id=job['id'], scope=item['scope'], ratio=item['ratio'], result=summary,
                  previous_output=str(old), previous_result=previous,
                  selected_identities_unchanged=selected_identical, seconds=seconds)
    write(receipt, result)
    print('COMPLETE', job['id'], item['scope'], item['ratio'],
          f"PSNR {previous['psnr']:.5f} -> {summary['psnr']:.5f}; {seconds:.1f}s", flush=True)
    return result


def evaluate_scene(job):
    lock = (ROOT/(job['id'].replace('/', '__')+'.lock')).open('w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        items = sorted(job['evaluations'], key=lambda item: (item['scope'] != 'full',
                                                            item['scope'], item['ratio']))
        return [evaluate(job, item) for item in items]
    finally:
        lock.close()


def main():
    lock = (ROOT/'original_rgb_reevaluation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT/'REEVALUATION_COMPLETE.json').exists():
        print('ALREADY_COMPLETE', flush=True)
        return
    state_path = BATCH/'LOD_PIPELINE.json'
    originals = [state_path, BATCH/'reports/LOD_METRICS.csv',
                 BATCH/'reports/DUAL_VIEW_LI_BATCH_REPORT.md']
    if not BACKUP.exists():
        BACKUP.mkdir()
        for path in originals:
            shutil.copy2(path, BACKUP/path.name)
    state = read(BACKUP/'LOD_PIPELINE.json')
    assert all(j['state'] == 'complete' for j in state['jobs'])
    jobs = [j for j in state['jobs'] if j['mode'] == 'stereo']
    assert len(jobs) == 3
    protected_paths = set()
    for job in jobs:
        protected_paths.update(p for p in Path(job['trained_model']).rglob('*') if p.is_file())
        for item in job['evaluations']:
            old = Path(item['result']['output'])
            protected_paths.update(old/name for name in ['metric_summary.json', 'per_frame_metrics.csv',
                                                        'selector_stats.json', 'evaluation_frames.json'])
    before_path = ROOT/'data/REEVALUATION_PROTECTED_BEFORE.json'
    if not before_path.exists():
        write(before_path, {str(path): sha(path) for path in sorted(protected_paths)})
    # 同场景串行并以文件锁防重；corridors沿用原29GiB内存保护。
    small = [job for job in jobs if job['scene'] != 'corridors-dolly']
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = sum(list(pool.map(evaluate_scene, small)), [])
    records += evaluate_scene(next(job for job in jobs if job['scene'] == 'corridors-dolly'))
    assert len(records) == 69
    protected = read(before_path)
    after = {path: sha(Path(path)) for path in protected}
    assert protected == after
    write(ROOT/'data/REEVALUATION_PROTECTED_AFTER.json', after)
    # 更新前确认原队列汇总仍是备份版本，不覆盖并行写入。
    for path in originals:
        assert sha(path) == sha(BACKUP/path.name), f'并行更新冲突：{path}'
    updated = copy.deepcopy(state)
    lookup = {(r['id'], r['scope'], r['ratio']): r for r in records}
    for job in updated['jobs']:
        job['full_psnr_protocol'] = ('historical_selection_rectified_left' if job['mode'] == 'stereo'
                                     else 'historical_selection_original_rgb')
        for item in job['evaluations']:
            record = lookup.get((job['id'], item['scope'], item['ratio']))
            if record:
                item['previous_rectified_result'] = item['result']
                item['previous_rectified_attempt'] = item['attempt']
                item['result'] = record['result']
                item['attempt'] = record['result']['output']
                item['process'] = read(Path(item['attempt'])/'process.json')
                item['attempt_history'].append(dict(output=item['attempt'], process=item['process'],
                        returncode=0, seconds=record['seconds'],
                        evaluation_protocol='cartgs_original_rgb_full_trajectory'))
            else:
                item['result']['evaluation_protocol'] = 'cartgs_original_rgb_full_trajectory'
    updated['metric_revision'] = dict(protocol='cartgs_original_rgb_full_trajectory',
          changed_records=69, unchanged_records=69, changed_at=time.time(), backup=str(BACKUP),
          note='只重评已训练模型；Full起点选择及历史起点PSNR仍保留原选择时口径，不重新回选模型')
    rows = []
    for job in updated['jobs']:
        base = {key: job[key] for key in ['id', 'dataset', 'scene', 'mode', 'winner',
                                         'full_psnr', 'full_psnr_protocol', 'state']}
        for item in job['evaluations']:
            rows.append(dict(base, ratio=item['ratio'], ratio_kind=item['kind'],
                             evaluation_state=item['state'], **item['result']))
    keys = list(dict.fromkeys(key for row in rows for key in row))
    csv_path = BATCH/'reports/LOD_METRICS.csv'
    with csv_path.with_suffix('.tmp').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)
    write(state_path, updated)
    csv_path.with_suffix('.tmp').replace(csv_path)
    report = BACKUP/'DUAL_VIEW_LI_BATCH_REPORT.md'
    text = report.read_text()+(
        '\n## 2026-11-08 评价口径更新\n\n'
        'VECtor三个双目场景的69条最终Full／全局LoD／局部LoD结果，已按用户要求改用CaRtGS未校正原图重新评价。'
        '训练checkpoint、评价轨迹／RGB关联、选点身份与实际K均未改变；不保存渲染图片。\n\n'
        'LOD_PIPELINE.json及LOD_METRICS.csv已更新为新指标。旧校正域评价目录完整保留；'
        f'汇总备份：{BACKUP}。新逐帧结果：evaluation_original_rgb_20261108。\n\n'
        '上表Full起点PSNR属于原选模记录，仍是选择当时的口径；不能当作本次未校正原图Full终点成绩。'
        '当前最终Full以LOD_METRICS.csv中scope=full的行作为依据。\n')
    (BATCH/'reports/DUAL_VIEW_LI_BATCH_REPORT.md').write_text(text)
    write(ROOT/'data/ORIGINAL_RGB_REEVALUATION.json', records)
    receipt = dict(complete=True, reevaluated_records=69, selected_identities_unchanged=True,
                   frames_and_K_unchanged=True, protected_files=len(protected),
                   protected_hashes_unchanged=True, backup=str(BACKUP),
                   state_sha256=sha(state_path), csv_sha256=sha(csv_path), saved_images=0)
    write(ROOT/'REEVALUATION_COMPLETE.json', receipt)
    print('REEVALUATION_COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

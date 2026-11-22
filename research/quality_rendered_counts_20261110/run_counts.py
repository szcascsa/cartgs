"""按场景补测冻结渲染的点数；只读历史实验，可恢复已完成场景。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'all_scene_quality_figures_20261108/data/ALL_SCENE_RESULTS.json'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def environment(batch):
    e = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLBACKEND='Agg',
             QT_QPA_PLATFORM='offscreen', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    for name in list(e):
        if name.startswith('CARTGS_LOD_DIAGNOSTIC_') or name in ['GEOMETRY_HEAD_VARIANT', 'LD_PRELOAD', 'CARTGS_BATCH_FULL_TAIL_ONLY']:
            e.pop(name)
    e['LD_LIBRARY_PATH'] = ':'.join([str(batch/'runtime/lib'),
        '/home/prmi/zwj/experiments/real_lod_full_joint_20261103/source/third_party/ORB-SLAM3/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib',
        '/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib', '/usr/local/cuda-11.8/lib64'])
    return e


def run_group(group):
    output = ROOT/'counts'/group['id']
    if (output/'COMPLETE.json').exists():
        return read(output/'COMPLETE.json')
    assert shutil.disk_usage(ROOT).free > 5*1024**3
    available = int(next(line for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:')).split()[1])*1024
    assert available > 4*1024**3, '系统可用内存不足4GiB'
    batch = Path(group['trained_model'].split('/runs/')[0])
    output.mkdir(parents=True, exist_ok=True)
    command = ['/home/prmi/anaconda3/envs/zwj/bin/python', '-u', str(ROOT/'count_worker.py'), group['id']]
    if group['id'] == 'VECtor/corridors-dolly/stereo':
        command = command[:1]+[str(batch/'memory_guard.py'), str(batch/'configs/corridors_stereo_memory.json'), str(output)]+command
    started = time.monotonic()
    with (output/'worker.log').open('a') as log:
        p = subprocess.Popen(command, cwd=ROOT, env=environment(batch), stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(output/'PROCESS.json', dict(pid=p.pid,
            start_ticks=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')', 1)[1].split()[19],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), command=command))
        code = p.wait()
    write(output/'EXIT.json', dict(returncode=code, seconds=time.monotonic()-started))
    assert code == 0, f"场景计数失败：{group['id']}，检查{output}/worker.log"
    records = read(output/'COMPLETE.json')
    assert len(records) == 24
    print('GROUP_COMPLETE', group['id'], 'SECONDS', time.monotonic()-started, flush=True)
    return records


def main():
    lock = (ROOT/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = read(SOURCE)
    before_path = ROOT/'data/PROTECTED_BEFORE.json'
    if not before_path.exists():
        paths = {SOURCE}
        for group in source['groups']:
            checkpoints = [Path(group['vanilla']['checkpoint']), *Path(group['trained_model']).glob('*_shutdown')]
            for checkpoint in checkpoints:
                paths.update(p for p in checkpoint.rglob('*') if p.is_file())
            for item in group['ours']:
                paths.update(Path(item['output'])/name for name in ['metric_summary.json', 'per_frame_metrics.csv'])
            batch = Path(group['trained_model'].split('/runs/')[0])
            paths.update([batch/'LOD_PIPELINE.json', batch/'reports/LOD_METRICS.csv'])
        write(before_path, {str(path):sha(path) for path in sorted(paths)})
    started = time.monotonic()
    # 先验证32组中规模较小的场景；每次两个只读计数进程，不运行训练。
    groups = sorted(source['groups'], key=lambda g:(g['id']=='VECtor/corridors-dolly/stereo',
                                                   g['ours'][0]['frames'], g['id']))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = sum(list(pool.map(run_group, groups)), [])
    assert len(results) == 768
    before = read(before_path)
    after = {name:sha(Path(name)) for name in before}
    assert before == after
    write(ROOT/'data/PROTECTED_AFTER.json', after)
    value = dict(source=str(SOURCE), source_sha256=sha(SOURCE), groups=source['groups'],
                 counts=results, elapsed_seconds=time.monotonic()-started,
                 total_rendered_views=sum(row['frames'] for row in results),
                 source_hashes_unchanged=True, images_saved=0, training_steps=0)
    write(ROOT/'data/QUALITY_RENDERED_COUNTS.json', value)
    write(ROOT/'COMPLETE.json', dict(complete=True, groups=len(groups), records=len(results),
          total_rendered_views=value['total_rendered_views'], protected_files=len(before),
          source_hashes_unchanged=True, elapsed_seconds=value['elapsed_seconds']))
    print('ALL_COMPLETE', json.dumps(read(ROOT/'COMPLETE.json')), flush=True)


if __name__ == '__main__':
    main()

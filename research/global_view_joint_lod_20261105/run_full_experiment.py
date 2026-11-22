"""串行运行 fr1_xyz 三组完整 Fine，保留失败现场和实际进程身份。"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HKT = datetime.timezone(datetime.timedelta(hours=8))
status_path = ROOT/'reports/FULL_EXPERIMENT_STATUS.json'


def save(value):
    value['checked_hkt'] = datetime.datetime.now(HKT).isoformat()
    temp = status_path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(status_path)


def identity(pid):
    stat = Path(f'/proc/{pid}/stat').read_text().split()
    return {'pid': pid, 'start_ticks': stat[21], 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def progress(folder):
    path = folder/'real_lod_steps.csv'
    if not path.exists():
        return 0
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-2048))
        for line in reversed(stream.read().decode().splitlines()):
            if line and line.split(',')[0].isdigit():
                return int(line.split(',')[0])
    return 0


def main():
    lock = (ROOT/'reports/full_experiment.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if status_path.exists():
        raise RuntimeError('已有运行记录，必须核查旧进程和结果后恢复；不自动覆盖')
    assert shutil.disk_usage(ROOT).free > 5*2**30
    hashes = json.loads((ROOT/'reports/SOURCE_HASHES.json').read_text())
    for name, sha in hashes.items():
        assert hashlib.sha256((ROOT/'source'/name).read_bytes()).hexdigest() == sha, name
    state = {'state': 'running', 'controller': identity(os.getpid()), 'groups': {}, 'seed': 0,
             'training_ratios': [.01, .05, .1, .3, .5, .7], 'fine_steps': 11580,
             'full_checkpoint': 'fr1_xyz RGB-D seed0_repeat2 11980_shutdown; fresh Adam/RNG',
             'evaluation': 'pending; full RGB trajectory, global export and nominal-frustum strict local budgets'}
    save(state)
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = ':'.join([
        str(ROOT/'runtime/lib'),
        '/home/prmi/zwj/experiments/real_lod_full_joint_20261103/source/third_party/ORB-SLAM3/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib',
        '/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib', '/usr/local/cuda-11.8/lib64'])
    env['OMP_NUM_THREADS'] = '2'
    for name in ['baseline', 'joint', 'global_control']:
        manifest = ROOT/'configs'/f'manifest_{name}_full.json'
        value = json.loads(manifest.read_text())
        output = Path(value['output'])
        assert not output.exists(), f'保留已有现场：{output}'
        for key, sha in value['input_sha256'].items():
            assert hashlib.sha256(Path(value[key]).read_bytes()).hexdigest() == sha, key
        assert shutil.disk_usage(ROOT).free > 5*2**30
        command = [str(ROOT/'runtime/bin/lod_from_full'), str(manifest)]
        started = time.monotonic()
        with (ROOT/'logs'/f'full_{name}.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            row = {'state': 'running', 'process': identity(child.pid), 'command': command,
                   'output': str(output), 'started_hkt': datetime.datetime.now(HKT).isoformat(), 'step': 0}
            state['groups'][name] = row; state['active_group'] = name; save(state)
            while child.poll() is None:
                row['step'] = progress(output)
                row['seconds'] = time.monotonic()-started
                save(state)
                time.sleep(5)
            row['returncode'] = child.returncode; row['seconds'] = time.monotonic()-started
        if child.returncode:
            row['state'] = 'failed'; state['state'] = 'failed'; save(state)
            raise RuntimeError(f'{name} 失败，先定位 logs/full_{name}.log；不自动重训或改参数')
        verified = json.loads((output/'training_verified.json').read_text())
        assert verified['passed'] and verified['steps'] == 11580 and verified['topology_unchanged']
        checkpoint = output/'23560_shutdown/ply'
        fine = json.loads((checkpoint/'real_lod_state.json').read_text())
        assert fine['optimizer_commits'] == fine['master_updates'] == fine['selector_updates'] == 11580
        assert fine['view_budget_training_renders'] == (0 if name == 'baseline' else 11580)
        # Fine does not change cameras. Reuse the exact input trajectory for full-frame evaluation.
        trajectory = Path(value['full_ply']).parent.parent/'CameraTrajectory_TUM.txt'
        shutil.copy2(trajectory, output/'CameraTrajectory_TUM.txt')
        row.update(state='complete', step=11580, checkpoint=str(checkpoint), verified=verified)
        save(state)
        print('TRAINING_COMPLETE', name, round(row['seconds'], 2), flush=True)
    state['state'] = 'training_complete'; state['active_group'] = None; save(state)


if __name__ == '__main__':
    main()

"""预注册的 equal-work / equal-time / 5+5 online，串行独占 GPU。"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['equal_work', 'equal_time', 'statistical', 'all'], default='all')
args = parser.parse_args()
dest = Path(__file__).resolve().parent
root = dest.parents[2]
production = Path('/home/prmi/zwj/mycartgs')
protocol = json.loads((dest/'protocol.json').read_text())
while not (dest/'fixed_map.exit').exists():
    time.sleep(2)
assert (dest/'fixed_map.exit').read_text().strip() == '0', 'Previous phase did not pass'
output = root/'results/selector_revised_acceptance_20261015'
output.mkdir(exist_ok=True)
python = '/home/prmi/anaconda3/envs/zwj/bin/python'
binary = root/'bin/tum_rgbd_selector_validation'
env = os.environ.copy()
env.update(CARTGS_EXPERIMENTAL_SELECTOR_CUDA='1', CUDA_HOME='/usr/local/cuda-11.8',
    CXX='/usr/bin/g++-9', MAX_JOBS='2', TORCH_CUDA_ARCH_LIST='8.6',
    TORCH_EXTENSIONS_DIR=str(root/'python_extensions'), MPLBACKEND='Agg', TORCH_HOME='/home/prmi/.cache/torch')
env['PATH'] = '/home/prmi/anaconda3/envs/zwj/bin:/home/prmi/.local/bin:/usr/local/cuda-11.8/bin:'+env.get('PATH', '')
env['LD_LIBRARY_PATH'] = str(root/'lib')+':'+str(production/'third_party/libtorch-cu118/libtorch/lib')+':/usr/local/cuda-11.8/lib64:'+env.get('LD_LIBRARY_PATH', '')
evaluation = root/'python/selector_revised_eval.py'
source = (root/'python/run.py').read_text().replace('from evo.tools import plot',
    'from evo.tools import settings\nsettings.SETTINGS["plot_backend"] = "Agg"\nfrom evo.tools import plot')
source = source.replace('import cv2\n', 'import cv2\ncv2.imwrite = lambda *args, **kwargs: True\n', 1)
evaluation.write_text(source)
config = root/'results/selector_online_ab_final_20261015/source_fast.yaml'
base = config.read_text()
assert re.search(r'Training.force_tail_optimization_enabled:\s*0', base)
assert re.search(r'GradientConstraint.enable:\s*0', base)
scenes = {
    'room': ('rgbd_dataset_freiburg1_room', 'tum_freiburg1_desk.yaml', 'fr1_room.txt'),
    'desk': ('rgbd_dataset_freiburg1_desk', 'tum_freiburg1_desk.yaml', 'tum_freiburg1_desk.txt'),
    'xyz': ('rgbd_dataset_freiburg2_xyz', 'tum_freiburg2_xyz.yaml', 'tum_freiburg2_xyz.txt'),
    'long': ('rgbd_dataset_freiburg3_long_office_household', 'tum_freiburg3_long_office_household.yaml', 'tum_freiburg3_long_office_household.txt'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


status_path = output/'status.json'
status = json.loads(status_path.read_text()) if status_path.exists() else {'runs': []}
identity = {str(p.relative_to(root)): sha(p) for p in [binary, root/'lib/libgaussian_mapper.so',
    root/'src/selector_cuda.cu', root/'src/selector_cuda.cpp', root/'include/selector_cuda.h', config, dest/'protocol.json']}
assert not status.get('identity') or identity == status['identity'], 'Frozen cohort changed'
status.update(identity=identity, protocol=protocol)


def save():
    status_path.write_text(json.dumps(status, indent=2)+'\n')


def execute(command, path, run_env):
    start = time.monotonic()
    with path.open('w') as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=root, env=run_env, stdout=stream, stderr=subprocess.STDOUT)
        status['active_pid'] = process.pid
        save()
        try:
            code = process.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
    return dict(exit=code, seconds=time.monotonic()-start, pid=process.pid,
                command=list(map(str, command)))


phases = ['equal_work', 'equal_time', 'statistical'] if args.phase == 'all' else [args.phase]
try:
    for phase in phases:
        for scene, (dataset, slam, association) in scenes.items():
            seeds = protocol['algorithm_seeds'] if phase == 'statistical' else [0]
            for index, seed in enumerate(seeds):
                for backend in (['legacy', 'cuda'] if index % 2 == 0 else ['cuda', 'legacy']):
                    name = f'{phase}/{scene}/seed{seed}/{backend}'
                    previous = next((x for x in status['runs'] if x['name'] == name), None)
                    if previous and previous.get('complete'):
                        continue
                    if shutil.disk_usage(root).free < 1024**3:
                        raise RuntimeError('Less than 1GiB free; preserve evidence and stop')
                    run = output/name
                    run.mkdir(parents=True, exist_ok=True)
                    item = previous or dict(name=name, phase=phase, scene=scene, backend=backend,
                        algorithm_seed=seed, run_id=name.replace('/', '_'), start_utc=datetime.now(timezone.utc).isoformat(),
                        initial_state='Fresh SLAM + mapper; constructor seed recorded; ORB internal RNG unchanged')
                    if previous is None:
                        status['runs'].append(item)
                    status.update(active=name, complete=False)
                    text = re.sub(r'^Selector.backend:.*\n|^Selector.max_mapping_iterations:.*\n', '', base, flags=re.M)
                    target = protocol['scenes'][scene]['committed_updates'] if phase == 'equal_work' else 0
                    text += f'\nSelector.backend: "{backend}"\nSelector.max_mapping_iterations: {target}\n'
                    run_config = run/'selector.yaml'
                    run_config.write_text(text)
                    item['config_sha256'] = sha(run_config)
                    run_env = dict(env, SELECTOR_VALIDATION_SEED=str(seed), SELECTOR_VALIDATION_MAPPING_SECONDS=str(
                        protocol['scenes'][scene]['mapping_work_seconds'] if phase == 'equal_time' else 0))
                    item['environment_controls'] = {k: run_env[k] for k in ['SELECTOR_VALIDATION_SEED', 'SELECTOR_VALIDATION_MAPPING_SECONDS']}
                    save()
                    if not item.get('train') or item['train']['exit']:
                        command = [binary, production/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',
                            root/'cfg/ORB_SLAM3/RGB-D/TUM'/slam, root/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml',
                            '/home/prmi/datasets/TUM_RGBD/'+dataset,
                            root/'cfg/ORB_SLAM3/RGB-D/TUM/associations'/association, run, 'no_viewer', '--selector-config', run_config]
                        item['train'] = execute(command, run/'train.log', run_env)
                        save()
                        assert item['train']['exit'] == 0, 'Training failed: '+name
                    log = (run/'train.log').read_text(errors='replace')
                    lines = re.findall(r'^\[SELECTOR REVISED\] (.*)', log, re.M)
                    assert len(lines) == 1, 'Missing test driver report'
                    item['counters'] = dict(re.findall(r'(\w+)=([^ ]+)', lines[0]))
                    if phase == 'equal_work':
                        assert int(item['counters']['mapping_cycles']) == target, 'Mapping cycle target not reached'
                        assert int(item['counters']['committed_updates']) == target, 'Committed target not reached'
                    if phase == 'equal_time':
                        assert item['counters']['time_budget_reached'] == '1', 'Time budget not reached before shutdown'
                    checkpoints = list(run.glob('*_shutdown/ply'))
                    assert len(checkpoints) == 1
                    checkpoint = checkpoints[0]
                    item['transform_state'] = json.loads((checkpoint/'transform_state.json').read_text())
                    stage_file = checkpoint/'mapping_stage_state.json'
                    assert stage_file.exists(), 'Actual Gaussian commit counter is required'
                    item['mapping_state'] = json.loads(stage_file.read_text())
                    actual_commits = item['mapping_state']['online']['gaussian_updates']
                    item['actual_gaussian_commits'] = actual_commits
                    if phase == 'equal_work':
                        assert actual_commits == target, f'Actual Gaussian commits {actual_commits} != {target}'
                    item['checkpoint'] = str(checkpoint)
                    if not item.get('evaluation') or item['evaluation']['exit']:
                        item['evaluation'] = execute([python, root/'python/eval_lod.py', run,
                            '/home/prmi/datasets/TUM_RGBD/'+dataset, '--output', run/'evaluation', '--run-script', evaluation],
                            run/'evaluation.log', env)
                        save()
                        assert item['evaluation']['exit'] == 0, 'Evaluation failed: '+name
                    files = list((run/'evaluation').rglob('eval.txt'))
                    assert len(files) == 5
                    item['metrics'] = {str(p.relative_to(run)): p.read_text() for p in files}
                    with (run/'evaluation/summary.csv').open() as stream:
                        item['evaluation_summary'] = list(csv.DictReader(stream))
                    item['trajectory_metrics'] = {str(p.relative_to(run)): p.read_text() for p in (run/'evaluation').rglob('metrics_traj.txt')}
                    item['complete'] = True
                    save()
                    print(name+' complete', flush=True)
    status.update(active=None, active_pid=None, complete=args.phase == 'all')
    save()
except Exception as error:
    status['error'] = repr(error)
    save()
    raise

"""当前分支 online A/B：同一 binary/config，先测 legacy 三次自身波动。"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

dest = Path(__file__).resolve().parent
root = dest.parents[2]
while not (dest/'optimizer_candidate_complete.txt').exists():
    time.sleep(2)
production = Path('/home/prmi/zwj/mycartgs')
output = root/'results/selector_online_ab_final_20261015'
output.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env.update(CARTGS_EXPERIMENTAL_SELECTOR_CUDA='1', CUDA_HOME='/usr/local/cuda-11.8',
           CXX='/usr/bin/g++-9', MAX_JOBS='2', TORCH_CUDA_ARCH_LIST='8.6',
           TORCH_EXTENSIONS_DIR=str(root/'python_extensions'), MPLBACKEND='Agg')
env['PATH'] = '/home/prmi/.local/bin:/usr/local/cuda-11.8/bin:'+env.get('PATH', '')
env['LD_LIBRARY_PATH'] = str(root/'lib')+':'+str(production/'third_party/libtorch-cu118/libtorch/lib')+':/usr/local/cuda-11.8/lib64:'+env.get('LD_LIBRARY_PATH', '')
# evo reads its own backend independently of MPLBACKEND. Use an isolated copy
# of the evaluator to override only plotting and image file writes; metric
# calculations and all rendering/model code stay identical.
evaluation_script = root/'python/selector_acceptance_eval.py'
evaluation_source = (root/'python/run.py').read_text()
evaluation_source = evaluation_source.replace('from evo.tools import plot',
    'from evo.tools import settings\nsettings.SETTINGS["plot_backend"] = "Agg"\nfrom evo.tools import plot')
evaluation_source = evaluation_source.replace('import cv2\n', 'import cv2\ncv2.imwrite = lambda *args, **kwargs: True\n', 1)
evaluation_script.write_text(evaluation_source)
env['TORCH_HOME'] = '/home/prmi/.cache/torch'
python = '/home/prmi/anaconda3/envs/zwj/bin/python'
binary = root/'bin/tum_rgbd_cuda_tf_online'
source_config = output/'source_fast.yaml'
if not source_config.exists():
    shutil.copy2(production/'cfg/selector/fast.yaml', source_config)
base = source_config.read_text()
assert re.search(r'Selector.enabled:\s*1', base)
assert re.search(r'Training.force_tail_optimization_enabled:\s*0', base)
for backend in ('legacy', 'cuda'):
    text = re.sub(r'^Selector.backend:.*\n', '', base, flags=re.M)
    (output/f'{backend}.yaml').write_text(text+f'\nSelector.backend: "{backend}"\n')
scenes = {
    'room': ('rgbd_dataset_freiburg1_room','tum_freiburg1_desk.yaml','fr1_room.txt'),
    'desk': ('rgbd_dataset_freiburg1_desk','tum_freiburg1_desk.yaml','tum_freiburg1_desk.txt'),
    'xyz': ('rgbd_dataset_freiburg2_xyz','tum_freiburg2_xyz.yaml','tum_freiburg2_xyz.txt'),
    'long': ('rgbd_dataset_freiburg3_long_office_household','tum_freiburg3_long_office_household.yaml','tum_freiburg3_long_office_household.txt'),
}
status_path = output/'status.json'
status = json.loads(status_path.read_text()) if status_path.exists() else {'runs': []}
status['scope'] = 'Unmodified online Mapping until SLAM shutdown; same inherited online-only driver for both backends; no post-SLAM tail.'
status['source_config_sha256'] = hashlib.sha256(source_config.read_bytes()).hexdigest()
status['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
status['mapper_library_sha256'] = hashlib.sha256((root/'lib/libgaussian_mapper.so').read_bytes()).hexdigest()


def save():
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n')


def execute(command, path, timeout):
    started = time.monotonic()
    with path.open('w') as stream:
        result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    return dict(exit=result.returncode, seconds=time.monotonic()-started, command=list(map(str, command)))


try:
    for scene, (dataset, slam, association) in scenes.items():
        for backend in ('legacy', 'cuda'):
            for repeat in range(1,4):
                name = f'{scene}/{backend}/repeat{repeat}'
                old = next((r for r in status['runs'] if r['name'] == name), None)
                if old and old.get('complete'):
                    continue
                run = output/name
                run.mkdir(parents=True, exist_ok=True)
                item = old or dict(name=name, scene=scene, backend=backend, repeat=repeat)
                if old is None:
                    status['runs'].append(item)
                status['active'] = name
                save()
                # Each process independently starts from seed0, exactly as the
                # production example. These are three repeated runs, not three seeds.
                if 'train' not in item or item['train']['exit'] != 0:
                    command = [str(binary), str(production/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt'),
                        str(root/'cfg/ORB_SLAM3/RGB-D/TUM'/slam), str(root/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml'),
                        '/home/prmi/datasets/TUM_RGBD/'+dataset,
                        str(root/'cfg/ORB_SLAM3/RGB-D/TUM/associations'/association), str(run), 'no_viewer',
                        '--selector-config', str(output/f'{backend}.yaml')]
                    item['train'] = execute(command, run/'train.log', 3600)
                    save()
                    if item['train']['exit']:
                        raise RuntimeError('Training failed: '+name)
                command = [python, str(root/'python/eval_lod.py'), str(run), '/home/prmi/datasets/TUM_RGBD/'+dataset,
                           '--output', str(run/'evaluation'), '--run-script', str(evaluation_script)]
                item['evaluation'] = execute(command, run/'evaluation.log', 3600)
                save()
                if item['evaluation']['exit']:
                    raise RuntimeError('Evaluation failed: '+name)
                # Remove only images generated inside this task's evaluation
                # directory after every requested metric file has been written.
                metric_files = list((run/'evaluation').rglob('eval.txt'))
                assert len(metric_files) >= 5, 'Full + four LoD metric sets are required'
                for image in (run/'evaluation').rglob('image'):
                    assert image.resolve().is_relative_to(output.resolve())
                    shutil.rmtree(image)
                item['metrics'] = {str(p.relative_to(run)): p.read_text() for p in metric_files}
                item['complete'] = True
                save()
                print(name+' complete', flush=True)
    status.update(active=None, complete=True)
    save()
except Exception as error:
    status['error'] = repr(error)
    save()
    raise

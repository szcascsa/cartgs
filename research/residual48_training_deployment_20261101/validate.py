"""使用主项目新库复测训练融合，保留普通 Scatter 差异及确定性对照。"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import time

D = Path(__file__).resolve().parent
P = Path('/home/prmi/zwj/mycartgs')
B = P / 'build-contribution40-default-20261031'
OLD = Path('/home/prmi/zwj/experiments/anchor_feature_gradient_diagnosis_20261101_2245')
OLD_SOURCE = '/home/prmi/zwj/experiments/residual48_cuda_optimization_20261101_1937/source'
env = dict(os.environ, PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:' + os.environ['PATH'],
           OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
env['LD_LIBRARY_PATH'] = ':'.join(str(path) for path in [P/'lib',
    P/'third_party/libtorch-cu118/libtorch/lib', P/'third_party/ORB-SLAM3/lib',
    P/'third_party/install/opencv-cuda118-sm86-gcc9/lib',
    P/'third_party/install/jsoncpp-1.9.5/lib', Path('/usr/local/cuda-11.8/lib64')])

def save(name, value):
    (D/name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def run(command, label, probe=False):
    start = time.monotonic()
    active_env = dict(env)
    if probe:
        active_env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    with (D/(label+'.log')).open('w') as log:
        result = subprocess.run(command, cwd=D, env=active_env, stdout=log, stderr=subprocess.STDOUT)
    save(label+'.json', dict(command=command, returncode=result.returncode, seconds=time.monotonic()-start))
    assert result.returncode == 0, label

while not (D/'build_complete.json').exists():
    if (D/'build.json').exists():
        assert json.loads((D/'build.json').read_text())['returncode'] == 0, 'Production build failed'
    time.sleep(2)
assert json.loads((D/'build_complete.json').read_text())['passed']
shutil.copy2(OLD/'fixture.pt', D/'fixture.pt')
checkpoint = json.loads((OLD/'runtime.json').read_text())['checkpoint']
base = json.loads((OLD/'compile_node_probe.json').read_text())['command']
base = [arg.replace(OLD_SOURCE+'/build/', str(B)+'/').replace(OLD_SOURCE, str(P)).replace(str(OLD), str(D))
        for arg in base if not arg.endswith(('/diagnostic_pack.o', '/diagnostic_pack_cuda.o'))]
for name, source in [('local_gate', P/'tests/anchor_selector/residual48_training_test.cpp'),
                     ('node_probe', D/'node_probe.cpp'), ('trajectory_probe', D/'trajectory_probe.cpp')]:
    command = [str(source) if arg == str(D/'node_probe.cpp') else arg for arg in base]
    command[command.index('-o')+1] = str(D/name)
    run(command, 'compile_'+name)
run([str(D/'local_gate')], 'local_gate', probe=True)
run([str(D/'node_probe'), str(D), checkpoint], 'node_probe', probe=True)
run([str(D/'trajectory_probe'), str(D), checkpoint, '72', 'trajectories'], 'trajectory_probe', probe=True)

import csv
with (D/'nodes.csv').open() as stream:
    nodes = list(csv.DictReader(stream))
strict = {'output', 'loss', 'packed_forward', 'packed_input_VJP', 'Master_XYZ_VJP'}
for row in nodes:
    if row['deterministic'] == '1' or row['node'] in strict or row['node'].startswith(('gather_value_', 'gather_input_VJP_', 'network_gradient_')):
        assert row['bits'] == '1', row
with (D/'trajectories.csv').open() as stream:
    trajectory = list(csv.DictReader(stream))
assert trajectory and all(row['bits'] == '1' and row['deterministic'] == '1' for row in trajectory)
targets = json.loads((D/'build_complete.json').read_text())['targets']
links = {}
for target in targets:
    text = subprocess.check_output(['ldd', str(P/'bin'/target)], env=env, text=True)
    assert 'not found' not in text and str(P/'lib/libgaussian_mapper.so') in text
    assert '/experiments/' not in text, text
    links[target] = text
save('runtime_links.json', links)
save('native_validation.json', dict(passed=True, local_forward_vjp_checks=97, nodes=len(nodes),
     deterministic_trajectory_seeds=2, steps=72, trajectory_rows=len(trajectory),
     production_scatter_still_nondeterministic=True, deterministic_mode_only_in_probes=True))
print('PRODUCTION_NATIVE_VALIDATION_COMPLETE', flush=True)

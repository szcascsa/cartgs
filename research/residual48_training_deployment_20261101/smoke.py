"""Replica office2 部署冒烟；1500步仅为独立测试预算，不改生产配置。"""
from pathlib import Path
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time

D = Path(__file__).resolve().parent
P = Path('/home/prmi/zwj/mycartgs')
assert json.loads((D/'native_validation.json').read_text())['passed']
run = D/'runs/ONLINE_NSYS'
run.mkdir(parents=True, exist_ok=False)
text = (P/'cfg/selector/selector_config.yaml').read_text()
assert 'CUDA.fused_residual48_training_input: 1' in text
off = text.replace('CUDA.fused_residual48_training_input: 1', 'CUDA.fused_residual48_training_input: 0')
(D/'selector_training_fusion_off.yaml').write_text(off)
text = re.sub(r'^Selector.max_mapping_iterations:.*$', 'Selector.max_mapping_iterations: 1500', text, flags=re.M)
assert 'Selector.max_mapping_iterations: 1500' in text
# Omit both new keys to verify C++ defaults, rather than only the explicit YAML values.
text = re.sub(r'^CUDA.fused_residual48_(?:training_)?input:.*\n', '', text, flags=re.M)
config = run/'selector.yaml'
config.write_text(text)
env = dict(os.environ, PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:' + os.environ['PATH'],
           OMP_NUM_THREADS='2', CARTGS_FUSION_PROFILE='1')
env['LD_LIBRARY_PATH'] = ':'.join(str(path) for path in [P/'lib',
    P/'third_party/libtorch-cu118/libtorch/lib', P/'third_party/ORB-SLAM3/lib',
    P/'third_party/install/opencv-cuda118-sm86-gcc9/lib',
    P/'third_party/install/jsoncpp-1.9.5/lib', Path('/usr/local/cuda-11.8/lib64')])
command = [str(P/'bin/replica_rgbd'), str(P/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt'),
           str(P/'cfg/ORB_SLAM3/RGB-D/Replica/office2.yaml'),
           str(P/'cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml'),
           '/home/prmi/zwj/deployments/contribution40_default_20261031/office2_first400',
           str(run), 'no_viewer', '--selector-config', str(config)]
nsys = shutil.which('nsys', path=env['PATH'])
assert nsys, 'Nsight Systems unavailable'
profile = D/'online_smoke'
profile_command = [nsys, 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                   '-o', str(profile), *command]
start = time.monotonic()
with (D/'smoke.log').open('w') as log:
    result = subprocess.run(profile_command, env=env, cwd=P, stdout=log, stderr=subprocess.STDOUT)
(D/'smoke_command.json').write_text(json.dumps(dict(command=profile_command, returncode=result.returncode,
    seconds=time.monotonic()-start, test_budget=1500, production_budget_modified=False), indent=2)+'\n')
assert result.returncode == 0, 'See smoke.log'
shutdown = list(run.glob('*_shutdown'))
assert len(shutdown) == 1, shutdown
root = shutdown[0]
files = list(root.rglob('*.yaml')) + list(root.rglob('*.json'))
(D/'smoke_output_files.json').write_text(json.dumps([str(p) for p in files], indent=2)+'\n')
subprocess.run([nsys, 'export', '--type', 'sqlite', '--output', str(profile)+'.sqlite',
                str(profile)+'.nsys-rep'], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
connection = sqlite3.connect(str(profile)+'.sqlite')
rows = connection.execute('''SELECT s.value, COUNT(*), SUM(k.end-k.start)
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
    WHERE s.value LIKE '%trainingPack(%' OR s.value LIKE '%trainingXYZVjp(%'
    GROUP BY s.value''').fetchall()
assert any('trainingPack(' in row[0] and row[1] > 0 for row in rows), rows
assert any('trainingXYZVjp(' in row[0] and row[1] > 0 for row in rows), rows
connection.close()
(D/'smoke_kernel_identity.json').write_text(json.dumps(dict(passed=True, kernels=rows,
    checkpoint=str(root/'ply'), actual_training_forward_and_backward_observed=True), indent=2)+'\n')
print('ONLINE_TRAINING_FUSION_OBSERVED', rows, flush=True)

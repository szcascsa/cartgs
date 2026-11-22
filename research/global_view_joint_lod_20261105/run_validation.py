"""串行执行本分支验收，任一失败立即停止，保留全部失败现场。"""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
env = dict(os.environ)
env['LD_LIBRARY_PATH'] = ':'.join([
    str(ROOT/'runtime/lib'),
    '/home/prmi/zwj/experiments/real_lod_full_joint_20261103/source/third_party/ORB-SLAM3/lib',
    '/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib',
    '/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib',
    '/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib',
    '/usr/local/cuda-11.8/lib64',
])
env['OMP_NUM_THREADS'] = '2'
commands = [('sampling', [str(ROOT/'runtime/bin/view_budget_training_test'), 'sampling'])]
for mode in ['joint', 'empty', 'control', 'off', 'off_repeat', 'zero', 'gradient', 'health']:
    commands.append((mode, [str(ROOT/'runtime/bin/view_budget_training_test'), mode, str(ROOT/'runs'/('test_'+mode+'_final'))]))
commands += [
    ('legacy_real_lod', [str(ROOT/'runtime/bin/real_lod_test'), str(ROOT/'runs/test_legacy_real_lod_final')]),
    ('legacy_lod_health', [str(ROOT/'runtime/bin/lod_health_guard_test')]),
]
results = []
for name, command in commands:
    started = time.monotonic()
    with (ROOT/'logs'/('test_'+name+'.log')).open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    results.append({'name': name, 'command': command, 'returncode': result.returncode, 'seconds': time.monotonic()-started})
    (ROOT/'reports/TEST_RESULTS.json').write_text(json.dumps(results, indent=2))
    print(name, result.returncode, round(results[-1]['seconds'], 2), flush=True)
    if result.returncode:
        raise SystemExit(f'{name} failed; inspect logs before rerun')
images = [str(p) for p in (ROOT/'runs').rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
assert not images, images[:10]
(ROOT/'reports/ZERO_IMAGES.json').write_text(json.dumps({'images': len(images)}))
print('ALL_VALIDATION_PASSED', flush=True)

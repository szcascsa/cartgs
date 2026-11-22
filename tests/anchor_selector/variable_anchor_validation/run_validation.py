"""可变成员 Anchor：隔离构建及生命周期、梯度、旧路径回归门控。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path('/home/prmi/zwj/variable-anchor-20261021')
repo = root/'source'
out = root/'validation'
out.mkdir(exist_ok=True)
manifest = json.loads((root/'source_manifest.json').read_text())
if not repo.exists():
    shutil.copytree('/home/prmi/zwj/serial-joint-lod-20261021/source', repo, symlinks=True)
for path, digest in manifest.items():
    dest = repo/path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root/'overlay'/path, dest)
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest
env = dict(os.environ, PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'],
           OMP_NUM_THREADS='2', QT_QPA_PLATFORM='offscreen', MPLBACKEND='Agg')
cmake = '/home/prmi/.local/bin/cmake'
def status(stage, **kw):
    (out/'status.json').write_text(json.dumps(dict(stage=stage, **kw), indent=2))
cmd = [cmake, '-S', str(repo), '-B', str(root/'build'), '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=86',
       '-DCMAKE_CXX_COMPILER=/usr/bin/g++-9', '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc',
       '-DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch',
       '-DOpenCV_DIR=/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib/cmake/opencv4',
       '-Djsoncpp_DIR=/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib/cmake/jsoncpp',
       '-DEigen3_DIR=/home/prmi/zwj/mycartgs/third_party/eigen-3.3.7/build-cmake',
       '-DCARTGS_BUILD_TESTS=ON', '-DCARTGS_EXPERIMENTAL_SELECTOR_CUDA=ON']
status('BUILDING')
with (out/'build.log').open('w') as log:
    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([cmake, '--build', str(root/'build'), '--target', 'variable_anchor_test',
                    'variable_anchor_mapping_test', 'explicit_anchor_test', 'anchor_opacity_pruning_test',
                    'serial_joint_lod_mapping_test', 'explicit_anchor_mapping_test', 'anchor_checkpoint_probe', 'tum_rgbd', '-j4'],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
gates = {}
def test(name, args, cwd=out):
    cwd.mkdir(exist_ok=True)
    status('CORRECTNESS', test=name)
    with (out/(name+'.log')).open('w') as log:
        result = subprocess.run(list(map(str, args)), cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    gates[name] = dict(passed=result.returncode == 0, code=result.returncode)
    (out/'correctness.json').write_text(json.dumps(gates, indent=2))
    assert result.returncode == 0, name
for device in ['cpu', 'cuda']:
    folder = out/device
    test('variable_'+device, [repo/'bin/variable_anchor_test', device], folder)
    test('python_'+device, ['/home/prmi/anaconda3/envs/zwj/bin/python',
                          repo/'tests/anchor_selector/variable_anchor_validation/check_python.py', str(repo), device], folder)
    test('fixed_'+device, [repo/'bin/explicit_anchor_test', device], folder)
    test('pruning_'+device, [repo/'bin/anchor_opacity_pruning_test', device], folder)
test('variable_mapping', [repo/'bin/variable_anchor_mapping_test', '--parity'])
test('fixed_mapping', [repo/'bin/explicit_anchor_mapping_test', '--parity'])
test('serial_joint_mapping', [repo/'bin/serial_joint_lod_mapping_test', '--parity'])
status('CORRECTNESS_PASSED')
print('PASS all variable Anchor gates', flush=True)

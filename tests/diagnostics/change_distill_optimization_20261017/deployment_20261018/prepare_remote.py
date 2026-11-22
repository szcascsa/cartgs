"""核对正式源码、备份更新范围和产物，构建链接正式库的同一benchmark。"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path('/home/prmi/zwj/mycartgs')
D = Path(__file__).resolve().parent
AUDIT = ROOT / 'results/change_distill_optimization/20261017'
manifest = json.loads((AUDIT / 'baseline/raw/source_manifest.json').read_text())
targets = json.loads((AUDIT / 'integrity.json').read_text())['production_files_changed']
targets += ['tests/frozen_teacher_target_cache_test.cpp']
backup = D / 'backup'
assert not backup.exists(), 'Do not overwrite an existing deployment backup'
backup.mkdir()
for name in manifest:
    assert (ROOT / name).read_text() == (AUDIT / 'baseline/source' / name).read_text(), name
records = {}
for name in targets:
    p = ROOT / name
    q = backup / name
    q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, q)
    records[name] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                     'normalized_sha256': hashlib.sha256(p.read_text().encode()).hexdigest()}
for name in ['bin', 'lib']:
    shutil.copytree(ROOT / name, backup / name, symlinks=True)
for name in ['CMakeCache.txt', 'build.ninja', 'libselector_cuda.a', 'libtransform_field_cuda.a']:
    p = ROOT / 'build-cuda118-sm86-gcc9' / name
    if p.exists():
        q = backup / 'build-cuda118-sm86-gcc9' / name
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
(D / 'source_before.json').write_text(json.dumps(records, indent=2))
T = ROOT / 'third_party/libtorch-cu118/libtorch'
J = ROOT / 'third_party/install/jsoncpp-1.9.5'
C = Path('/usr/local/cuda-11.8')
command = ['/usr/bin/g++', '-O3', '-DNDEBUG', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=1',
           '-DCARTGS_EXPERIMENTAL_SELECTOR_CUDA', '-I' + str(ROOT), '-I' + str(T / 'include'),
           '-I' + str(T / 'include/torch/csrc/api/include'), '-I' + str(J / 'include'),
           '-I' + str(C / 'include'), '-I' + str(AUDIT / 'baseline'),
           str(AUDIT / 'baseline/benchmark.cpp'), '-o', str(D / 'benchmark_production')]
for p in [ROOT / 'lib', T / 'lib', J / 'lib', C / 'lib64', ROOT / 'third_party/ORB-SLAM3/lib']:
    command += ['-L' + str(p), '-Wl,-rpath,' + str(p)]
command += ['-lgaussian_mapper', '-ltorch', '-ltorch_cpu', '-ltorch_cuda', '-lc10', '-lc10_cuda',
            '-lcudart', '-lnvToolsExt', '-ljsoncpp']
(D / 'benchmark_build_command.json').write_text(json.dumps(command, indent=2))
subprocess.run(command, check=True)
(D / 'runner_sha256.txt').write_text(hashlib.sha256((D / 'benchmark_production').read_bytes()).hexdigest())
print('Source normalization parity, backup and production-linked runner PASS')

"""部署已审计的性能版；仅覆盖清单文件，备份旧源码与运行产物。"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

stage = Path(__file__).resolve().parent
root = Path('/home/prmi/zwj/mycartgs')
build = root / 'build-performance-20261023-121515'
manifest = json.loads((stage / 'manifest.json').read_text())
changes = json.loads((stage / 'changes.json').read_text())
status = {'started': time.time(), 'root': str(root), 'build': str(build)}


def save():
    (stage / 'status.json').write_text(json.dumps(status, indent=2))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, name, env=None, cwd=root):
    status['phase'] = name
    save()
    with (stage / (name + '.log')).open('w') as log:
        result = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{name} failed: {result.returncode}')


try:
    assert root.resolve() == root and not root.is_symlink()
    resume = '--resume' in sys.argv
    assert resume or (not build.exists() and not (stage / 'backup').exists())
    with zipfile.ZipFile(stage / 'payload.zip') as archive:
        assert set(archive.namelist()) == set(manifest['files'])
        for name, expected in manifest['files'].items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
        for change in changes:
            if resume:
                continue
            target = root / change['path']
            assert target.resolve().is_relative_to(root)
            assert (sha(target) if target.is_file() else None) == change['before']
            if target.exists():
                backup = stage / 'backup/source' / change['path']
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
        for folder in ['bin', 'lib']:
            if resume:
                continue
            for target in (root / folder).iterdir():
                if target.is_file():
                    backup = stage / 'backup/artifacts' / folder / target.name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, backup, follow_symlinks=False)
        for change in changes:
            if resume:
                continue
            target = root / change['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(change['path']))
    assert all(sha(root / name) == expected for name, expected in manifest['files'].items())
    status['source_files_verified'] = len(manifest['files'])
    status['changed_files'] = len(changes)
    save()
    cmake = '/home/prmi/.local/bin/cmake'
    run([cmake, '-S', root, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_CXX_COMPILER=/usr/bin/g++-9',
         '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc',
         '-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-9',
         '-DCMAKE_CUDA_ARCHITECTURES=86', '-DCARTGS_BUILD_TESTS=ON',
         '-DEigen3_DIR=' + str(root / 'third_party/eigen-3.3.7/build-cmake'),
         '-Djsoncpp_DIR=' + str(root / 'third_party/install/jsoncpp-1.9.5/lib/cmake/jsoncpp'),
         '-DTorch_DIR=' + str(root / 'third_party/libtorch-cu118/libtorch/share/cmake/Torch'),
         '-DOpenCV_DIR=' + str(root / 'third_party/install/opencv-cuda118-sm86-gcc9/lib/cmake/opencv4')],
        'configure')
    targets = ['tum_rgbd', 'tum_mono', 'replica_rgbd', 'replica_mono',
               'vector_mono', 'vector_stereo', 'view_result', 'train_colmap',
               'fused_mapping_adam_test', 'fused_compensation_decode_test',
               'fused_anchor_input_test', 'fused_anchor_stats_test',
               'fused_selector_sampling_test', 'renderer_workspace_test',
               'ssim_backward_test', 'private_compensation_mapping_test',
               'phase2_staged_mapping_benchmark']
    run([cmake, '--build', build, '-j2', '--target', *targets], 'compile')
    env = dict(os.environ, OMP_NUM_THREADS='2', QT_QPA_PLATFORM='offscreen')
    env['LD_LIBRARY_PATH'] = ':'.join([str(root / 'lib'),
        str(root / 'third_party/libtorch-cu118/libtorch/lib'),
        str(root / 'third_party/install/opencv-cuda118-sm86-gcc9/lib'),
        '/usr/local/cuda-11.8/lib64', env.get('LD_LIBRARY_PATH', '')])
    fixture = Path('/home/prmi/zwj/anchor-selector-r3-performance-20261022/native1')
    checkpoint = Path('/home/prmi/zwj/private-compensation-online-20261022/r3_independent14_seed2026/7465_shutdown/ply')
    full = fixture / 'Full_parity.pt'
    for binary, args in [
        ('fused_mapping_adam_test', [fixture]),
        ('fused_compensation_decode_test', [fixture]),
        ('fused_anchor_input_test', [checkpoint, fixture]),
        ('fused_anchor_stats_test', [checkpoint, full]),
        ('fused_selector_sampling_test', [checkpoint, full]),
        ('renderer_workspace_test', [checkpoint, full]),
        ('ssim_backward_test', [checkpoint, full]),
    ]:
        run([root / 'bin' / binary, *args, stage / 'validation' / binary], binary, env)
    lifecycle = stage / 'validation/r3_lifecycle'
    lifecycle.mkdir(parents=True)
    run([root / 'bin/private_compensation_mapping_test', 'cuda', 'fused'], 'r3_lifecycle', env, lifecycle)
    run(['/home/prmi/anaconda3/envs/zwj/bin/python', root / 'tests/interactive_launcher_test.py'],
        'launcher')
    assert all(sha(root / name) == expected for name, expected in manifest['files'].items())
    status.update(phase='complete', finished=time.time(), targets=targets,
                  passed=True, backup=str(stage / 'backup'))
    save()
    print('DEPLOY_AND_VALIDATION_PASS', flush=True)
except Exception as error:
    status.update(phase='failed', error=str(error), failed=time.time())
    save()
    raise

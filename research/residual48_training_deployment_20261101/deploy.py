"""部署 Residual48 输入训练融合默认，逐文件备份并重编译全部生产入口。"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time

D = Path(__file__).resolve().parent
P = Path('/home/prmi/zwj/mycartgs')
B = P / 'build-contribution40-default-20261031'
CMAKE = '/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake'
TARGETS = ['replica_rgbd', 'replica_mono', 'tum_rgbd', 'tum_mono',
           'vector_mono', 'vector_stereo', 'train_colmap', 'view_result']
env = dict(os.environ, PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:' + os.environ['PATH'])
env['LD_LIBRARY_PATH'] = ':'.join(str(p) for p in [P/'lib',
    P/'third_party/libtorch-cu118/libtorch/lib', P/'third_party/ORB-SLAM3/lib',
    P/'third_party/install/opencv-cuda118-sm86-gcc9/lib',
    P/'third_party/install/jsoncpp-1.9.5/lib', Path('/usr/local/cuda-11.8/lib64')])


def save(name, value):
    (D / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def execute(command, label):
    start = time.monotonic()
    with (D / (label + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=P, env=env, stdout=log, stderr=subprocess.STDOUT)
    save(label + '.json', dict(command=command, returncode=result.returncode,
                              seconds=time.monotonic() - start))
    assert result.returncode == 0, label + ' failed; see log'


def main():
    assert not (D / 'before.json').exists(), 'Deployment already started; inspect state before resuming'
    manifest = json.loads((D / 'deploy_manifest.json').read_text())
    scope = json.loads((D / 'source_manifest.json').read_text())
    unexpected = [n for n, h in scope.items() if n not in manifest and sha(P / n) != h]
    assert not unexpected, ('Unrelated source drift', unexpected)
    active = []
    for q in Path('/proc').iterdir():
        if not q.name.isdigit():
            continue
        try:
            args = (q / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if args and args[0].decode(errors='replace') in [str(P / 'bin' / n) for n in TARGETS]:
            active.append(int(q.name))
    assert not active, ('Production process active; do not overwrite running binaries', active)
    names = list(manifest) + ['bin/' + n for n in TARGETS]
    names += [str(p.relative_to(P)) for p in (P / 'lib').iterdir() if p.is_file()]
    before = {}
    for n in names:
        p = P / n
        assert p.resolve().is_relative_to(P.resolve()), n
        before[n] = sha(p)
        if p.is_file():
            target = D / 'before' / n
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    save('before.json', before)
    save('source_before.json', {n: sha(P / n) for n in scope})
    incoming = D / 'incoming'
    with tarfile.open(D / 'source_update.tar.gz') as archive:
        assert sorted(m.name for m in archive.getmembers()) == sorted(manifest)
        for m in archive.getmembers():
            assert m.isfile() and (incoming / m.name).resolve().is_relative_to(incoming.resolve())
        archive.extractall(incoming)
    assert all(sha(incoming / n) == h for n, h in manifest.items())
    for n in manifest:
        (P / n).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(incoming / n, P / n)
    assert all(sha(P / n) == h for n, h in scope.items())
    save('source_verified.json', dict(passed=True, files=len(scope), changed_files=list(manifest)))
    execute([CMAKE, '-S', str(P), '-B', str(B), '-DCARTGS_BUILD_TESTS=OFF'], 'configure')
    execute([CMAKE, '--build', str(B), '--parallel', '3', '--target', *TARGETS], 'build')
    assert all(sha(P / n) == h for n, h in scope.items())
    artifacts = {str(p.relative_to(P)): sha(p) for p in
                 [*(P / 'bin' / n for n in TARGETS), *(P / 'lib').glob('*.so')]}
    save('build_complete.json', dict(passed=True, targets=TARGETS, source_files=len(scope),
                                    artifacts=artifacts, backup=str(D / 'before')))
    print('RESIDUAL48_TRAINING_PRODUCTION_BUILD_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

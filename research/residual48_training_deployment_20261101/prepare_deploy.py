"""核对上轮受保护文件，只备份和同步本次 manifest 的差异。"""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile

D = Path(__file__).resolve().parent
P = Path('/home/prmi/zwj/mycartgs')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

protected = json.loads((D / 'protected_before.json').read_text())
drift = [name for name, value in protected.items() if sha(P / name) != value]
assert not drift, ('Unexpected production changes since diagnosis', drift)
scope = json.loads((D / 'source_manifest.json').read_text())
expected = D / 'source_expected'
with tarfile.open(D / 'source_snapshot.tar.gz') as archive:
    assert sorted(item.name for item in archive.getmembers()) == sorted(scope)
    for item in archive.getmembers():
        assert item.isfile() and (expected / item.name).resolve().is_relative_to(expected.resolve())
    archive.extractall(expected)
assert all(sha(expected / name) == value for name, value in scope.items())
changed = {name: value for name, value in scope.items() if sha(P / name) != value}
(D / 'deploy_manifest.json').write_text(json.dumps(changed, indent=2) + '\n')
with tarfile.open(D / 'source_update.tar.gz', 'w:gz') as archive:
    for name in changed:
        archive.add(expected / name, arcname=name)
print('changed source files', len(changed), sorted(changed), flush=True)
with (D / 'deployment.log').open('w') as log:
    process = subprocess.Popen(['/home/prmi/anaconda3/envs/zwj/bin/python', '-u', str(D / 'deploy.py')],
                               cwd=D, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(D / 'deployment_process.json').write_text(json.dumps({'pid': process.pid}) + '\n')
print('deployment_pid', process.pid, flush=True)

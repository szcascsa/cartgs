"""本地证据包校验：先验证路径/类型/归档SHA，再解包并逐文件核验。"""
from pathlib import Path, PurePosixPath
import hashlib,json,tarfile
HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()

summary=json.loads((HERE/'CPU_MIDRATIO_INFERENCE_ARTIFACTS.sha256.json').read_text())
archive=HERE/summary['archive'];assert sha(archive)==summary['sha256']
with tarfile.open(archive) as tar:
    members=tar.getmembers();assert len({m.name for m in members})==len(members)
    for member in members:
        path=PurePosixPath(member.name)
        assert member.isfile() and not path.is_absolute() and '..' not in path.parts and ':' not in member.name,member.name
        target=(HERE/str(path)).resolve();assert target.is_relative_to(HERE.resolve()),target
    manifest=json.load(tar.extractfile('reports/ARTIFACT_MANIFEST.json'))
    assert set(m.name for m in members)==set(manifest['files'])|{'reports/ARTIFACT_MANIFEST.json'}
    for member in members:
        data=tar.extractfile(member).read()
        if member.name in manifest['files']:
            expected=manifest['files'][member.name];assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'],member.name
        target=HERE/member.name
        if target.exists():assert target.read_bytes()==data,('Local conflict',member.name)
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
changed=[name for name,item in manifest['files'].items() if sha(HERE/name)!=item['sha256']]
assert not changed,changed
result=dict(passed=True,archive_sha256=summary['sha256'],files_checked=len(manifest['files']),changed=changed)
(HERE/'reports/LOCAL_DOWNLOAD_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

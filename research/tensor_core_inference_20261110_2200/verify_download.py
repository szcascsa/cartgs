"""下载归档和逐文件校验；已有文件冲突不覆盖。"""
from pathlib import Path,PurePosixPath
import tarfile,hashlib,json
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary=json.loads((HERE/'TENSOR_CORE_INFERENCE_ARTIFACTS.sha256.json').read_text());archive=HERE/summary['archive'];assert sha(archive)==summary['sha256']
with tarfile.open(archive) as tar:
    members=tar.getmembers();assert len({m.name for m in members})==len(members)
    manifest=json.load(tar.extractfile('reports/ARTIFACT_MANIFEST.json'))
    assert set(m.name for m in members)==set(manifest['files'])|{'reports/ARTIFACT_MANIFEST.json'}
    for m in members:
        path=PurePosixPath(m.name);assert m.isfile() and not path.is_absolute() and '..' not in path.parts and ':' not in m.name
        dest=(HERE/m.name).resolve();assert dest.is_relative_to(HERE.resolve())
        data=tar.extractfile(m).read()
        if m.name in manifest['files']:
            item=manifest['files'][m.name];assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
        if dest.exists():assert dest.read_bytes()==data,('Local conflict',m.name)
        else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
changed=[name for name,item in manifest['files'].items() if sha(HERE/name)!=item['sha256']];assert not changed
result=dict(passed=True,archive_sha256=summary['sha256'],files_checked=len(manifest['files']),changed=changed)
(HERE/'reports/LOCAL_DOWNLOAD_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

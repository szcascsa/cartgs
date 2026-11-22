"""在本地核对服务器交付包及每个证据文件；只解包至本研究目录。"""
from pathlib import Path
import hashlib,json,tarfile,time

HERE=Path(__file__).resolve().parent
identity=json.loads((HERE/'reports/DELIVERY_ARCHIVE.json').read_text())
archive=HERE/'evaluation/DELIVERY.tar.gz'
assert archive.stat().st_size==identity['bytes']
assert hashlib.sha256(archive.read_bytes()).hexdigest()==identity['sha256']
with tarfile.open(archive,'r:gz') as stream:
    for member in stream.getmembers():(HERE/member.name).resolve().relative_to(HERE.resolve())
    stream.extractall(HERE,filter='data')
manifest=json.loads((HERE/'reports/DELIVERY_FILE_HASHES.json').read_text());errors=[]
for name,expected in manifest.items():
    if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=expected:errors.append(name)
figure=json.loads((HERE/'figures/DATA_MANIFEST.json').read_text())
for name,expected in figure['source_csv'].items():
    assert hashlib.sha256((HERE/'reports'/name).read_bytes()).hexdigest()==expected,name
result=dict(passed=not errors,checked_at=time.time(),files=len(manifest),errors=errors,archive_sha256=identity['sha256'],plot_sources_match=True)
(HERE/'reports/LOCAL_DELIVERY_VERIFICATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert not errors

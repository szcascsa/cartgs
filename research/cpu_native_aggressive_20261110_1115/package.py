"""只打包本轮源码、二进制和必要证据；不复制数据集/checkpoint或共享环境。"""
from pathlib import Path
import hashlib,json,tarfile,time

HERE=Path(__file__).resolve().parent
assert HERE==Path('/home/prmi/zwj/experiments/cpu_native_aggressive_20261110_1115')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()

integrity=json.loads((HERE/'reports/INTEGRITY_FINAL.json').read_text());assert integrity['passed']
paths=set(HERE.glob('*.py'))|set(HERE.glob('*.md'))|set(HERE.glob('SOURCE_V*.tar.gz'))
for directory in ['source','reports','logs','figures','noise_snapshots']:
    for path in (HERE/directory).rglob('*'):
        if path.is_file() and path.suffix in ['.py','.cpp','.h','.json','.jsonl','.csv','.md','.log','.pt','.png','.svg']:
            paths.add(path)
for directory in ['build','build_v1','build_v2','build_v3','build_v4','build_sanitize','build_ubsan']:
    paths.update((HERE/directory).glob('*.so'))
paths={p for p in paths if p.name!='ARTIFACT_MANIFEST.json'}
manifest=dict(created_at=time.time(),files={str(p.relative_to(HERE)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)},
    checkpoints_copied=False,datasets_copied=False,objects_included=False,production_deployed=False)
target=HERE/'reports/ARTIFACT_MANIFEST.json';target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));paths.add(target)
archive=HERE/'CPU_AGGRESSIVE_INFERENCE_ARTIFACTS.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
    for path in sorted(paths):tar.add(path,arcname=str(path.relative_to(HERE)),recursive=False)
result=dict(archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,files=len(paths))
(HERE/'CPU_AGGRESSIVE_INFERENCE_ARTIFACTS.sha256.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

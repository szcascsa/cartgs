"""仅打包本轮源码和证据，不含模型、数据集、可重建对象或共享环境。"""
from pathlib import Path
import json,hashlib,tarfile,time
HERE=Path(__file__).resolve().parent
assert HERE==Path('/home/prmi/zwj/experiments/tensor_core_inference_20261110_2200')
assert json.loads((HERE/'reports/INTEGRITY_FINAL.json').read_text())['passed']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
paths=set(HERE.glob('*.py'))|set(HERE.glob('*.md'))|set(HERE.glob('*.txt'))
suffixes=['.py','.cpp','.cu','.cuh','.h','.inc','.json','.jsonl','.csv','.md','.log','.png','.svg','.patch','.so','.nsys-rep','.txt']
for directory in ['source','reports','logs','figures']+[f'development_v{i}' for i in range(1,7)]:
    paths.update(p for p in (HERE/directory).rglob('*') if p.is_file() and p.suffix in suffixes)
paths.update((HERE/'build').glob('*.so'));paths.update((HERE/'build').glob('build.ninja'))
paths={p for p in paths if p.name!='ARTIFACT_MANIFEST.json'}
manifest=dict(created_at=time.time(),files={str(p.relative_to(HERE)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)},models_copied=False,datasets_copied=False,objects_included=False)
p=HERE/'reports/ARTIFACT_MANIFEST.json';p.write_text(json.dumps(manifest,indent=2,ensure_ascii=False));paths.add(p)
archive=HERE/'TENSOR_CORE_INFERENCE_ARTIFACTS.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
    for p in sorted(paths):tar.add(p,arcname=str(p.relative_to(HERE)),recursive=False)
result=dict(archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,files=len(paths))
(HERE/'TENSOR_CORE_INFERENCE_ARTIFACTS.sha256.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

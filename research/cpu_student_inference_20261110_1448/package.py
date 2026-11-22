"""本轮源码/证据/二进制打包；禁止复制checkpoint、数据集和共享环境。"""
from pathlib import Path
import hashlib,json,tarfile,time
HERE=Path(__file__).resolve().parent
assert HERE==Path('/home/prmi/zwj/experiments/cpu_student_inference_20261110_1448')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()
assert json.loads((HERE/'reports/INTEGRITY_FINAL.json').read_text())['passed']
paths=set(HERE.glob('*.py'))|set(HERE.glob('*.md'));paths={p for p in paths if p.name!='transport.py'}
for directory in ['source','backend_snapshot','reports','logs','figures','noise_snapshots']:
    for path in (HERE/directory).rglob('*'):
        if path.is_file() and (path.suffix in ['.py','.cpp','.h','.json','.jsonl','.csv','.md','.log','.pt','.png','.svg','.patch','.so'] or path.name.startswith('asan')):paths.add(path)
for name in ['build','build_asan','build_ubsan']:
    paths.update((HERE/name).glob('*.so'))
    paths.update((HERE/name).glob('*.cpp'));paths.update((HERE/name).glob('*.h'));paths.update((HERE/name).glob('build.ninja'))
paths={p for p in paths if p.name!='ARTIFACT_MANIFEST.json'}
manifest=dict(created_at=time.time(),files={str(p.relative_to(HERE)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)},
    checkpoints_copied=False,datasets_copied=False,objects_included=False,production_deployed=False)
target=HERE/'reports/ARTIFACT_MANIFEST.json';target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));paths.add(target)
archive=HERE/'CPU_STUDENT_INFERENCE_ARTIFACTS.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
    for path in sorted(paths):tar.add(path,arcname=str(path.relative_to(HERE)),recursive=False)
result=dict(archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,files=len(paths))
(HERE/'CPU_STUDENT_INFERENCE_ARTIFACTS.sha256.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

"""只归档可复现证据，不复制数据集/checkpoint/对象文件。"""
from pathlib import Path
import hashlib
import json
import tarfile
import time

ROOT=Path(__file__).resolve().parent
S=ROOT/'background_stage'
delivery=S/'reports/delivery'
assert json.loads((delivery/'INTEGRITY.json').read_text())['passed']
assert json.loads((S/'reports/POST_CHECKS.json').read_text())['state']=='complete'
files=set(ROOT.glob('*.py'))
files.update(ROOT.glob('*.cuh'))
files.update(ROOT.glob('*.md'))
files.update(p for p in (ROOT/'source').rglob('*') if p.is_file())
files.update(p for p in (S/'source').rglob('*') if p.is_file())
files.update(p for p in S.glob('*.py') if p.is_file())
files.update(ROOT.glob('build/*.so'));files.update(S.glob('build/*.so'))
files.update(p for p in delivery.rglob('*') if p.is_file())
files.update(p for p in (ROOT/'reports/delivery').rglob('*') if p.is_file())
for parent in (ROOT,S):
    files.update(parent.glob('reports/*.json'))
    files.update(parent.glob('reports/jobs/*.json'))
    files.update(parent.glob('logs/*'))
for folder in ('validation01','validation02','validation03','validation04','validation05',
               'lifecycle03','memcheck01'):
    files.update(p for p in (ROOT/'reports'/folder).glob('*') if p.is_file())
for folder in ('validation01','memcheck01','initcheck01','synccheck01','PROFILE'):
    files.update(p for p in (S/'reports'/folder).glob('*') if p.is_file())
files.update(S.glob('reports/bg_profile01.nsys-rep'))
manifest={str(p.relative_to(ROOT)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
          for p in sorted(files)}
manifest_path=ROOT/'DELIVERY_SHA256.json'
manifest_path.write_text(json.dumps(manifest,indent=2))
files.add(manifest_path)
archive=ROOT/'RENDER_COUNT_SCALING_EVIDENCE.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
    for path in sorted(files):tar.add(path,arcname=path.relative_to(ROOT),recursive=False)
print('DELIVERY_ARCHIVE',len(files),archive.stat().st_size,hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),flush=True)
# SQLite可由所保留nsys-rep重新导出，编译对象可重建，仅清理本轮自己的这些文件。
cleanup=[]
for path in [*(ROOT/'build').glob('*.o'),*(S/'build').glob('*.o'),
             *(ROOT/'reports').glob('*.sqlite'),*(S/'reports').glob('*.sqlite')]:
    absolute=path.resolve()
    assert ROOT.resolve() in absolute.parents and path.is_file()
    cleanup.append(dict(path=str(path),bytes=path.stat().st_size))
    path.unlink()
(ROOT/'REBUILDABLE_CLEANUP.json').write_text(json.dumps(dict(time=time.time(),files=cleanup),indent=2))
print('OWN_REBUILDABLE_CLEANUP',len(cleanup),sum(x['bytes'] for x in cleanup),flush=True)

"""完整保留研究证据；只读取本目录，不包含认证信息或外部数据集。"""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert (ROOT/'all_analysis_complete').exists()
sources={}
for p in ROOT.glob('af_*.py'):
    digest=hashlib.sha256(p.read_bytes()).hexdigest();sources[p.name]=digest
    snapshot=ROOT/'code_versions'/(p.stem+'_'+digest[:12]+'.py')
    if not snapshot.exists():snapshot.write_bytes(p.read_bytes())
(ROOT/'current_research_source_manifest.json').write_text(json.dumps(sources,indent=2))
files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['DELIVERY_MANIFEST.json','DELIVERY_ARCHIVE.json'] and not p.name.endswith(('.tar.gz','.zip'))]
manifest={str(p.relative_to(ROOT)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
(ROOT/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
archive=ROOT.parent/'anchor-feature-research-20261027-complete.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
    for p in files+[ROOT/'DELIVERY_MANIFEST.json']:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
record=dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(manifest))
(ROOT/'DELIVERY_ARCHIVE.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)

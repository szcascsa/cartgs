"""仅打包本轮可复现证据，保留逐文件SHA，不复制模型/数据集。"""
from pathlib import Path
import hashlib,json,tarfile,time

HERE=Path(__file__).resolve().parent
assert json.loads((HERE/'reports/INTEGRITY_FINAL.json').read_text())['passed']
files=[]
for name in ['source','reports','logs','figures']:
    files.extend(p for p in (HERE/name).rglob('*') if p.is_file() and p.name not in ['DELIVERY_FILE_HASHES.json','DELIVERY_ARCHIVE.json'])
for name in ['build','build_v2','build_v3','build_v4']:
    files.extend(p for p in (HERE/name).glob('*') if p.is_file() and (p.suffix=='.so' or p.name=='build.ninja'))
files.extend(HERE.glob('*.py'));files.extend(HERE.glob('source*.tar.gz'))
files.append(HERE/'LARGE_FOOTPRINT_CUDA_OPTIMIZATION_REPORT.md')
files=sorted(set(files));hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
manifest=HERE/'reports/DELIVERY_FILE_HASHES.json';manifest.write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
path=HERE/'evaluation/DELIVERY.tar.gz';assert not path.exists()
with tarfile.open(path,'w:gz') as archive:
    for p in files+[manifest]:archive.add(p,arcname=str(p.relative_to(HERE)),recursive=False)
identity=dict(created_at=time.time(),files=len(files),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),path=str(path))
(HERE/'reports/DELIVERY_ARCHIVE.json').write_text(json.dumps(identity,indent=2));print(json.dumps(identity,indent=2))

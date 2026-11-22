"""仅打包本轮可复现证据；不包含数据集、原模型或共享缓存。"""
from pathlib import Path
import hashlib,json,tarfile,time

HERE=Path(__file__).resolve().parent
verification=json.loads((HERE/'DELIVERY_VERIFICATION.json').read_text());assert verification['passed']
archive=HERE/'LOW_POINT_RENDERING_SIX_HOUR_DELIVERY.tar.gz';assert not archive.exists()
files=[]
for path in sorted(HERE.iterdir()):
    if path.is_file() and path.suffix in ['.py','.md','.csv','.json','.txt'] and path.name not in [
        'DELIVERY_RECEIPT.json','DELIVERY_MANIFEST.json','LOCAL_DELIVERY_VERIFIED.json']:files.append(path)
for name in ['source','reports','logs','evaluation','figures']:
    directory=HERE/name
    if directory.exists():files += [p for p in sorted(directory.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p!=HERE/'reports/PACKAGE_DELIVERY_LOG.txt']
for name in ['build','build_graph']:
    directory=HERE/name
    files += [p for p in sorted(directory.iterdir()) if p.is_file() and (p.suffix=='.so' or p.name=='build.ninja')]
assert all(path.resolve().is_relative_to(HERE) and not path.is_symlink() for path in files)
assert not any(path.suffix.lower() in ['.pt','.pth','.ply','.npy','.npz','.o','.sqlite'] for path in files)
manifest={str(path.relative_to(HERE)):dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in files}
manifest_path=HERE/'DELIVERY_MANIFEST.json'
manifest_path.write_text(json.dumps(dict(created_at=time.time(),files=manifest,
    excludes='原模型/数据集/共享环境/重建对象/冗余archive；A/B保持原位只读，路径及SHA见协议'),indent=2,ensure_ascii=False))
files.append(manifest_path)
with tarfile.open(archive,'w:gz',compresslevel=6) as target:
    for path in files:target.add(path,arcname=str(path.relative_to(HERE)),recursive=False)
receipt=dict(archive=archive.name,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    members=len(files),verified_result=verification,production_deployed=False,commit_push=False)
(HERE/'DELIVERY_RECEIPT.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False))
print(json.dumps(receipt,indent=2,ensure_ascii=False))

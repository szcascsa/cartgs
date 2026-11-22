from common import *
import tarfile,csv
assert read(ROOT/'status.json')['all_passed']and read(ROOT/'final_checks.json')['all_passed']
assert sha(read(ROOT/'input_manifest.json')['cameras'])==read(ROOT/'camera_checks.json')['camera_archive_sha256']
for path,val in read(ROOT/'frozen_sources.json').items():assert sha(path)==val['sha256']
checks=read(ROOT/'final_checks.json');checks['camera_archive_unchanged']=True
checks['report_sha256']=sha(ROOT/'FR1_XYZ_HIERARCHICAL_SAME_K_AUDIT.md');save(ROOT/'delivery_checks.json',checks)
files={}
for path in sorted(ROOT.rglob('*')):
    rel=path.relative_to(ROOT)
    if not path.is_file()or any(s in ['build','__pycache__']for s in rel.parts)or path.suffix=='.gz' or rel.name in ['bundle_manifest.json','controller.lock','pack.log']:continue
    if rel.name=='cameras.pt':continue
    files[str(rel)]=dict(bytes=path.stat().st_size,sha256=sha(path))
save(ROOT/'bundle_manifest.json',dict(files=files,large_camera_archive_not_bundled=True,camera_archive_SHA=read(ROOT/'camera_checks.json')['camera_archive_sha256']))
with tarfile.open(ROOT/'fr1_xyz_same_k_evidence.tar.gz','w:gz')as tar:
    for name in files:tar.add(ROOT/name,arcname=name,recursive=False)
    tar.add(ROOT/'bundle_manifest.json',arcname='bundle_manifest.json')
print(json.dumps(dict(files=len(files),bundle_bytes=(ROOT/'fr1_xyz_same_k_evidence.tar.gz').stat().st_size,bundle_sha256=sha(ROOT/'fr1_xyz_same_k_evidence.tar.gz'))),flush=True)

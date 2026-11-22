from pathlib import Path
import json,hashlib,tarfile
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(ROOT/'status.json')['all_passed']and read(ROOT/'final_checks.json')['all_passed']and read(ROOT/'delivery_checks.json')['all_passed']
for s in ['fr1_room','fr1_xyz','fr2_xyz']:
    pc=read(ROOT/s/'prior_checks.json')
    for path,v in pc['source_sha'].items():assert sha(path)==v['sha256'],path
    for path,digest in pc['artifact_sha'].items():assert sha(path)==digest,path
    assert sha(ROOT/'preregistration.json')==pc['preregistration_sha256']
files={}
for p in sorted(ROOT.rglob('*')):
    r=p.relative_to(ROOT)
    if not p.is_file()or any(v in ['build','__pycache__']for v in r.parts)or p.suffix=='.gz'or r.name in ['bundle_manifest.json','controller.lock','pack.log']:continue
    files[str(r)]=dict(bytes=p.stat().st_size,sha256=sha(p))
(ROOT/'bundle_manifest.json').write_text(json.dumps(dict(files=files,external_inputs='Master/TF/camera/GT remain at original SHA-verified server paths'),indent=2)+'\n')
with tarfile.open(ROOT/'d50_normalized_hierarchy_evidence.tar.gz','w:gz')as tar:
    for n in files:tar.add(ROOT/n,arcname=n,recursive=False)
    tar.add(ROOT/'bundle_manifest.json',arcname='bundle_manifest.json')
print(json.dumps(dict(files=len(files),bytes=(ROOT/'d50_normalized_hierarchy_evidence.tar.gz').stat().st_size,sha256=sha(ROOT/'d50_normalized_hierarchy_evidence.tar.gz'))),flush=True)

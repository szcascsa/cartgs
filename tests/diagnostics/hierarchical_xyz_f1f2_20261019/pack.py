from pathlib import Path
import json,hashlib,tarfile
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(ROOT/'status.json')['all_passed']and read(ROOT/'delivery_checks.json')['all_passed']
for scene in ['fr1_xyz','fr2_xyz']:
    out=ROOT/scene;assert read(out/'final_checks.json')['all_passed']
    for path,v in read(out/'frozen_sources.json').items():assert sha(path)==v['sha256'],path
    for path,v in read(out/'prior_artifact_checks.json')['files'].items():assert sha(path)==v['sha256'],path
files={}
for path in sorted(ROOT.rglob('*')):
    rel=path.relative_to(ROOT)
    if not path.is_file()or'__pycache__'in rel.parts or path.suffix=='.gz'or rel.name in ['bundle_manifest.json','controller.lock','pack.log']:continue
    files[str(rel)]=dict(bytes=path.stat().st_size,sha256=sha(path))
(ROOT/'bundle_manifest.json').write_text(json.dumps(dict(files=files,external_frozen_inputs='Master/TF/GT/camera archive remain at SHA-verified original server paths'),indent=2)+'\n')
with tarfile.open(ROOT/'xyz_f1_f2_mechanism_evidence.tar.gz','w:gz')as tar:
    for name in files:tar.add(ROOT/name,arcname=name,recursive=False)
    tar.add(ROOT/'bundle_manifest.json',arcname='bundle_manifest.json')
print(json.dumps(dict(files=len(files),bundle_bytes=(ROOT/'xyz_f1_f2_mechanism_evidence.tar.gz').stat().st_size,bundle_sha256=sha(ROOT/'xyz_f1_f2_mechanism_evidence.tar.gz'))),flush=True)

import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
with tarfile.open(ROOT/'geometry_marginal_validation_bundle.tar.gz','r:gz')as f:
    for m in f.getmembers():assert m.isfile()and(ROOT/m.name).resolve().is_relative_to(ROOT.resolve()),m.name
    f.extractall(ROOT,filter='data')
m=json.loads((ROOT/'bundle_manifest.json').read_text())
for name,v in m['files'].items():
    p=ROOT/name;assert p.stat().st_size==v['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'],name
out=dict(all_passed=True,files=len(m['files']),bundle_sha256=hashlib.sha256((ROOT/'geometry_marginal_validation_bundle.tar.gz').read_bytes()).hexdigest())
(ROOT/'local_delivery_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

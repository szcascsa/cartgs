from pathlib import Path
import json,hashlib,tarfile
ROOT=Path(__file__).resolve().parent
files=[]
for p in sorted(ROOT.rglob('*')):
    rel=p.relative_to(ROOT)
    if not p.is_file()or '__pycache__'in p.parts or 'runtime'in p.parts or p.name=='audit_worker'or p.suffix=='.gz'or p.name=='artifact_manifest.json':continue
    files.append(p)
manifest={str(p.relative_to(ROOT)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)for p in files}
(ROOT/'artifact_manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n')
with tarfile.open(ROOT/'internal_rep_utility_evidence.tar.gz','w:gz')as tf:
    for p in files+[ROOT/'artifact_manifest.json']:tf.add(p,arcname=str(p.relative_to(ROOT)))
print('PACKED',len(files),(ROOT/'internal_rep_utility_evidence.tar.gz').stat().st_size,flush=True)

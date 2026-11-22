from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'final_validation.json').read_text())['all_passed']
files=[]
for pat in ['*.py','*.json','*.csv','*.npz','*.bin','*.log','*.md','*.png','*.txt','audit_worker','native.cpp','structure_run1/*','structure_run2/*','runtime/lib/*']:
    files.extend(p for p in ROOT.glob(pat)if p.is_file())
files=sorted(set(files)-{ROOT/'artifact_manifest.json'})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'artifact_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
with tarfile.open(ROOT/'hierarchical_progressive_marginal_evidence.tar.gz','w:gz')as tf:
    for p in files+[ROOT/'artifact_manifest.json']:tf.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
print(json.dumps({'files':len(files),'bytes':(ROOT/'hierarchical_progressive_marginal_evidence.tar.gz').stat().st_size,'sha256':sha(ROOT/'hierarchical_progressive_marginal_evidence.tar.gz')},indent=2))

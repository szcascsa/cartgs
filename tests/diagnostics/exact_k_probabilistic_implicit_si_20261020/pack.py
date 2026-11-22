from pathlib import Path
import json,hashlib,tarfile
ROOT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
files=[]
for p in ROOT.rglob('*'):
    if not p.is_file():continue
    if any(n in p.relative_to(ROOT).parts for n in ['build','runtime','__pycache__','reproduced_tree','old_tree','legacy_models']):continue
    if p.name=='artifact_manifest.json'or p.suffix not in['.py','.cpp','.h','.txt','.json','.csv','.md','.npz','.pt','.log','.png']:continue
    files.append(p)
manifest={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size)for p in files};(ROOT/'artifact_manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n')
out=ROOT.parent/(ROOT.name+'_evidence.tar.gz')
with tarfile.open(out,'w:gz')as t:
    for p in files+[ROOT/'artifact_manifest.json']:t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
result=dict(path=str(out),sha256=sha(out),bytes=out.stat().st_size,files=len(files)+1);(ROOT/'pack_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

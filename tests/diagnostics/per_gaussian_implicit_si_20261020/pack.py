"""只封装本轮代码/数值证据；冻结大模型与运行库用SHA和原路径定位。"""
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
    parts=p.relative_to(ROOT).parts
    if any(x in parts for x in ['build','runtime','__pycache__','reproduced_tree','old_tree']):continue
    if p.name=='artifact_manifest.json'or p.suffix not in ['.py','.cpp','.txt','.json','.csv','.md','.npz','.pt','.log','.png']:continue
    files.append(p)
manifest={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size)for p in files}
(ROOT/'artifact_manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n')
out=ROOT.parent/(ROOT.name+'_evidence.tar.gz')
with tarfile.open(out,'w:gz')as t:
    for p in files+[ROOT/'artifact_manifest.json']:t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
print(json.dumps(dict(path=str(out),bytes=out.stat().st_size,sha256=sha(out),files=len(files)+1)),flush=True)

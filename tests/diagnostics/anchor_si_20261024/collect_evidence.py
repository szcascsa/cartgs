"""打包小型原始证据，模型和完整 checkpoint 留服务器不移动。"""
from pathlib import Path
import hashlib,json,zipfile

root=Path('/home/prmi/zwj/anchor-si-20261024')
folders=['validation','micro','regression','v1_row_order']
folders += ['frozen_'+a+'_'+str(s) for s in [2026,2027] for a in ['off','gaussian','anchor']]
folders += ['online_'+a+'_'+str(s) for s in [2026,2027] for a in ['off','anchor']]
paths=set(p for p in root.iterdir() if p.suffix in {'.csv','.json','.log'})
for folder in folders:
    paths.update(p for p in (root/folder).rglob('*') if p.is_file() and
        (p.suffix in {'.csv','.json','.log','.yaml'} or p.name=='anchor_si_state.pt'))
files={}
with zipfile.ZipFile(root/'final_evidence.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(paths):
        relative=p.relative_to(root).as_posix();data=p.read_bytes();archive.writestr(relative,data);files[relative]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
(root/'evidence_manifest.json').write_text(json.dumps(files,indent=2))
print('EVIDENCE',len(files),(root/'final_evidence.zip').stat().st_size)

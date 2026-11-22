"""按用户明确指示同步本地项目源码/配置；不删除服务器数据或额外文件。"""
import hashlib,json,shutil,tarfile
from pathlib import Path
R=Path('/home/prmi/zwj/mycartgs');ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=json.loads((ROOT/'code_sync_manifest.json').read_text());extract=ROOT/'synchronized_sources';extract.mkdir(exist_ok=True)
with tarfile.open(ROOT/'project_code.tar.gz','r:gz')as f:
    for m in f.getmembers():assert m.isfile()and(extract/m.name).resolve().is_relative_to(extract.resolve())
    f.extractall(extract)
changes=[]
for n,h in spec['files'].items():
    p=R/n;s=extract/n;assert sha(s)==h,n
    before=sha(p)if p.exists()else None
    if before==h:continue
    if p.exists():
        b=ROOT/'pre_sync_sources'/n;b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,b)
    p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,p);changes.append(dict(path=n,before=before,after=h))
for n,h in spec['files'].items():assert sha(R/n)==h,n
(ROOT/'code_sync_result.json').write_text(json.dumps(dict(all_passed=True,files_checked=len(spec['files']),changes=changes,deleted_files=0,policy='local project code/config authoritative; datasets/results/models/dependencies/build caches preserved'),indent=2))
print('SYNC_CODE_EXACT',len(spec['files']),len(changes),flush=True)

"""保留本轮正式模型与全部证据，认证信息不进入任何文件。"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    # 复用既有独立研究的完整依赖快照；不覆盖历史源码或数据。
    old=Path('/home/prmi/zwj/autonomous-compensation-20261027')
    dependency=ROOT/'dependencies';dependency.mkdir(exist_ok=True)
    for path in (old/'dependencies').rglob('*'):
        if path.is_file() and path.suffix in ['.py','.json']:
            target=dependency/path.relative_to(old/'dependencies');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    for path in ROOT.glob('fixture_*.json'):
        meta=json.loads(path.read_text());source=Path(meta['source']);target=ROOT/'frozen_fixtures'/path.stem
        target.mkdir(parents=True,exist_ok=True)
        for name in ['selected_sets.npz','compensation.npz','evaluation_manifest.json']:
            if (source/name).exists():shutil.copy2(source/name,target/name)
    code=ROOT/'code_versions';code.mkdir(exist_ok=True)
    for path in ROOT.glob('*.py'):
        target=code/(path.stem+'_'+sha(path)[:12]+'.py')
        if not target.exists():shutil.copy2(path,target)
    files=[];models=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:continue
        if path.name in ['artifact_manifest.json','archive_integrity.json'] or path.suffix in ['.gz','.tar']:continue
        if path.suffix=='.pt':continue
        if path.suffix in ['.py','.json','.csv','.md','.log','.txt','.npz','.npy','.png']:files.append(path)
    for path in (ROOT/'results').glob('*/*/*/*/formal/manifest.json'):
        p=Path(json.loads(path.read_text())['checkpoint']);assert p.is_relative_to(ROOT);models.append(p)
    models=sorted(set(models))
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        evidence=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files],
        final_models=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in models],
        scope='all formal endpoints retained; all short and intermediate checkpoints remain on server',
        teacher='read-only reference evidence; never candidate model parameters')
    (ROOT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    for filename,items in [('neural_v1_evidence.tar.gz',files+[ROOT/'artifact_manifest.json']),('neural_v1_models.tar.gz',models)]:
        with tarfile.open(ROOT/filename,'w:gz',compresslevel=3) as stream:
            for path in items:stream.add(path,arcname=str(path.relative_to(ROOT)))
    checks={name:dict(bytes=(ROOT/name).stat().st_size,sha256=sha(ROOT/name)) for name in ['neural_v1_evidence.tar.gz','neural_v1_models.tar.gz']}
    (ROOT/'archive_integrity.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print('ARCHIVED',len(files),'evidence',len(models),'formal models',json.dumps(checks),flush=True)

if __name__=='__main__':main()

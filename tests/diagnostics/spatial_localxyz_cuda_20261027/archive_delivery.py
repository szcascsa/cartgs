"""保留本轮源码、原始证据和模型；不包含认证信息或完整数据集。"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parent
REFERENCE=Path('/home/prmi/zwj/neural-v1-20261027')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    historical=json.loads((REFERENCE/'artifact_manifest.json').read_text())
    names=['main_results.csv','mechanism_comparisons.csv','NEURAL_V1_RESEARCH_REPORT.md','research_state.json']
    records=[dict(path=v['path'],sha256=sha(REFERENCE/v['path']),expected=v['sha256']) for v in historical['evidence'] if v['path'] in names]
    assert len(records)==4 and all(v['sha256']==v['expected'] for v in records)
    (ROOT/'historical_preservation.json').write_text(json.dumps(dict(all_match=True,records=records,incident='same-name report module collision regenerated two historical documents; original bytes restored and checked against pre-existing manifest; metrics unchanged; fixed with explicit importlib file path'),indent=2))
    snapshots=ROOT/'reference_sources';snapshots.mkdir(exist_ok=True)
    for name in ['neural.py','run.py','matrix_runtime.py','environment_final.json','fixture_fr1_xyz_R3_3100.json','fixture_fr2_xyz_R3_4100.json','scenes.json']:
        if (REFERENCE/name).exists():shutil.copy2(REFERENCE/name,snapshots/name)
    for scene in ['fr1_xyz','fr2_xyz']:
        src=REFERENCE/'results'/scene/'scene_manifest.json'
        if src.exists():shutil.copy2(src,snapshots/(scene+'_scene_manifest.json'))
    for src in (REFERENCE/'dependencies').rglob('*'):
        if src.is_file() and src.suffix in ['.py','.json']:
            dest=snapshots/'dependencies'/src.relative_to(REFERENCE/'dependencies');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    for scene,seed in [('fr1_xyz',3100),('fr2_xyz',4100)]:
        meta=json.loads((REFERENCE/f'fixture_{scene}_R3_{seed}.json').read_text());target=ROOT/'frozen_fixtures'/scene;target.mkdir(parents=True,exist_ok=True)
        for name in ['selected_sets.npz','compensation.npz','evaluation_manifest.json']:
            src=Path(meta['source'])/name
            if src.exists():shutil.copy2(src,target/name)
    target=ROOT/'reference_checkpoints';target.mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'spatial_localxyz_default.json').read_text())
    for scene,source in cfg['checkpoints'].items():shutil.copy2(source,target/(scene+'_step_12000.pt'))
    excluded={'artifact_manifest.json','archive_integrity.json','TASK_STATE.json','delivery_verification.json'}
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in excluded and p.suffix not in ['.gz','.tar']]
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),kind='model' if p.suffix=='.pt' else 'evidence') for p in files],production_changed=False,optimization_committed=False)
    (ROOT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    output={}
    for name,items in [('spatial_localxyz_cuda_evidence.tar.gz',[p for p in files if p.suffix!='.pt']+[ROOT/'artifact_manifest.json']),('spatial_localxyz_cuda_models.tar.gz',[p for p in files if p.suffix=='.pt'])]:
        with tarfile.open(ROOT/name,'w:gz',compresslevel=3) as tar:
            for p in items:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
        output[name]=dict(bytes=(ROOT/name).stat().st_size,sha256=sha(ROOT/name))
    (ROOT/'archive_integrity.json').write_text(json.dumps(output,indent=2));print('ARCHIVES',len(files),json.dumps(output),flush=True)


if __name__=='__main__':main()

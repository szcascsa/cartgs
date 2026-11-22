import json,hashlib,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(ROOT/'complete.json')['all_passed']and read(ROOT/'independent_verification.json')['all_passed']
manifest_path=ROOT/'geometry_marginal_manifest.json';provenance=read(manifest_path)
provenance['delivered_script_hashes']={p.name:sha(p)for p in ROOT.glob('*.py')}
provenance['script_hash_note']='executed_scripts is the reporting-time snapshot; delivered_script_hashes records final delivered helpers, including later packaging-only revisions'
manifest_path.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
external=[];files=[]
exclude={'source_after','pre_sync_sources','synchronized_sources','runtime_before','runtime','__pycache__'}
for p in ROOT.rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(ROOT)
    if any(part in exclude for part in rel.parts):continue
    if p.name in ['bundle_manifest.json','external_artifacts.json','geometry_marginal_validation_bundle.tar.gz','pack.log']:continue
    if p.suffix not in ['.py','.h','.cpp','.json','.csv','.npz','.bin','.md','.png','.log','.patch','.txt','.yaml','.yml','.toml']and p.name!='project_code.tar.gz':continue
    if any(part.endswith('_shutdown')for part in rel.parts)or p.name.startswith('marginal_snapshot_'):
        external.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size));continue
    # 大型final Master/TF/metadata与snapshot保留服务器并逐文件SHA引用；小型指标/源码完整交付。
    files.append(p)
for m in read(ROOT/'online_training_results.json'):
    ck=Path(m['checkpoint'])
    for p in [ck/'ply/selector.pt',ck/'ply/transform_field.pt',ck/'ply/point_cloud'/('iteration_'+ck.name.split('_')[0])/'point_cloud.ply',ck/'ply/point_cloud'/('iteration_'+ck.name.split('_')[0])/'selector_metadata.ply']:
        external.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
(ROOT/'external_artifacts.json').write_text(json.dumps(external,indent=2));files.append(ROOT/'external_artifacts.json')
manifest=dict(files={p.relative_to(ROOT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size)for p in sorted(files)},external_artifacts='external_artifacts.json; all retained on server, not deleted')
(ROOT/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(ROOT/'geometry_marginal_validation_bundle.tar.gz','w:gz')as f:
    for p in files+[ROOT/'bundle_manifest.json']:f.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
print('PACKED',len(files),(ROOT/'geometry_marginal_validation_bundle.tar.gz').stat().st_size,flush=True)

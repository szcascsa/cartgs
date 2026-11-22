"""只打包本轮证据/脚本/patch；保护原baseline与历史实验。"""
import hashlib
import json
from pathlib import Path
import tarfile
R=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022');S=R/'source'
old=Path('/home/prmi/zwj/private-compensation-online-20261022')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=json.loads((old/'source_manifest.json').read_text());unchanged={n:sha(old/'source'/n)==h for n,h in original.items()}
assert all(unchanged.values()),'Original production snapshot changed'
files=[]
for p in R.iterdir():
    if p.is_file() and p.suffix in ['.json','.csv','.py','.log','.sqlite','.nsys-rep']:files.append(p)
for name in ['phase1_final','phase1_alias_final','phase2_final','phase2_edge_final','phase3_final','legacy_a','legacy_b','adam','adam_decode','fused_a','fused_b',
             'profile_legacy','profile_fused','continuous_legacy_a','continuous_legacy_b','continuous_fused_a','continuous_fused_b',
             'fixed_replay_gate','scaling','build_final','reduction_v1_failure']:
    if (R/name).exists():files.extend(p for p in (R/name).rglob('*') if p.is_file())
files.extend(p for p in (R/'patch').rglob('*') if p.is_file())
for name in ['online_legacy_a','online_legacy_b','online_fused_a','online_fused_b']:
    d=R/name
    files.extend(p for p in d.iterdir() if p.is_file())
    files.extend(p for p in d.glob('*_shutdown/ply/*.json') if p.is_file())
    files.extend(p for p in d.glob('*_shutdown/ply/point_cloud/iteration_*/*state*.json') if p.is_file())
manifest={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(files))}
source_manifest={str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file() and not p.is_symlink() and
    any(str(p.relative_to(S)).startswith(prefix) for prefix in ['src/','include/','improvements/','cfg/selector/'])}
(R/'final_source_manifest.json').write_text(json.dumps(source_manifest,indent=2))
(R/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
(R/'delivery_validation.json').write_text(json.dumps(dict(original_source_unchanged=all(unchanged.values()),
    evidence_files=len(manifest),evidence_bytes=sum(x['bytes'] for x in manifest.values()),
    raw_checkpoints='preserved on server in each online_* run;source frozen checkpoint unchanged',
    no_default_enabled=True,git_operations='no stage,commit,push'),indent=2))
with tarfile.open(R/'delivery_evidence.tar.gz','w:gz',compresslevel=1) as tar:
    for p in sorted(set(files+[R/'evidence_manifest.json',R/'final_source_manifest.json',R/'delivery_validation.json'])):
        tar.add(p,arcname=str(p.relative_to(R)),recursive=False)
print('PASS EVIDENCE',len(manifest),(R/'delivery_evidence.tar.gz').stat().st_size,flush=True)

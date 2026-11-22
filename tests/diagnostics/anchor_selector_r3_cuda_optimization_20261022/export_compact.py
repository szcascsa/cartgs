"""本地交付紧凑证据，完整各分支Tensor快照仍保留在服务器完整包。"""
import hashlib
import json
from pathlib import Path
import tarfile
R=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022')
files=[]
for p in R.iterdir():
    if p.is_file() and p.suffix in ['.py','.json','.csv','.log','.sqlite','.nsys-rep']:files.append(p)
for name in ['patch','phase1_alias_final','phase2_edge_final','phase3_final','legacy_a','legacy_b','adam','adam_decode','fused_a','fused_b',
             'continuous_legacy_a','continuous_legacy_b','continuous_fused_a','continuous_fused_b','reduction_v1_failure']:
    if (R/name).exists():files.extend(p for p in (R/name).rglob('*') if p.is_file() and p.suffix!='.pt')
files.append(R/'fused_a/lod_0.050000_parity.pt')
for name in ['online_legacy_a','online_fused_a','online_fused_b','online_legacy_b']:
    files.extend(p for p in (R/name).iterdir() if p.is_file())
manifest={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(files))}
(R/'compact_manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(R/'compact_evidence.tar.gz','w:gz',compresslevel=1) as tar:
    for p in sorted(set(files+[R/'compact_manifest.json'])):tar.add(p,arcname=str(p.relative_to(R)),recursive=False)
print('PASS compact',len(manifest),(R/'compact_evidence.tar.gz').stat().st_size,flush=True)

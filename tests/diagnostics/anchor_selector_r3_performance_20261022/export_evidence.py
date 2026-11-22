"""仅打包本次审计allowlist，保留原始checkpoint和运行产物。"""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path('/home/prmi/zwj/anchor-selector-r3-performance-20261022')
files=[]
for name in ['online_runs.json','online_done.json','source_manifest.json','native_gpu_samples.json',
             'fixture_verification.json','scale_stress.csv','native_profile.sqlite','native_profile.nsys-rep',
             'build_commands.json','native_benchmark.cpp','run_online.py','run_native.py','stress_and_verify.py',
             'build_native.py','export_evidence.py','verification.log','online.log','rebuild.log']:
    if (ROOT/name).exists():files.append(ROOT/name)
for folder in ['trace1','trace2','untraced']:
    files.extend((ROOT/folder).glob('mapping_trace.csv'))
    files.extend((ROOT/folder).glob('training.log'))
    for pattern in ['*_shutdown/ply/mapping_stage_state.json','*_shutdown/ply/ultrafast_rotation_state.json',
                    '*_shutdown/ply/point_cloud/iteration_*/anchor_selector_state.json']:
        files.extend((ROOT/folder).glob(pattern))
for folder in ['native1','native2','nsight']:
    files.extend((ROOT/folder).glob('*.csv'));files.extend((ROOT/folder).glob('*.json'))
    if folder=='native1':files.extend((ROOT/folder).glob('*_parity.pt'))
manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
(ROOT/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(ROOT/'evidence.tar.gz','w:gz',compresslevel=2) as archive:
    for path in files+[ROOT/'evidence_manifest.json']:
        archive.add(path,arcname=str(path.relative_to(ROOT)))
print('EXPORTED',len(files),(ROOT/'evidence.tar.gz').stat().st_size,flush=True)

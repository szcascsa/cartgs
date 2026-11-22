"""核对本地上传的 source SHA、原源码/模型保护，导出本轮验证与窗口证据。"""
import hashlib
import json
from pathlib import Path
import tarfile
import subprocess
import sys

root=Path('/home/prmi/zwj/octree-anchor-pruning-20261021');repo=root/'source';out=root/'validation'
old=Path('/home/prmi/zwj/octree-explicit-20261021')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=json.loads((repo/'tests/anchor_selector/anchor_opacity_pruning_validation/source_manifest.json').read_text())
assert all(sha(repo/p)==h for p,h in source.items())
original=json.loads((old/'validation/source_manifest.json').read_text())
assert all(sha(old/'source'/p)==h and sha(Path('/home/prmi/zwj/mycartgs')/p)==h for p,h in original.items())
assert all(sha(repo/p)==source.get(p,h) for p,h in original.items())
models=json.loads((old/'validation/evaluation_plan.json').read_text())['model_sha256']
assert all(sha(p)==h for p,h in models.items())
gates=json.loads((out/'correctness.json').read_text());assert len(gates)==11 and all(v['passed'] for v in gates.values())
subprocess.run([sys.executable,str(repo/'tests/anchor_selector/anchor_opacity_pruning_validation/analyze_windows.py')],check=True,
    stdout=(out/'analysis.log').open('w'),stderr=subprocess.STDOUT)
(out/'source_manifest.json').write_text(json.dumps(source,indent=2))
run=root/'smoke_seed2026';cloud=run/'10000_shutdown/ply/point_cloud/iteration_10000'
selected=list(out.glob('*.json'))+list(out.glob('*.csv'))+list(out.glob('*.log'))+list(out.glob('*.npz'))
for device in ['cpu','cuda']:selected+=list((out/device).glob('*.pt'))
selected+=list((run/'anchor_pruning').glob('*'))
selected += [run/'training.log',cloud/'anchor_pruning_state.pt',cloud/'anchor_selector_state.json',
    run/'10000_shutdown/ply/selector_config.yaml']
selected=sorted(set(p for p in selected if p.is_file()))
files={str(p.relative_to(root)):sha(p) for p in selected}
(root/'delivery_verification.json').write_text(json.dumps(dict(baseline_commit='497463a664fa198dd1bec0be2f3124bd7f878899',
    source_sha_matched=len(source),original_source_and_production_unchanged=True,original_models_unchanged=True,
    correctness_gates=len(gates),native_binary_sha={str(p.relative_to(root)):sha(p) for p in
        [repo/'bin/tum_rgbd',repo/'bin/anchor_opacity_pruning_test',repo/'lib/libgaussian_mapper.so']},
    file_sha256=files),indent=2))
with tarfile.open(root/'pruning_evidence.tar.gz','w:gz') as t:
    for p in selected:t.add(p,arcname=str(p.relative_to(root)))
    t.add(root/'delivery_verification.json',arcname='delivery_verification.json')
(root/'delivery_complete.json').write_text(json.dumps(dict(archive=str(root/'pruning_evidence.tar.gz'),
    sha256=sha(root/'pruning_evidence.tar.gz')),indent=2))
print('EXPORTED',len(files),'files')

"""等待复跑结束，核对插桩等价性与原产物 SHA，打包只读证据。"""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import torch

root=Path('/home/prmi/zwj/octree-explicit-diagnostics-20261021')
old=Path('/home/prmi/zwj/octree-explicit-20261021')
out=root/'evidence'
deadline=time.monotonic()+600
while json.loads((out/'status.json').read_text())['stage']!='COMPLETED':
    assert time.monotonic()<deadline,'Replay did not complete'
    time.sleep(3)
subprocess.run([sys.executable,'/home/prmi/zwj/analyze_anchor_replay_20261021.py'],check=True)
parity={}
for device in ['cpu','cuda']:
    info=[]
    for name in ['explicit_selector.pt','explicit_fixture.pt']:
        a=torch.jit.load(str(old/'validation'/device/name),map_location='cpu')
        b=torch.jit.load(str(out/device/name),map_location='cpu')
        sa=dict(a.named_parameters());sa.update(dict(a.named_buffers()))
        sb=dict(b.named_parameters());sb.update(dict(b.named_buffers()))
        assert set(sa)==set(sb)
        different=[];max_abs=0.
        for key in sa:
            assert sa[key].shape==sb[key].shape
            exact=torch.equal(sa[key],sb[key])
            if not exact:
                different.append(key)
                max_abs=max(max_abs,float((sa[key]-sb[key]).abs().max()))
                assert device=='cuda' and torch.allclose(sa[key],sb[key],atol=2e-6,rtol=2e-5),(device,name,key,max_abs)
        info.append(dict(file=name,tensors=len(sa),exact_equal=not different,max_abs=max_abs,
                         differing_keys=different,allclose=True,atol=2e-6,rtol=2e-5))
    parity[device]=info
(out/'instrumentation_parity.json').write_text(json.dumps(parity,indent=2))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((out/'original_model_manifest.json').read_text())
s=json.loads((out/'original_source_manifest.json').read_text())
assert all(sha(p)==h for p,h in m.items())
assert all(sha(old/'source'/p)==h and sha(Path('/home/prmi/zwj/mycartgs')/p)==h for p,h in s.items())
patch=[]
for p in ['improvements/selector/anchor_bank.h','improvements/selector/explicit_anchor_blocks.h','src/gaussian_model.cpp']:
    patch+=difflib.unified_diff((old/'source'/p).read_text().splitlines(True),(root/'source'/p).read_text().splitlines(True),
        fromfile='original/'+p,tofile='diagnostic/'+p)
(out/'diagnostic_only.patch').write_text(''.join(patch))
for seed in [2026,2027]:
    run=root/f'diagnostic_seed{seed}'
    target=out/f'diagnostic_seed{seed}';target.mkdir(exist_ok=True)
    import shutil
    for name in ['lifecycle_trace.csv','training.log','input_selector.yaml']:
        shutil.copy2(run/name,target/name)
    mapping=run/'10000_shutdown/ply/mapping_stage_state.json'
    shutil.copy2(mapping,target/'mapping_stage_state.json')
frozen=root/'frozen_evidence'
files=[p for base in [out,frozen] for p in base.rglob('*') if p.is_file()]
hashes={str(p.relative_to(root)):sha(p) for p in files}
(root/'delivery_verification.json').write_text(json.dumps(dict(original_models_unchanged=True,
    original_source_unchanged=True,server_production_source_unchanged=True,unit_fixture_cpu_exact=True,
    unit_fixture_cuda_allclose=True,cuda_baseline_repeat_measured=True,
    file_sha256=hashes),indent=2))
with tarfile.open(root/'diagnostic_evidence.tar.gz','w:gz') as tar:
    for base in [out,frozen]:tar.add(base,arcname=base.name)
    tar.add(root/'delivery_verification.json',arcname='delivery_verification.json')
(root/'delivery_complete.json').write_text(json.dumps(dict(archive=str(root/'diagnostic_evidence.tar.gz'),
    sha256=sha(root/'diagnostic_evidence.tar.gz')),indent=2))
print('EVIDENCE_COMPLETE')

"""核验在线调用、默认值、源码与文件变更范围，写入部署身份。"""
from pathlib import Path
import datetime
import hashlib
import json
import re
import shutil

D = Path(__file__).resolve().parent
P = Path('/home/prmi/zwj/mycartgs')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
def save(name, value):
    (D/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

assert json.loads((D/'native_validation.json').read_text())['passed']
assert 'PACK_FORWARD_VJP_BITWISE_PASS checks=97' in (D/'local_gate.log').read_text()
identity = json.loads((D/'smoke_kernel_identity.json').read_text())
assert identity['passed']
cp = Path(identity['checkpoint'])
stage = json.loads((cp/'mapping_stage_state.json').read_text())
anchor = json.loads(next(cp.rglob('anchor_selector_state.json')).read_text())
assert anchor['compensation_method'] == 'spatial_offset3_residual48'
assert anchor['compensation_residual_mode'] == 1 and anchor['compensation_input_dim'] == 54
assert anchor['physical_scale_additive'] and anchor['lod_start_iteration'] == 1000
assert all(item['finite'] for item in anchor['compensation_feature_state'])
assert sum(stage[key]['mappings'] for key in ['online','tail']) == 1500
assert sum(stage[key]['lod_nonfinite'] for key in ['online','tail']) == 0
assert sum(stage[key]['lod_renders'] for key in ['online','tail']) > 0
exported = []
for path in cp.parent.rglob('*.yaml'):
    text = path.read_text()
    if all(re.search(r'^'+re.escape(key)+r':\s*1\s*$', text, re.M) for key in
           ['CUDA.fused_residual48_input','CUDA.fused_residual48_training_input']):
        exported.append(str(path))
assert exported, 'Missing effective C++ defaults in exported settings'
save('online_validation.json', dict(passed=True, checkpoint=str(cp), stage=stage, anchor_state=anchor,
     effective_settings=exported, training_forward_and_backward=identity['kernels'],
     full_quality_regression=False, end_to_end_speedup_claimed=False))
scope = json.loads((D/'source_manifest.json').read_text())
assert all(sha(P/name) == value for name, value in scope.items())
before = json.loads((D/'protected_before.json').read_text())
allowed = set(json.loads((D/'before.json').read_text()))
after = {name: sha(P/name) for name in before}
changed = [name for name in before if before[name] != after[name]]
unexpected = [name for name in changed if name not in allowed]
assert not unexpected, unexpected
save('scope_integrity.json', dict(passed=True, protected_files=len(before), changed=changed,
     unrelated_changed=unexpected, source_files=len(scope), source_all_match=True))
marker = P/'DEPLOYED_VERSION.json'
backup = json.loads((D/'before.json').read_text())
if 'DEPLOYED_VERSION.json' not in backup:
    backup['DEPLOYED_VERSION.json'] = sha(marker)
    if marker.is_file():
        shutil.copy2(marker, D/'before/DEPLOYED_VERSION.json')
    save('before.json', backup)
version = json.loads((D/'version.json').read_text())
version.update(deployed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    deployment=str(D), source_all_match=True, native_validation=True, online_smoke=True,
    training_fusion_default=True, native_atomic_scatter_nondeterministic=True)
marker.write_text(json.dumps(version, indent=2)+'\n')
save('deployment_complete.json', version)
print('TRAINING_FUSION_DEPLOYMENT_VERIFIED', version['commit'], flush=True)

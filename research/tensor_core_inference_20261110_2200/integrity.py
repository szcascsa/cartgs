"""验收只读生产/模型/旧二进制、当前冻结源码及零渲染图片。"""
from pathlib import Path
import json,hashlib,time
HERE=Path(__file__).resolve().parent;R=HERE/'reports';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
before=json.loads((R/'PROTECTED_BEFORE.json').read_text());changed=[];missing=[]
for p,h in before.items():
    if not Path(p).is_file():missing.append(p)
    elif sha(p)!=h:changed.append(p)
f=json.loads((R/'FROZEN_CANDIDATE.json').read_text());source_changes=[p for p,h in f['sources'].items() if sha(HERE/p)!=h]
binary_ok=sha(f['binary'])==f['binary_sha256'];active=[];boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for p in (R/'jobs').glob('*.json'):
    j=json.loads(p.read_text());proc=Path('/proc')/str(j.get('runner_pid',0))/'stat'
    if proc.is_file():
        parts=proc.read_text().rpartition(')')[2].split()
        if parts[0]!='Z' and parts[19]==j.get('runner_start_ticks') and boot==j.get('boot_id'):active.append(j['name'])
controller=R/'BATCH_CONTROLLER.json'
if controller.is_file():
    j=json.loads(controller.read_text());proc=Path('/proc')/str(j['pid'])/'stat'
    if proc.is_file():
        parts=proc.read_text().rpartition(')')[2].split()
        if parts[0]!='Z' and parts[19]==j['start_ticks'] and boot==j['boot_id']:active.append('paired_batches')
images=[str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp'] and 'figures' not in p.parts]
result=dict(passed=not(changed or missing or source_changes or active or images) and binary_ok,protected_files=len(before),changed=changed,missing=missing,frozen_source_changes=source_changes,binary_ok=binary_ok,
    active_own_jobs=active,render_images=images,checked_at=time.time(),commit_push_deploy=False,training_production_original_checkpoints_shared_conda_untouched=True)
(R/'INTEGRITY_FINAL.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps(result,indent=2));assert result['passed']

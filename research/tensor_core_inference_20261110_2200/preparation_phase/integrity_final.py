from pathlib import Path
import json,hashlib,time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;R=HERE/'reports'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protected=json.loads((ROOT/'reports/PROTECTED_BEFORE.json').read_text());changed=[];missing=[]
for path,expected in protected.items():
    if not Path(path).is_file():missing.append(path)
    elif sha(path)!=expected:changed.append(path)
frozen=json.loads((R/'FROZEN_JOINT_CANDIDATE.json').read_text())
source_changes=[name for name,h in frozen['sources'].items() if sha(HERE/name)!=h]
earlier=json.loads((ROOT/'reports/FROZEN_CANDIDATE.json').read_text())
old_changes=[name for name,h in earlier['sources'].items() if sha(ROOT/name)!=h]
assert sha(earlier['binary'])==earlier['binary_sha256']
assert sha(earlier['baseline'])==earlier['baseline_sha256']
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();active=[]
for base in [ROOT,HERE]:
    for path in (base/'reports/jobs').glob('*.json'):
        j=json.loads(path.read_text());proc=Path('/proc')/str(j.get('runner_pid',0))/'stat'
        if proc.is_file():
            fields=proc.read_text().rpartition(')')[2].split()
            if fields[0]!='Z' and fields[19]==j.get('runner_start_ticks') and boot==j.get('boot_id'):active.append(str(path))
for path in [ROOT/'reports/BATCH_CONTROLLER.json',R/'FORMAL_CONTROLLER.json',R/'COMPONENT_CONTROLLER.json']:
    j=json.loads(path.read_text());proc=Path('/proc')/str(j['pid'])/'stat'
    if proc.is_file():
        fields=proc.read_text().rpartition(')')[2].split()
        if fields[0]!='Z' and fields[19]==j['start_ticks'] and boot==j['boot_id']:active.append(str(path))
images=[str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp'] and 'figures' not in p.parts]
passed=not(changed or missing or source_changes or old_changes or active or images) and sha(frozen['binary'])==frozen['binary_sha256']
result=dict(passed=passed,protected_files=len(protected),changed=changed,missing=missing,frozen_source_changes=source_changes,
    previous_C_source_changes=old_changes,original_A_C_binary_unchanged=True,active_own_jobs=active,render_images=images,
    checked_at=time.time(),original_start_epoch=1794319200,total_budget_hours=9,
    elapsed_from_original_start_hours=(time.time()-1794319200)/3600,elapsed_preparation_hours=(time.time()-1794325857.4705095)/3600,
    production_training_checkpoints_shared_conda_untouched=True,commit_push_deploy=False)
(R/'INTEGRITY_FINAL.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2));assert passed

"""原生产/模型/上一版、本轮冻结源与二进制、零渲染图及活跃任务核验。"""
from common import *
before=json.loads((HERE/'reports/PROTECTED_BEFORE.json').read_text());changed=[];missing=[]
for name,value in before.items():
    if not Path(name).is_file():missing.append(name)
    elif sha(name)!=value:changed.append(name)
frozen=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());source_changes=[name for name,value in frozen['sources'].items() if sha(HERE/name)!=value]
binary_ok=sha(frozen['binary'])==frozen['binary_sha256'];active=[];boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for path in (HERE/'reports/jobs').glob('*.json'):
    d=json.loads(path.read_text());proc=Path('/proc')/str(d.get('runner_pid',0))/'stat'
    if proc.is_file():
        parts=proc.read_text().rpartition(')')[2].split()
        if parts[0]!='Z' and parts[19]==d.get('runner_start_ticks') and boot==d.get('boot_id'):active.append(d['name'])
images=[str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp'] and 'figures' not in p.parts]
result=dict(passed=not(changed or missing or source_changes or active or images) and binary_ok,checked_at=time.time(),protected_files=len(before),changed=changed,missing=missing,
    frozen_source_changes=source_changes,binary_ok=binary_ok,active_own_jobs=active,render_images=images,disk_free_bytes=__import__('shutil').disk_usage(HERE).free,
    production_training_shared_conda_untouched=True,commit_push_deploy=False)
save(HERE/'reports/INTEGRITY_FINAL.json',result);print(json.dumps(result,ensure_ascii=False,indent=2));assert result['passed']

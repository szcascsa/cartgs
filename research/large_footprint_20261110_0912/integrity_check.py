"""只读核验生产/训练/原checkpoint/上一轮证据和本轮冻结身份。"""
from pathlib import Path
import hashlib,json,time,shutil

HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()

protected=json.loads((HERE/'reports/PROTECTED_BEFORE.json').read_text());changed=[];missing=[]
for name,expected in protected.items():
    if not Path(name).is_file():missing.append(name)
    elif sha(name)!=expected:changed.append(name)
frozen=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text())
source_changes=[name for name,expected in frozen['sources'].items() if sha(HERE/name)!=expected]
binary_ok=sha(frozen['binary'])==frozen['binary_sha256'];active=[]
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for path in (HERE/'reports/jobs').glob('*.json'):
    identity=json.loads(path.read_text());proc=Path('/proc')/str(identity.get('runner_pid',0))/'stat'
    if proc.is_file():
        data=proc.read_text().rpartition(')')[2].split()
        if data[0]!='Z' and data[19]==identity.get('runner_start_ticks') and boot==identity.get('boot_id'):active.append(identity['name'])
images=[str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp'] and 'figures' not in p.relative_to(HERE).parts]
result=dict(passed=not(changed or missing or source_changes or active or images) and binary_ok,
    checked_at=time.time(),protected_files=len(protected),changed=changed,missing=missing,
    frozen_source_changes=source_changes,binary_ok=binary_ok,active_own_jobs=active,render_images=images,
    disk_free_bytes=shutil.disk_usage(HERE).free,commit_push_deploy=False)
(HERE/'reports/INTEGRITY_FINAL.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2));assert result['passed']

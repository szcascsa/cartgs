"""只读核对生产、原始推理二进制和参考源码。"""
from pathlib import Path
import hashlib,json

HERE=Path(__file__).resolve().parent
before=json.loads((HERE/'reports/PROTECTED_BEFORE.json').read_text())
after={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() if Path(name).is_file() else None for name in before}
changed=[name for name in before if before[name]!=after[name]]
(HERE/'reports/PROTECTED_AFTER.json').write_text(json.dumps(after,indent=2))
result=dict(passed=not changed,files=len(before),changed=changed)
(HERE/'reports/INTEGRITY.json').write_text(json.dumps(result,indent=2))
source={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in [HERE/'source'] for p in folder.rglob('*') if p.is_file()}
source.update({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
(HERE/'reports/SOURCE_MANIFEST.json').write_text(json.dumps(source,indent=2))
print(json.dumps(result,indent=2));assert not changed

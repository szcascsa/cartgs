"""仅恢复本部署 before.json 列出的文件，不触碰模型、数据和历史结果。"""
from pathlib import Path
import hashlib
import json
import shutil

D = Path(__file__).resolve().parent
P = Path('/home/prmi/zwj/mycartgs').resolve()
before = json.loads((D/'before.json').read_text())
for name, digest in before.items():
    target = P/name
    assert target.resolve().is_relative_to(P), name
    if digest is None:
        if target.is_file():
            target.unlink()
    else:
        source = D/'before'/name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, name
        shutil.copy2(source, target)
assert all((hashlib.sha256((P/name).read_bytes()).hexdigest() if (P/name).is_file() else None) == digest
           for name, digest in before.items())
print('ROLLBACK_COMPLETE')

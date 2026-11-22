"""冻结正式 A+B+C；本目录独立保存 Transport V2 实验。"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

D = Path(__file__).resolve().parent
ROOT = D.parents[2]
OLD = D.parent / 'change_distill_optimization_20261017'
stage = sys.argv[1]
out = D / stage
assert not out.exists(), out
out.mkdir()
(out / 'raw').mkdir()
manifest = json.loads((OLD / 'stage_c/raw/source_manifest.json').read_text())
manifest.update({p: {} for p in ['include/frozen_teacher_target_cache.h', 'include/selector_cuda.h']})
hashes = {}
for name in manifest:
    data = (ROOT / name).read_bytes()
    target = out / 'source' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    hashes[name] = {'sha256': hashlib.sha256(data).hexdigest(),
                    'normalized_sha256': hashlib.sha256(data.decode('utf-8').replace('\r\n', '\n').encode()).hexdigest()}
(out / 'raw/source_manifest.json').write_text(json.dumps(hashes, indent=2))
for name in ['benchmark.cpp', 'perf_scope.h']:
    shutil.copy2(OLD / 'stage_c' / name, out / name)
for cmd, name in [(['diff', '--binary'], 'working_tree.diff'),
                  (['diff', '--cached', '--binary'], 'staged.diff'),
                  (['status', '--porcelain=v1'], 'status.txt'), (['rev-parse', 'HEAD'], 'head.txt')]:
    (out / name).write_bytes(subprocess.check_output(['git', *cmd], cwd=ROOT))
print(out)

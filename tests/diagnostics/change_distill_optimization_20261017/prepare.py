"""保存输入工作树和逐阶段源码；不改写原审计。"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

D = Path(__file__).resolve().parent
ROOT = D.parents[2]
AUDIT = D.parent / 'change_distill_performance_20261017'
stage = sys.argv[1]
out = D / stage
out.mkdir(parents=True, exist_ok=True)
manifest = json.loads((AUDIT / 'raw/source_manifest.json').read_text())
hashes = {}
for name, original in manifest.items():
    data = (ROOT / name).read_bytes()
    hashes[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    if stage == 'baseline':
        assert hashes[name] == original, name
    target = out / 'source' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
(out / 'raw').mkdir(exist_ok=True)
(out / 'raw/source_manifest.json').write_text(json.dumps(hashes, indent=2))
for name in ['benchmark.cpp', 'perf_scope.h', 'generate_instrumentation.py', 'build_native.py']:
    shutil.copy2(AUDIT / name, out / name)
# Independent native-only rebuild; CUDA objects are reusable only while CUDA source hashes match.
build = (out / 'build_native.py').read_text(encoding='utf-8')
build = build.replace("versions=[('native', 'source'), ('instrumented', 'instrumented')]", "versions=[('native', 'source')]")
(out / 'build_native.py').write_text(build, encoding='utf-8', newline='\n')
for command, name in [(['diff', '--binary'], 'working_tree.diff'), (['diff', '--cached', '--binary'], 'staged.diff'),
                       (['status', '--porcelain=v1'], 'status.txt'), (['rev-parse', 'HEAD'], 'head.txt')]:
    (out / name).write_bytes(subprocess.check_output(['git', *command], cwd=ROOT))
print(out)

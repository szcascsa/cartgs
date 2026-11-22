"""只生成独立研究补丁与CUDA资源证据，不改原版。"""
from pathlib import Path
import difflib
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
BASE = Path('/home/prmi/zwj/experiments/compensator_distillation_20261108/source/python')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
changes, patch = [], []
for path in sorted((HERE / 'source').iterdir()):
    if not path.is_file():
        continue
    original = BASE / path.name
    old = original.read_text().splitlines(keepends=True) if original.is_file() else []
    new = path.read_text().splitlines(keepends=True)
    if old == new:
        continue
    changes.append(dict(file=path.name,
                        original_sha256=sha(original) if original.is_file() else None,
                        final_sha256=sha(path)))
    patch.extend(difflib.unified_diff(old, new,
                 fromfile='a/python/' + path.name if old else '/dev/null',
                 tofile='b/python/' + path.name))
target = HERE / 'reports/TENSOR_CORE_INFERENCE.patch'
target.write_text(''.join(patch), encoding='utf-8')
(HERE / 'reports/SOURCE_PATCH_METADATA.json').write_text(json.dumps(
    dict(baseline=str(BASE), changes=changes, patch_sha256=sha(target)), indent=2))
raw = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-resource-usage',
    str(HERE / 'build/mycartgs_tensor_deployment_v1.so')], text=True)
(HERE / 'reports/CUDA_RESOURCE_USAGE.txt').write_text(raw)
print('SOURCE_PATCH_AND_RESOURCES', len(changes), 'files', len(raw), 'resource chars')

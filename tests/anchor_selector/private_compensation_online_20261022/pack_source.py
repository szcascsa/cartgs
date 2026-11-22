"""打包明确源码allowlist及本轮测试；不包含无关实验产物或第三方依赖。"""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

base=Path(__file__).resolve().parent;repo=base.parents[2]
paths=subprocess.check_output(['git','ls-files','-z'],cwd=repo).decode().split('\0')
paths += ['improvements/selector/gaussian_compensation_bank.h','tests/anchor_selector/private_compensation_test.cpp',
          'tests/anchor_selector/private_compensation_python_test.py','tests/anchor_selector/private_compensation_mapping_test.cpp']
paths += [str(p.relative_to(repo)).replace('\\','/') for p in base.glob('*.py')]
allowed=[]
for name in sorted(set(paths)):
    if not (name=='CMakeLists.txt' or name.startswith(('src/','include/','improvements/','python/','cfg/','examples/','tests/','cuda_rasterizer/','viewer/'))):continue
    if Path(name).suffix.lower() not in {'.cpp','.h','.hpp','.c','.cc','.cu','.cuh','.inc','.py','.yaml','.yml','.txt','.cmake','.json','.glsl','.frag','.vert'}:continue
    if (repo/name).is_file():allowed.append(name)
manifest={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in allowed}
previous=json.loads((base/'source_manifest.json').read_text()) if (base/'source_manifest.json').exists() else {}
(base/'delta_paths.json').write_text(json.dumps([p for p,h in manifest.items() if previous.get(p)!=h],indent=2))
(base/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(base/'source.tar.gz','w:gz') as archive:
    for name in allowed:archive.add(repo/name,arcname=name)
print(json.dumps(dict(files=len(allowed),bytes=(base/'source.tar.gz').stat().st_size)))

"""把当前第一方源码打包到隔离构建目录；不包含生产结果或第三方依赖。"""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

dest = Path(__file__).resolve().parent
root = dest.parents[2]
names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
prefixes = ('include/', 'src/', 'improvements/', 'cuda_rasterizer/', 'viewer/', 'examples/', 'cfg/', 'python/', 'scripts/')
files = [n for n in names if n and (n == 'CMakeLists.txt' or n.startswith(prefixes) or
         (n.startswith('tests/') and not n.startswith('tests/diagnostics/')))]
files += [str(p.relative_to(root)).replace('\\', '/') for p in dest.glob('*') if p.suffix in ('.cpp', '.py', '.sh')]
files += ['tests/diagnostics/cuda_tf_implementation_20261013/online_driver.cmake',
          'tests/diagnostics/cuda_tf_implementation_20261013/online_tf_profiler.h']
manifest = {}
with tarfile.open(dest/'integration_source.tar.gz', 'w:gz') as archive:
    for name in sorted(set(files)):
        path = root/name
        if path.is_file():
            manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=name, recursive=False)
(dest/'integration_source_hashes.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
print('packaged_files', len(manifest))

"""当前分支的第一方源码交接包：不打包模型、二进制或第三方依赖。"""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

dest = Path(__file__).resolve().parent
root = dest.parents[2]
output = dest/'server_source_update'
output.mkdir(exist_ok=True)
names = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().strip('\0').split('\0'))
names.update(str(p.relative_to(root)).replace('\\', '/') for p in dest.iterdir() if p.is_file())
names.update(['CUDA_SELECTOR_REVISED_ACCEPTANCE_REPORT.md', 'SELECTOR_CUDA_USAGE.md'])
prefixes = ('include/', 'src/', 'improvements/', 'cuda_rasterizer/', 'viewer/', 'examples/', 'cfg/', 'python/', 'scripts/', 'tests/')
suffixes = {'.cpp', '.h', '.hpp', '.cuh', '.cu', '.c', '.py', '.sh', '.cmake', '.md', '.yaml', '.yml', '.in'}
selected = []
for name in sorted(names):
    path = root/name
    if not path.is_file() or name.startswith('third_party/'):
        continue
    if '/' not in name:
        include = name in ['CMakeLists.txt', 'build.sh'] or path.suffix == '.md'
    else:
        include = name.startswith(prefixes) and (path.suffix in suffixes or path.name == 'CMakeLists.txt')
        include |= name == 'tests/diagnostics/cuda_selector_revised_20261015/protocol.json'
    if include:
        selected.append(name)
manifest = dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    source='Current working tree including verified backward/memory changes and manual-test configuration', files={})
with tarfile.open(output/'source.tar.gz', 'w:gz') as archive:
    for name in selected:
        data = (root/name).read_bytes().replace(b'\r\n', b'\n')
        manifest['files'][name] = hashlib.sha256(data).hexdigest()
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mode = 0o755 if name.endswith('.sh') else 0o644
        archive.addfile(info, io.BytesIO(data))
(output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('source_files', len(selected), 'archive_bytes', (output/'source.tar.gz').stat().st_size)

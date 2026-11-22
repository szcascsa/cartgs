"""选择性打包本分支源码；不包含模型、历史实验或构建产物。"""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
allowed = ('include/', 'src/', 'improvements/', 'cfg/', 'python/', 'cuda_rasterizer/',
           'tests/anchor_selector/', 'third_party/tinyply/', 'research/global_view_joint_lod_20261105/')
files = sorted({name for name in names if name and (name.startswith(allowed) or name in {
    'CMakeLists.txt', 'tests/lod_health_guard_test.cpp', 'tests/real_lod_schedule_test.cpp'})
    and (ROOT/name).is_file() and not name.endswith(('.tar.gz', '.pyc', '.pt', '.ply'))
    and not name.endswith('/SOURCE_HASHES.json')
    and '__pycache__' not in name and '/evidence/' not in name})
manifest = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
(OUT/'SOURCE_HASHES.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
with tarfile.open(OUT/'source.tar.gz', 'w:gz') as archive:
    for name in files:
        archive.add(ROOT/name, arcname=name, recursive=False)
print(json.dumps({'files': len(files), 'archive_bytes': (OUT/'source.tar.gz').stat().st_size}))

from pathlib import Path
import hashlib
import json
import tarfile

D = Path(__file__).resolve().parent
paths = []
for stage in ['baseline', 'stage_a', 'stage_a_grouped', 'stage_b', 'stage_c']:
    root = D / stage
    for path in root.rglob('*'):
        if not path.is_file() or 'objects_abi1' in path.parts or path.suffix in ['.o', '.pyc']:
            continue
        if path.name.startswith('benchmark_') and path.suffix == '': continue
        paths.append(path)
for path in D.iterdir():
    if path.is_file() and path.suffix in ['.py', '.h', '.cpp', '.json', '.csv', '.diff', '.txt', '.md', '.log']:
        paths.append(path)
paths += list((D / 'tests').glob('*'))
paths.append(D / 'stage_c.tar.gz') # Initial C source before unused-signature cleanup.
manifest = {str(p.relative_to(D)): {'bytes': p.stat().st_size,
              'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths if p.name != 'artifact_manifest.json'}
(D / 'file_manifest.json').write_text(json.dumps(manifest, indent=2))
with tarfile.open(D / 'profiling_outputs.tar.gz', 'w:gz', compresslevel=5) as archive:
    for path in paths:
        if path.name != 'artifact_manifest.json': archive.add(path, arcname=str(path.relative_to(D)), recursive=False)
    archive.add(D / 'file_manifest.json', arcname='file_manifest.json')
path = D / 'profiling_outputs.tar.gz'
(D / 'artifact_manifest.json').write_text(json.dumps({'archive': path.name, 'bytes': path.stat().st_size,
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'files': len(manifest)}, indent=2))
print((D / 'artifact_manifest.json').read_text())

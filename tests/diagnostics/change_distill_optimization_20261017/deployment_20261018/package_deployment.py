from pathlib import Path
import hashlib
import json
import tarfile

D = Path(__file__).resolve().parent
paths = [p for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts
         and p.name not in ['deployment_evidence.tar.gz', 'deployment_archive_manifest.json', 'deployment_file_manifest.json']]
records = {str(p.relative_to(D)): {'bytes': p.stat().st_size,
    'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
(D / 'deployment_file_manifest.json').write_text(json.dumps(records, indent=2))
with tarfile.open(D / 'deployment_evidence.tar.gz', 'w:gz') as archive:
    for path in paths:
        archive.add(path, arcname=str(path.relative_to(D)), recursive=False)
    archive.add(D / 'deployment_file_manifest.json', arcname='deployment_file_manifest.json')
p = D / 'deployment_evidence.tar.gz'
(D / 'deployment_archive_manifest.json').write_text(json.dumps({'bytes': p.stat().st_size,
    'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'files': len(paths)}, indent=2))
print((D / 'deployment_archive_manifest.json').read_text())

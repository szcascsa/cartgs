from pathlib import Path
import hashlib
import json
import tarfile
D=Path(__file__).resolve().parent
files=[]
for p in D.rglob('*'):
    if not p.is_file():continue
    r=p.relative_to(D)
    if 'backup' in r.parts or 'cuda_objects' in r.parts or 'formal_libraries' in r.parts:continue
    if p.suffix in {'.a','.o','.zip','.gz'} or p.name.startswith('core'):continue
    if p.name.startswith('benchmark_') and not p.suffix:continue
    if p.name=='quaternion_probe':continue
    files.append(p)
archive=D/'raw_profiling_and_sources.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=5) as tar:
    for p in files:tar.add(p,arcname=str(p.relative_to(D)))
record={'archive':archive.name,'files':len(files),'bytes':archive.stat().st_size,
        'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'excluded':'executables, link objects, core dumps, binary backups; raw CSV/JSON/PT/Nsys and all source snapshots included'}
(D/'archive_manifest.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))

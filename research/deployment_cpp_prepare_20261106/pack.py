from pathlib import Path
import tarfile
r=Path(__file__).resolve().parent;p=r.parents[1]
with tarfile.open(r/'update.tar.gz','w:gz') as t:
    for f in [*p.joinpath('python').glob('deployment_*'),p/'python/DEPLOYMENT_RUNTIME.md',*r.glob('*.py')]:
        if f.is_file():t.add(f,arcname='source/'+f.relative_to(p).as_posix())

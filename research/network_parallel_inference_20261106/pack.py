from pathlib import Path
import tarfile
r=Path(__file__).resolve().parent;p=r.parents[1]
with tarfile.open(r/'update.tar.gz','w:gz') as t:
 for f in [p/'python/frozen_inference_cuda.py',p/'python/frozen_inference_bindings.cpp',p/'python/anchor_budget.py',*r.glob('*.py'),*r.glob('*.cpp'),*r.glob('*.cu')]:
  t.add(f,arcname='source/'+f.relative_to(p).as_posix())

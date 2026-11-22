from pathlib import Path
import tarfile
r=Path(__file__).resolve().parent;p=r.parents[1]
with tarfile.open(r/'update.tar.gz','w:gz') as t:
 for f in [*p.joinpath('python').glob('frozen_inference*'),p/'python/anchor_budget.py',p/'python/anchor_visibility.py',p/'python/budget_viewer.py',*r.glob('*.py')]:
  if f.is_file():t.add(f,arcname='source/'+str(f.relative_to(p)).replace('\\','/'))

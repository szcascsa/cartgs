"""等待已授权实验完成，顺序统计并导出证据；不接触生产参数。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path('/home/prmi/zwj/selector-capacity-matrix-20261026')
REPO=Path('/home/prmi/zwj/mycartgs')

def main():
    while not (ROOT/'matrix.exit').exists():time.sleep(20)
    code=int((ROOT/'matrix.exit').read_text())
    assert code==0,('matrix failed',code)
    while not (ROOT/'supplement.exit').exists():time.sleep(20)
    assert (ROOT/'supplement.exit').read_text().strip()=='0','Matched controls failed'
    status=json.loads((ROOT/'controller_status.json').read_text())
    assert status['stage']=='ALL_EXPERIMENTS_COMPLETE',status
    for script in ['source_audit.py','analyze.py','report.py']:
        with (ROOT/(script[:-3]+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    initial=json.loads((ROOT/'production_source_baseline.json').read_text())
    mismatches=[name for name,h in initial['source_sha256'].items() if not (REPO/name).is_file() or hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=h]
    assert not mismatches,('production source changed',mismatches)
    manifests={}
    for path in ROOT.glob('*.py'):manifests[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    from importlib.util import find_spec
    package=find_spec('diff_gaussian_rasterization')
    native=next(Path(package.origin).parent.glob('_C*.so'))
    info=dict(source_head=initial['head'],production_files_verified=len(initial['source_sha256']),production_mismatches=mismatches,
        prototype_sha256=manifests,renderer_binary=str(native),renderer_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        matrix_status=status,isolated_frozen_map=True,production_defaults_changed=False,staged=False,committed=False,pushed=False)
    (ROOT/'delivery_verification.json').write_text(json.dumps(info,indent=2))
    paths=[]
    for pattern in ['*.json','*.csv','*.md','*.py','*.npz','*.exit']:
        paths.extend(ROOT.glob(pattern))
    for folder in ['analysis','results']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and path.suffix in ['.json','.csv','.npy','.png']:paths.append(path)
    for name in ['matrix.log','source_audit.log','analyze.log','report.log']:paths.append(ROOT/name)
    # A reproducible final checkpoint for each tested independent-scene arm; all others remain on the server.
    paths.extend((ROOT/'results/fr2_xyz/external').glob('*/4100/step_12000.pt'))
    with tarfile.open(ROOT/'evidence.tar.gz','w:gz') as archive:
        for path in sorted(set(paths)):archive.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    print('DELIVERY_READY',str(ROOT/'evidence.tar.gz'),(ROOT/'evidence.tar.gz').stat().st_size,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'finalize.exit').write_text('1');raise
    (ROOT/'finalize.exit').write_text('0')

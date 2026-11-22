"""交付fixed-grid实验结果与完整模型；不打包旧地图/GT/evaluator二进制。"""
from pathlib import Path
import hashlib,json,shutil,tarfile
ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'correctness.json').read_text())['all_passed']
dependencies=ROOT/'dependencies';dependencies.mkdir(exist_ok=True)
source=ROOT.parent/'spatial-mlp-optimization-fix-20261020/source'
for filename in ['train_worker.cpp','calibrated_selection.h']:shutil.copy2(source/filename,dependencies/filename)
files=[p for p in ROOT.iterdir()if p.is_file()and p.name!='artifact_manifest.json'and p.suffix in ['.py','.cpp','.txt','.json','.csv','.md','.npz','.png','.log']]
files.extend(dependencies.iterdir())
for scene in ['fr1_room','fr1_xyz','fr2_xyz']:
    for filename in ['cell_mapping.npz','grid.bin','normalization.json','occupancy_search.csv','frozen_tensor_checks.json','worker.log']:
        if (ROOT/scene/filename).exists():files.append(ROOT/scene/filename)
    for case in (ROOT/scene/'models').iterdir():files.extend(p for p in case.iterdir()if p.is_file()and p.suffix in ['.pt','.npz','.json','.log'])
for case in (ROOT/'lr_sanity').iterdir():files.extend(p for p in case.iterdir()if p.is_file()and p.suffix in ['.pt','.npz','.json','.log'])
manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in files}
(ROOT/'artifact_manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n');files.append(ROOT/'artifact_manifest.json')
with tarfile.open(ROOT/'delivery.tar.gz','w:gz',compresslevel=1)as archive:
    for p in sorted(files):archive.add(p,arcname=p.relative_to(ROOT),recursive=False)
print(json.dumps(dict(files=len(files),bytes=(ROOT/'delivery.tar.gz').stat().st_size)))

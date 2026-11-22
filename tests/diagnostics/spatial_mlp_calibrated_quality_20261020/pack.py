"""仅打包本轮明确列出的结果、模型与源文件，不包含旧地图或相机GT。"""
from pathlib import Path
import hashlib,json,shutil,tarfile
ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'correctness.json').read_text())['all_passed']
source=ROOT/'source';source.mkdir(exist_ok=True)
for name in ['train_worker.cpp','calibrated_selection.h']:
    shutil.copy2(ROOT.parent/'spatial-mlp-optimization-fix-20261020/source'/name,source/name)
files=[p for p in ROOT.iterdir()if p.is_file()and p.name!='artifact_manifest.json'and p.suffix in ['.csv','.json','.py','.md','.npz','.png','.log']]
files.extend(source.iterdir())
for scene in ['fr1_room','fr1_xyz','fr2_xyz']:
    for case in (ROOT/scene/'models').iterdir():
        files.extend(p for p in case.iterdir()if p.is_file()and p.suffix in ['.pt','.npz','.json','.log'])
    for name in ['frozen_tensor_checks.json','worker.log']:
        if (ROOT/scene/name).exists():files.append(ROOT/scene/name)
manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in files}
(ROOT/'artifact_manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n');files.append(ROOT/'artifact_manifest.json')
with tarfile.open(ROOT/'delivery.tar.gz','w:gz',compresslevel=1)as archive:
    for p in sorted(files):archive.add(p,arcname=p.relative_to(ROOT),recursive=False)
print(json.dumps(dict(files=len(files),archive_bytes=(ROOT/'delivery.tar.gz').stat().st_size)))

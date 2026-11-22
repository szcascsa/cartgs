"""只打包本轮明确的回归/质量结果，不携带旧地图与GT。"""
from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'correctness.json').read_text())['all_passed']
files=[p for p in ROOT.iterdir()if p.is_file()and p.name!='artifact_manifest.json'and p.suffix in ['.py','.json','.csv','.md','.npz','.png','.log']]
for scene in ['fr1_room','fr1_xyz','fr2_xyz']:
    for case in (ROOT/scene/'models').iterdir():files.extend(p for p in case.iterdir()if p.is_file())
    for name in ['worker.log','frozen_tensor_checks.json']:
        if (ROOT/scene/name).exists():files.append(ROOT/scene/name)
manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in files}
(ROOT/'artifact_manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n');files.append(ROOT/'artifact_manifest.json')
with tarfile.open(ROOT/'delivery.tar.gz','w:gz',compresslevel=1)as archive:
    for p in sorted(files):archive.add(p,arcname=p.relative_to(ROOT),recursive=False)
print(json.dumps(dict(files=len(files),bytes=(ROOT/'delivery.tar.gz').stat().st_size)))

"""打包Cached推理结果，不复制旧模型、GT或renderer二进制。"""
from pathlib import Path
import hashlib,json,shutil,tarfile
ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'correctness.json').read_text())['all_passed']
assert (ROOT/'GRID_MLP_CACHED_BERNOULLI_INFERENCE_AUDIT.md').exists()
dependencies=ROOT/'dependencies';dependencies.mkdir(exist_ok=True)
shutil.copy2(ROOT.parent/'spatial-mlp-optimization-fix-20261020/source/calibrated_selection.h',dependencies/'calibrated_selection.h')
shutil.copy2(ROOT.parent/'mycartgs/include/selector_handoff.h',dependencies/'selector_handoff.h')
files=[p for p in ROOT.rglob('*')if p.is_file()and 'build'not in p.relative_to(ROOT).parts and p.name!='artifact_manifest.json'and p.suffix in ['.py','.cpp','.h','.txt','.json','.csv','.npz','.png','.md','.log']]
manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in files}
(ROOT/'artifact_manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n');files.append(ROOT/'artifact_manifest.json')
with tarfile.open(ROOT/'delivery.tar.gz','w:gz',compresslevel=1)as archive:
    for p in sorted(files):archive.add(p,arcname=p.relative_to(ROOT),recursive=False)
print(json.dumps(dict(files=len(files),bytes=(ROOT/'delivery.tar.gz').stat().st_size,sha256=hashlib.sha256((ROOT/'delivery.tar.gz').read_bytes()).hexdigest())))

"""验证并解包本次独立实验结果；不触碰生产代码。"""
import csv
import hashlib
import json
import math
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
archive = ROOT / 'evidence.tar.gz'
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
assert digest == '0ca07cd8a700825c09fa6e48f00d345967ec7d51c49dbfbe67dd2a4f670faa46'
destination = ROOT / 'evidence'
destination.mkdir(exist_ok=True)
with tarfile.open(archive) as pack:
    for member in pack.getmembers():
        assert member.isfile() and not member.issym() and not member.islnk()
        target = (destination / member.name).resolve()
        assert target.is_relative_to(destination.resolve()) and not Path(member.name).is_absolute()
    pack.extractall(destination)
models = sorted((destination / 'results/fr1_xyz').glob('*/*/complete.json'))
assert len(models) == 4
for model in models:
    info = json.loads(model.read_text())
    assert info['steps'] == 12000 and info['master_frozen']
    assert all(value == 3000 for value in info['ratio_exposures'].values())
with (destination / 'analysis/quality_per_view.csv').open() as stream:
    rows = list(csv.DictReader(stream))
assert all(math.isfinite(float(row[key])) for row in rows for key in ['psnr', 'ssim', 'lpips', 'photo'])
production = json.loads((ROOT / 'production_verification.json').read_text(encoding='utf-8-sig'))
assert not production['production_mismatches'] and not production['staged_files']
result = dict(archive_sha256=digest, new_models=len(models), production=production,
    new_online_slam=False, scene='fr1_xyz', seed=3100, steps=12000, exposure_per_ratio=3000,
    finite_quality_rows=len(rows), all_four_metrics_finite=True)
(ROOT / 'delivery_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))

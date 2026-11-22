"""本地交付验收：ZIP/文件哈希、逐帧行数、独立复算ratio汇总。"""
from pathlib import Path
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    delivery = json.loads((ROOT/'DELIVERY_VERIFIED.json').read_text())
    archive = ROOT/delivery['archive']
    assert archive.stat().st_size==delivery['archive_bytes'] and sha(archive)==delivery['sha256']
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (ROOT/member.filename).resolve()
            assert target.is_relative_to(ROOT.resolve()), member.filename
            assert not member.is_dir()
        bundle.extractall(ROOT)
    hashes = json.loads((ROOT/'DELIVERY_SHA256.json').read_text())
    assert all(sha(ROOT/name)==value for name,value in hashes.items())
    validation = json.loads((ROOT/'reports/VALIDATION.json').read_text())
    assert validation['passed'] and validation['workload_rows']==714084 and validation['views']==59507
    assert sha(ROOT/'reports/PER_FRAME_WORKLOAD.csv.gz')==validation['raw_data_sha256']
    totals = defaultdict(lambda: [0,0.,0.])
    rows = 0
    with gzip.open(ROOT/'reports/PER_FRAME_WORKLOAD.csv.gz','rt',encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['scene'],row['method']
            value = totals[key]
            value[0] += 1
            value[1] += float(row['relative_active_pixel_tests'])
            value[2] += float(row['relative_effective_alpha_blends'])
            assert row['diagnostic_RGB_bitwise']=='True'
            rows += 1
    assert rows==714084 and len(totals)==384
    with (ROOT/'reports/SCENE_RATIO_WORKLOAD.csv').open(encoding='utf-8',newline='') as stream:
        scenes = list(csv.DictReader(stream))
    for row in scenes:
        count,tests,blends = totals[row['scene'],row['method']]
        assert count==int(row['frames'])
        assert math.isclose(tests/count,float(row['relative_active_pixel_tests_mean']),rel_tol=1e-12)
        assert math.isclose(blends/count,float(row['relative_effective_alpha_blends_mean']),rel_tol=1e-12)
    with (ROOT/'reports/SIX_CURVE_DATA.csv').open(encoding='utf-8',newline='') as stream:
        groups = list(csv.DictReader(stream))
    assert len(groups)==72
    for group in groups:
        subset = [s for s in scenes if (s['dataset'],s['mode'],s['method'])==(group['dataset'],group['mode'],group['method'])]
        assert len(subset)==int(group['scenes'])
        for key in ['relative_active_pixel_tests_mean','relative_effective_alpha_blends_mean']:
            assert math.isclose(sum(float(s[key]) for s in subset)/len(subset),float(group[key]),rel_tol=1e-12)
    result = dict(passed=True,archive_sha256=sha(archive),files_verified=len(hashes),
        raw_rows=rows,scene_ratio_groups=len(totals),plot_groups=len(groups),independent_aggregation_passed=True)
    (ROOT/'LOCAL_DELIVERY_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()

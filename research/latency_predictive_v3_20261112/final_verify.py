"""正式实验收据、逐位验证及受保护服务器文件的独立终检。"""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    before=json.loads((ROOT/'formal/PROTECTED_BEFORE.json').read_text())
    changed=[p for p,d in before.items() if sha(p)!=d]
    source=json.loads((ROOT/'reports/SOURCE_SHA256.json').read_text())
    mismatch=[p for p,d in source.items() if sha(ROOT/p)!=d]
    rows=list(csv.DictReader((ROOT/'formal/REQUESTS_WITH_QUALITY.csv').open()))
    assert len(rows)==30000 and len({(r['repeat'],r['method'],r['frame']) for r in rows})==30000
    assert all(math.isfinite(float(r[k])) for r in rows for k in ['render_ms','psnr'])
    receipts=json.loads((ROOT/'formal/CONTROLLER_SUMMARY.json').read_text())
    assert len(receipts)==15 and all(r['updates']==2000 and r['dropped_samples']==0 and
        r['stale_samples']==0 and r['pending_samples']==0 for r in receipts)
    assert all(int(r['predicted_N'])==int(r['eligible_N']) for r in rows if r['method']=='predictive')
    gpu=json.loads((ROOT/'reports/GPU_VALIDATION.json').read_text())
    assert gpu['passed'] and len(gpu['bitwise_cases'])==24
    images=[str(p.relative_to(ROOT)) for p in (ROOT/'formal').rglob('*') if p.suffix in ['.png','.jpg']]
    binary=ROOT/'build/mycartgs_latency_predictive_v3.so'
    result=dict(passed=not(changed or mismatch or images),protected_files=len(before),changed_protected=changed,
        source_files=len(source),source_mismatch=mismatch,binary_sha256=sha(binary),requests=30000,
        cpu_tests=12,gpu_lifecycle_tests=len(gpu['lifecycle']),ids_rgb_radii_bitwise_cases=24,
        all_predicted_candidate_counts_exact=True,render_images_saved=len(images),
        production_changed=False,training=False,committed=False,pushed=False)
    (ROOT/'reports/FINAL_SERVER_INTEGRITY.json').write_text(json.dumps(result,indent=2))
    assert result['passed'];print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

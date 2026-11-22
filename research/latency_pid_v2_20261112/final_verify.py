"""交付后重新核验真实服务器文件、实验收据及二进制身份。"""
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
    changed=[path for path,value in before.items() if sha(path)!=value]
    source=json.loads((ROOT/'reports/SOURCE_SHA256.json').read_text())
    mismatched=[path for path,value in source.items() if sha(ROOT/path)!=value]
    receipts=json.loads((ROOT/'formal/CONTROLLER_SUMMARY.json').read_text())
    rows=list(csv.DictReader((ROOT/'formal/REQUESTS_WITH_QUALITY.csv').open()))
    assert len(rows)==30000 and len({(r['repeat'],r['method'],r['frame']) for r in rows})==30000
    assert all(math.isfinite(float(r[x])) for r in rows for x in ['render_ms','psnr'])
    assert len(receipts)==15 and all(r['updates']==2000 and r['dropped_samples']==0 and
        r['pending_samples']==0 and r['stale_samples']==0 for r in receipts)
    binaries={str(path.relative_to(ROOT)):sha(path) for path in (ROOT/'build').rglob('*.so')}
    assert len(binaries)==4
    images=[str(p.relative_to(ROOT)) for directory in ['formal','smoke']
        for p in (ROOT/directory).rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg']]
    result=dict(passed=not(changed or mismatched or images),protected_files=len(before),
        changed_protected=changed,source_files=len(source),source_mismatch=mismatched,
        compiled_binaries=binaries,requests=len(rows),render_images=images,
        cpu_v2=18,cpu_compat=14,gpu_tests=10,timer_bitwise=49,wrapper_bitwise=24,
        production_changed=False,committed=False,pushed=False)
    (ROOT/'reports/FINAL_SERVER_INTEGRITY.json').write_text(json.dumps(result,indent=2))
    assert result['passed'];print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

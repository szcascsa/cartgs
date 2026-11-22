from pathlib import Path
import json,hashlib,csv
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
checks=[]
for s in ['fr1_room','fr1_xyz','fr2_xyz']:
    p=ROOT/s;prior=read(p/'prior_checks.json');info=read(p/'input_manifest.json')
    for name,a in prior['source_sha'].items():assert sha(name)==a['sha256']
    for name,h in prior['previous_artifacts'].items():assert sha(name)==h
    assert sha(info['cameras'])==prior['camera_sha256']
    for lib,h in prior['runtime_sha'].items():assert sha(p/'runtime/lib'/lib)==h
    tensor=read(p/'frozen_tensor_checks.json');assert tensor['ok']and tensor['maturity_all_one']
    for filename in ['spatial_coverage.csv','redundancy_statistics.csv']:
        a=list(csv.DictReader(open(p/filename)))
        for r in [.01,.05,.1,.15]:
            baseline=next(v for v in a if v['arm']=='C'and float(v['ratio'])==r);copy=next(v for v in a if v['arm']=='RF-C'and float(v['ratio'])==r)
            for k in baseline:
                if k!='arm':assert baseline[k]==copy[k]
    checks.append(dict(scene=s,original_sources=len(prior['source_sha']),previous_artifacts=len(prior['previous_artifacts']),camera_archive_unchanged=True,runtime_sha_unchanged=True,tensors=tensor,RF_C_spatial_redundancy_exact=True))
assert read(ROOT/'status.json')['all_passed']and read(ROOT/'correctness.json')['all_passed']
(ROOT/'final_input_checks.json').write_text(json.dumps(dict(all_passed=True,scenes=checks,frozen_static_hierarchy_library_sha256=sha('/home/prmi/zwj/sparse-hash-root-grid-20261019/build/libhierarchy.a')),indent=2)+'\n')
print('FINAL_FROZEN_CHECKS_PASS',checks,flush=True)

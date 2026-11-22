"""本地从原始数组复算质量、配对统计、same-K 和旧基线复用，检查文件哈希。"""
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np

root=Path(__file__).resolve().parent;repo=root.parents[2];evidence=root/'evidence'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
table=lambda p:list(csv.DictReader(p.open(newline='')))
assert sha(root/'evidence.tar.gz')==read(root/'archive_info.json')['sha256']
with tarfile.open(root/'evidence.tar.gz') as archive:archive.extractall(evidence,filter='data')
manifest=read(evidence/'evidence_manifest.json')
assert all(sha(evidence/p)==h for p,h in manifest.items())
source=read(evidence/'source_manifest.json')
assert all(sha(repo/p)==h for p,h in source.items())
for name in ['build_validate.py','train.py','evaluate.py','collect.py','verify_v1.py']:
    assert sha(root/name)==sha(evidence/name),name
out=evidence/'validation';plan=read(out/'evaluation_plan.json')
assert len(plan['frames'])==len(set(plan['frames']))==64
summary=table(out/'quality_summary.csv');views=table(out/'quality_per_view.csv');paired=table(out/'paired_statistics.csv')
assert (len(summary),len(views),len(paired))==(22,1408,52)
cache={}
for row in summary:
    key=row['method'],row['policy'],row['level'];folder=evidence/'quality'/Path(*key)
    assert read(folder/'evaluation_frames.json')['gt_indices']==plan['frames']
    data=[v for v in views if (v['method'],v['policy'],v['level'])==key]
    assert [int(v['gt_index']) for v in data]==plan['frames']
    if row['level']!='Full':
        ids=np.load(folder/'selected_rows.npz')['rows']
        assert len(ids)==len(np.unique(ids))==int(row['selected'])
        assert ids.min()>=0 and ids.max()<int(row['total'])
        if row['policy']=='same_k':
            assert int(row['selected'])==int(row['level'][1:])
            assert np.isclose(float(row['ratio']),int(row['selected'])/int(row['total']),atol=1e-15,rtol=0)
            assert .01<=float(row['ratio'])<=.15
    cache[key]={}
    for metric in ['psnr','ssim','lpips','photo']:
        raw=np.loadtxt(folder/(metric+'.txt'));assert raw.shape==(64,) and np.isfinite(raw).all()
        assert np.array_equal(raw,[float(v[metric]) for v in data])
        assert np.isclose(raw.mean(),float(row[metric]),atol=1e-12,rtol=0)
        cache[key][metric]=raw
for level in ['1','5','10','15']:
    load=lambda name:np.load(evidence/'quality'/name/'cached_bernoulli'/level/'selected_rows.npz')['rows']
    assert np.array_equal(load('appearance_v2'),load('appearance_off'))
for row in paired:
    left='appearance_off' if row['comparison'].endswith('appearance_off') else 'geometry_v1'
    key=lambda name:(name,row['policy'],row['level'])
    d=cache[key('appearance_v2')][row['metric']]-cache[key(left)][row['metric']]
    sign=1 if row['metric'] in ['psnr','ssim'] else -1
    expected=dict(mean=d.mean(),median=np.median(d),p10=np.quantile(d,.1),p90=np.quantile(d,.9),improved_fraction=(sign*d>0).mean())
    assert all(np.isclose(v,float(row[k]),atol=1e-12,rtol=0) for k,v in expected.items())
old=repo/'tests/anchor_selector/shared_compensation_validation/evidence'
reused=read(out/'reused_baseline.json')
for destination,entry in reused.items():
    source_suffix=entry['source'].split('/shared-selector-compensation-20261021/',1)[1]
    dest_suffix=destination.split('/shared-selector-appearance-20261021/',1)[1]
    assert sha(old/source_suffix)==entry['sha256']==sha(evidence/dest_suffix)
protocol=read(out/'training_protocol.json')
assert protocol['training_runs']==1 and protocol['no_tuning'] and read(out/'audit_summary.json')['passed']
assert read(out/'correctness.json')['passed']
result=dict(passed=True,files_sha_verified=len(manifest),source_sha_verified=len(source),reused_baseline_files=len(reused),
    arms=len(summary),per_view_rows=len(views),paired_rows=len(paired),same_selected_rows_exact=True)
(root/'local_delivery_checks.json').write_text(json.dumps(result,indent=2));print(result)

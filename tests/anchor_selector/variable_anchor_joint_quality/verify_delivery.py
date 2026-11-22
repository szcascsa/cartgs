"""本地重算所有逐视图指标和配对统计，验证既有固定基线的复用。"""
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np

root = Path(__file__).resolve().parent
repo = root.parents[2]
evidence = root/'evidence'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
table = lambda p: list(csv.DictReader(p.open(encoding='utf-8',newline='')))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'evidence.tar.gz') == read(root/'archive_info.json')['sha256']
with tarfile.open(root/'evidence.tar.gz') as archive:
    archive.extractall(evidence,filter='data')
manifest = read(evidence/'evidence_manifest.json')
assert all(sha(evidence/p) == h for p,h in manifest.items())
validation = evidence/'validation'
protocol = read(validation/'training_protocol.json')
assert all(sha(repo/p) == h for p,h in protocol['source_sha256'].items())
for script in ['train.py','evaluate.py','collect.py']:
    assert sha(root/script) == sha(evidence/script)
plan = read(validation/'evaluation_plan.json')
assert plan['population'] == dict(fixed=42640,variable=7099)
assert plan['common_k'] == [427,639,852,1064] and not plan['serial_joint']
assert len(plan['frames']) == len(set(plan['frames'])) == 64
amendment = read(validation/'protocol_amendment.json')
assert amendment['baseline'] == '/home/prmi/zwj/anchor-slot-deactivation-20261021/enabled_seed2026'
reused = read(validation/'reused_fixed_evidence.json')
for path, info in reused.items():
    suffix = info['source'].split('/serial-joint-lod-20261021/',1)[1]
    old = repo/'tests/anchor_selector/serial_joint_lod_validation/evidence'/suffix
    assert sha(old) == info['sha256'] == sha(evidence/path)
for name in ['fixed','variable']:
    state = read(validation/(name+'_ultrafast_rotation_state.json'))
    assert not state.get('serial_full_lod',False) and state.get('post_topology_full_steps',0) == 0
    assert state['ultrafast_total_normal_steps'] == 10000
base = repo/'tests/anchor_selector/slot_deactivation_validation/online_evidence/enabled_seed2026/input_selector.yaml'
expected = base.read_text()+'\nTraining.serial_full_lod: 0\nTraining.serial_lod_detach_master: 0\nTraining.post_topology_full_steps: 0\nSelector.variable_anchor_members: 1\n'
assert (evidence/'variable_seed2026/input_selector.yaml').read_text() == expected
summary = table(validation/'quality_summary.csv')
views = table(validation/'quality_per_view.csv')
paired = table(validation/'paired_statistics.csv')
assert len(summary) == 18 and len(views) == 1152 and len(paired) == 36
cache = {}
for row in summary:
    key = row['method'],row['policy'],row['level']
    folder = evidence/'quality'/Path(*key)
    assert read(folder/'evaluation_frames.json')['gt_indices'] == plan['frames']
    data = [v for v in views if (v['method'],v['policy'],v['level']) == key]
    assert [int(v['gt_index']) for v in data] == plan['frames']
    selected = int(row['selected'])
    if row['level'] == 'Full':
        assert selected == int(row['total'])
    else:
        ids = np.load(folder/'selected_rows.npz')['rows']
        assert len(ids) == len(np.unique(ids)) == selected and ids.min() >= 0 and ids.max() < int(row['total'])
        if row['policy'] == 'same_k':
            assert selected == int(row['level'][1:])
            assert np.isclose(float(row['ratio']),selected/int(row['total']),atol=1e-15,rtol=0)
            assert .01 <= float(row['ratio']) <= .15
    cache[key] = {}
    for metric in ['psnr','ssim','lpips','photo']:
        raw = np.loadtxt(folder/(metric+'.txt'))
        assert raw.shape == (64,) and np.isfinite(raw).all()
        assert np.array_equal(raw,[float(v[metric]) for v in data])
        assert np.isclose(raw.mean(),float(row[metric]),atol=1e-12,rtol=0)
        cache[key][metric] = raw
for row in paired:
    metric,policy,level = row['metric'],row['policy'],row['level']
    d = cache['variable',policy,level][metric]-cache['fixed',policy,level][metric]
    sign = 1 if metric in ['psnr','ssim'] else -1
    calculated = dict(mean=d.mean(),median=np.median(d),p10=np.quantile(d,.1),p90=np.quantile(d,.9),improved_fraction=(sign*d>0).mean())
    assert all(np.isclose(x,float(row[k]),atol=1e-12,rtol=0) for k,x in calculated.items())
assert read(validation/'audit_summary.json')['native_python_export_masks_exact']
result = dict(passed=True,archive_sha256=sha(root/'evidence.tar.gz'),evidence_files_sha_verified=len(manifest),
              source_files_sha_verified=len(protocol['source_sha256']),baseline_reused_files=len(reused),
              fixed_retraining_used_in_comparison=False,serial_joint=False,quality_arms=18,per_view_rows=1152,
              paired_rows=36,raw_metrics_and_pairs_recomputed=True,common_k=plan['common_k'])
(root/'local_delivery_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

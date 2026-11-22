"""在本地从原始评价数据独立复算 Full100 证据，不需要 torch。"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parent
repo = root.parents[2]
evidence = root / 'evidence'
validation = evidence / 'validation'
read_json = lambda p: json.loads(p.read_text(encoding='utf-8'))
read_csv = lambda p: list(csv.DictReader(p.open(encoding='utf-8', newline='')))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = read_json(evidence / 'delivery_verification.json')
assert sha(root / 'evidence.tar.gz') == '2096628756efb7cdc09b7d3a12bc90e37ae9188851164dc7d3d97aa675da6982'
for name, digest in manifest['file_sha256'].items():
    assert sha(evidence / name) == digest, name
for name, digest in manifest['source_sha256'].items():
    assert sha(repo / name) == digest, name

baseline_config = repo / 'tests/anchor_selector/slot_deactivation_validation/online_evidence/enabled_seed2026/input_selector.yaml'
candidate_config = evidence / 'full100_seed2026/input_selector.yaml'
assert candidate_config.read_text(encoding='utf-8') == baseline_config.read_text(encoding='utf-8') + '\nTraining.post_topology_full_steps: 100\n'
gates = read_json(validation / 'correctness.json')
assert len(gates) == 6 and all(g['passed'] and g['code'] == 0 for g in gates.values())

trace = [{k: int(v) for k, v in r.items()} for r in read_csv(evidence / 'full100_seed2026/post_topology_full_trace.csv')]
assert [r['mapping'] for r in trace] == list(range(1, 10001))
remaining = 0
recovery_count = 0
triggers = []
for row in trace:
    assert row['recovery'] == int(remaining > 0), row
    if remaining:
        assert row['full'] == 1 and row['lod'] == row['selector_update'] == row['tf_update'] == 0, row
        remaining -= 1
        recovery_count += 1
    if row['topology_triggered']:
        assert row['full'] == 1 and row['lod'] == 0, row
        remaining = 100
        triggers.append(row['mapping'])
    assert row['remaining'] == remaining, row
state = read_json(evidence / 'full100_seed2026/10000_shutdown/ply/ultrafast_rotation_state.json')
assert sum(r['full'] for r in trace) == state['full_render_count'] == 5650
assert sum(r['lod'] for r in trace) == state['lod_render_count'] == 7106
assert recovery_count == state['recovery_steps'] == 1800
assert remaining == state['recovery_remaining'] == 100
assert triggers == list(range(1000, 10001, 500))
assert state['balanced'] and state['ultrafast_total_normal_steps'] == 10000
assert sum(r['complete'] == 'True' for r in read_csv(validation / 'recovery_windows.csv')) == 18

plan = read_json(validation / 'evaluation_plan.json')
assert len(plan['frames']) == len(set(plan['frames'])) == 64
summary = read_csv(validation / 'quality_summary.csv')
views = read_csv(validation / 'quality_per_view.csv')
paired = read_csv(validation / 'paired_statistics.csv')
assert len(summary) == 18 and len(views) == 1152 and len(paired) == 36
metrics = ['psnr', 'ssim', 'lpips', 'photo']
cache = {}
for arm in summary:
    key = (arm['method'], arm['policy'], arm['level'])
    folder = evidence / 'quality' / Path(*key)
    assert read_json(folder / 'evaluation_frames.json')['gt_indices'] == plan['frames']
    rows = [v for v in views if (v['method'], v['policy'], v['level']) == key]
    assert [int(v['gt_index']) for v in rows] == plan['frames']
    count = int(arm['selected'])
    if arm['level'] == 'Full':
        assert count == int(arm['total'])
    else:
        ids = np.load(folder / 'selected_rows.npz')['rows']
        assert len(ids) == len(np.unique(ids)) == count
        assert ids.min() >= 0 and ids.max() < int(arm['total'])
    if arm['policy'] == 'same_k':
        assert count == plan['common_k'][str(float(arm['ratio']))]
    cache[key] = {}
    for metric in metrics:
        raw = np.loadtxt(folder / (metric + '.txt'))
        assert raw.shape == (64,) and np.isfinite(raw).all()
        assert np.array_equal(raw, [float(v[metric]) for v in rows])
        assert np.isclose(raw.mean(), float(arm[metric]), atol=1e-12, rtol=0)
        cache[key][metric] = raw
for row in paired:
    metric, policy, level = row['metric'], row['policy'], row['level']
    delta = cache['full100', policy, level][metric] - cache['baseline', policy, level][metric]
    sign = 1 if metric in ['psnr', 'ssim'] else -1
    calculated = dict(mean=delta.mean(), median=np.median(delta), p10=np.quantile(delta, .1),
                      p90=np.quantile(delta, .9), improved_fraction=(sign * delta > 0).mean())
    assert all(np.isclose(value, float(row[name]), atol=1e-12, rtol=0) for name, value in calculated.items()), row

result = dict(passed=True, archive_sha256=sha(root / 'evidence.tar.gz'),
              evidence_file_sha256_checked=len(manifest['file_sha256']),
              local_source_sha256_checked=len(manifest['source_sha256']),
              only_config_difference='Training.post_topology_full_steps: 100',
              correctness_gates=6, trace_rows=10000, completed_recovery_windows=18,
              recovery_mappings=recovery_count, last_window_censored=True,
              quality_arms=18, quality_view_rows=1152, paired_rows=36,
              raw_metrics_recomputed=True, same_absolute_k_verified=True,
              native_python_export_masks_equal=read_json(validation / 'diagnostics.json')['real_native_python_export_masks_equal'])
(root / 'local_delivery_checks.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))

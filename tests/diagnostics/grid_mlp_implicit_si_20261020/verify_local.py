"""本地离线复核交付SHA、cell映射、same-K排序、预算移动和逐view质量统计。"""
from pathlib import Path
import csv, hashlib, json, math, subprocess
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'spatial_mlp_implicit_si_20261020'
load = lambda name: list(csv.DictReader(open(ROOT / name)))
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = read(ROOT / 'artifact_manifest.json')['files']
for name, item in manifest.items():
    path = ROOT / name
    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], name

sets = np.load(ROOT / 'selected_sets.npz')
rankings = np.load(ROOT / 'rankings.npz')
locked = read(ROOT / 'selected_before_quality.json')
assert sha(ROOT / 'selected_sets.npz') == locked['selected_SHA']
assert sha(ROOT / 'rankings.npz') == locked['rankings_SHA']
budget = load('cell_budget_redistribution.csv')
maximum_gauge_error = 0.
maximum_score_recompute_error = 0.
models = 0
for scene in ['fr1_room', 'fr1_xyz', 'fr2_xyz']:
    grid = np.load(ROOT / scene / 'cell_mapping.npz')
    original = np.load(OLD / scene / 'ranking_inputs.npz')
    ids, cell, occupancy = grid['gaussian_source_row'], grid['cell_id'], grid['occupancy']
    np.testing.assert_array_equal(ids, original['row_ids'])
    xyz = original['xyz'].astype(np.float64)
    keys, inverse = np.unique(np.floor(xyz / grid['h']).astype(np.int64), axis=0, return_inverse=True)
    np.testing.assert_array_equal(keys, grid['cell_key'])
    np.testing.assert_array_equal(inverse, cell)
    np.testing.assert_array_equal(occupancy, np.bincount(cell))
    np.testing.assert_array_equal(grid['cell_center'], (keys + .5) * grid['h'])
    centers = grid['cell_center']
    np.testing.assert_array_equal(grid['normalized_cell_center'], 2 * (centers-centers.min(0)) / np.ptp(centers, axis=0)-1)
    n, m = len(ids), len(keys)
    assert occupancy.sum() == n and occupancy.min() > 0 and m < n
    base = np.log(original['raw_I'].astype(float) + 1e-12)
    raw_order = np.lexsort((ids, -base))
    np.testing.assert_array_equal(raw_order, np.lexsort((ids, -original['raw_I'])))
    local_order = np.lexsort((ids, -base, cell))
    for arm in ['Grid-MLP', 'Independent-Cell']:
        for ratio in [.01, .05, .1, .15]:
            k = int(n*ratio + .5)
            raw_set = np.sort(ids[raw_order[:k]])
            raw_count = np.bincount(cell[raw_order[:k]], minlength=m)
            for seed in [2026, 2027]:
                key = f'{scene}_{arm}_{ratio}_{seed}'
                field = np.load(ROOT / scene / 'models' / f'{arm}_{int(100*ratio)}_seed_{seed}' / 'final_field.npz')
                a = field['cell_coefficient']
                np.testing.assert_array_equal(field['gaussian_correction'], a[cell])
                score = base + a[cell]
                # Windows/Linux的FP64 log可相差末位；排序和selected rows仍要求exact。
                maximum_score_recompute_error = max(maximum_score_recompute_error, float(abs(score-field['final_score']).max()))
                np.testing.assert_allclose(score, field['final_score'], atol=5e-14, rtol=0)
                np.testing.assert_array_equal(local_order, np.lexsort((ids, -score, cell)))
                order = np.lexsort((ids, -score))
                np.testing.assert_array_equal(field['final_ranking'], ids[order])
                np.testing.assert_array_equal(rankings[key + '_step_2000'], ids[order])
                selected = np.sort(ids[order[:k]])
                np.testing.assert_array_equal(field['selected_set'], selected)
                np.testing.assert_array_equal(sets[key], selected)
                np.testing.assert_array_equal(sets[key + '_step_0'], raw_set)
                assert len(selected) == len(np.unique(selected)) == k
                error = abs(float(np.dot(occupancy, a) / n))
                maximum_gauge_error = max(maximum_gauge_error, error)
                assert error < 1e-12 * max(1, abs(a).max())
                delta = np.bincount(cell[order[:k]], minlength=m) - raw_count
                assert delta.sum() == 0 and delta.clip(min=0).sum() == len(np.setdiff1d(selected, raw_set))
                rows = [z for z in budget if z['scene'] == scene and z['arm'] == arm and float(z['ratio']) == ratio and int(z['seed']) == seed]
                np.testing.assert_array_equal([int(z['delta_k']) for z in rows], delta)
                models += 1

q, views = load('quality_summary.csv'), load('quality_per_view.csv')
assert len(q) == 264 and len(views) == 15620
vm = {(z['scene'], z['split'], z['arm'], float(z['ratio']), int(z['seed']), int(z['frame_index'])): z for z in views}
assert len(vm) == len(views)
aggregate_error = paired_error = 0.
for row in q:
    info = read(OLD / row['scene'] / 'input_manifest.json')
    frames = info['bank_frames' if row['split'] == 'train' else 'evaluation_frames']
    assert not set(info['bank_frames']) & set(info['evaluation_frames'])
    for metric in ['psnr', 'ssim', 'lpips', 'photo']:
        mean = math.fsum(float(vm[row['scene'], row['split'], row['arm'], float(row['ratio']), int(row['seed']), f][metric]) for f in frames) / len(frames)
        aggregate_error = max(aggregate_error, abs(mean-float(row[metric])))
for row in load('paired_statistics.csv'):
    info = read(OLD / row['scene'] / 'input_manifest.json')
    frames = info['bank_frames' if row['split'] == 'train' else 'evaluation_frames']
    refseed = -1 if row['reference'] in ['Raw', 'Fixed', 'C'] else int(row['seed'])
    values = np.array([float(vm[row['scene'], row['split'], row['arm'], float(row['ratio']), int(row['seed']), f][row['metric']])-float(vm[row['scene'], row['split'], row['reference'], float(row['ratio']), refseed, f][row['metric']]) for f in frames])
    sign = 1 if row['metric'] in ['psnr', 'ssim'] else -1
    expected = dict(mean=values.mean(), median=np.median(values), p10=np.quantile(values,.1), p90=np.quantile(values,.9), improved_view_fraction=np.mean(sign*values > 0))
    for metric, value in expected.items():
        paired_error = max(paired_error, abs(float(row[metric])-value))
assert aggregate_error < 1e-12 and paired_error < 1e-12
curves = load('training_curves.csv')
assert len(curves) == 96000
assert max(float(z['normalized_gradient_sum']) for z in curves) < 1e-12
assert read(ROOT / 'correctness.json')['all_passed']
repo = ROOT.parents[2]
assert not subprocess.check_output(['git', 'diff', '--name-only'], cwd=repo).strip()
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=repo).strip()
result = dict(all_passed=True, manifest_files=len(manifest), models=models, formal_updates=len(curves), same_K=True, intra_cell_violations=0, weighted_mean_max_abs=maximum_gauge_error, cross_platform_score_recompute_max_abs=maximum_score_recompute_error, aggregate_perview_max_abs=aggregate_error, paired_max_abs=paired_error, tracked_and_index_unchanged=True)
(ROOT / 'local_delivery_checks.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, indent=2))

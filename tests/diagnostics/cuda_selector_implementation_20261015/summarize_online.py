"""从本次原始日志/metric 计算 A/B 和 legacy 自身波动，不设人为质量容差。"""
import json
from pathlib import Path
import re
import statistics
import sys

output = Path(sys.argv[1])
status = json.loads((output/'status.json').read_text())


def numeric(text):
    return {k: float(v) for k, v in re.findall(r'([A-Za-z0-9_]+)=(-?[0-9]+(?:\.[0-9]+)?)', text)}


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return dict(n=len(values), mean=statistics.mean(values), min=values[0], max=values[-1],
                std=statistics.stdev(values) if len(values) > 1 else None,
                p50=statistics.median(values), p95=values[max(0, (len(values)*95+99)//100-1)])


runs = []
signatures = {}
for item in status['runs']:
    if not item.get('complete'):
        continue
    run = output/item['name']
    lines = (run/'train.log').read_text(errors='replace').splitlines()
    row = {k: item[k] for k in ('name', 'scene', 'backend', 'repeat')}
    row['training_process_wall_seconds'] = item['train']['seconds']
    for marker, field in [('[Mapping PERF FINAL]', 'mapping'), ('[CUDA TF ONLINE]', 'shutdown')]:
        matches = [line for line in lines if marker in line]
        row[field] = numeric(matches[-1]) if matches else {}
    if 'complete_mapping_mean_ms' in row['mapping']:
        row['mapping']['accumulated_mapping_wall_ms'] = row['mapping']['complete_mapping_mean_ms']*row['mapping']['cycles']
    steps = [numeric(line) for line in lines if '[UltrafastRotation]' in line and 'training_ms=' in line]
    signatures[item['name']] = [{k: v[k] for k in ('normal_training_step', 'keyframe_id', 'ratio', 'total_gaussians', 'active_gaussians', 'loss') if k in v}
                                 for v in steps]
    row['step_statistics'] = {key: distribution([v[key] for v in steps if key in v]) for key in
        ('training_ms', 'active_gaussians', 'total_gaussians', 'tf_transformed_gaussians')}
    row['committed_updates_from_step_log'] = {key: sum(v.get(key, 0) for v in steps) for key in
        ('gaussian_optimizer_step', 'selector_optimizer_step', 'tf_optimizer_step')}
    lod = [v for v in steps if v.get('is_lod_step') == 1]
    row['lod_statistics'] = dict(selected=distribution([v['active_gaussians'] for v in lod]),
        actual_ratio=distribution([v['active_gaussians']/v['total_gaussians'] for v in lod if v['total_gaussians']]),
        tf_transformed=distribution([v['tf_transformed_gaussians'] for v in lod]),
        tf_transformed_total=sum(v['tf_transformed_gaussians'] for v in lod))
    row['selected_by_ratio'] = {}
    for ratio in (.01, .05, .1, .15):
        selected = [v for v in steps if abs(v.get('ratio', -1)-ratio) < 1e-7]
        row['selected_by_ratio'][str(ratio)] = dict(count=distribution([v['active_gaussians'] for v in selected]),
            actual_ratio=distribution([v['active_gaussians']/v['total_gaussians'] for v in selected if v['total_gaussians']]),
            tf_transformed=distribution([v['tf_transformed_gaussians'] for v in selected]))
    row['quality'] = {}
    for path in (run/'evaluation').rglob('eval.txt'):
        text = path.read_text()
        values = {k: float(v) for k, v in re.findall(r'^(psnr|ssim|lpips):\s*([-+0-9.eE]+)', text, re.M)}
        trajectory = path.parent/'ATE.txt'
        if trajectory.exists():
            values['trajectory_text'] = trajectory.read_text()
        row['quality'][path.parent.name] = values
    # run.py stores evo's full aligned trajectory statistics separately.
    row['trajectory_files'] = {str(p.relative_to(run)): p.read_text() for p in (run/'evaluation').rglob('*.txt')
                               if any(k in p.name.lower() for k in ('ate', 'ape', 'error', 'traj')) and p.stat().st_size < 20000}
    trajectories = list((run/'evaluation').rglob('metrics_traj.txt'))
    row['trajectory'] = {}
    if trajectories:
        errors = re.findall(r'^\s*rmse\s+([-+0-9.eE]+)', trajectories[0].read_text(), re.M)
        if len(errors) == 2:
            row['trajectory'] = dict(translation_rmse=float(errors[0]), rotation_rmse=float(errors[1]))
    row['evaluation_selector_statistics'] = {str(p.relative_to(run)): json.loads(p.read_text())
                                            for p in (run/'evaluation').rglob('selector_stats.json')}
    snapshot = next(iter(row['evaluation_selector_statistics'].values()), {})
    row['shutdown_population'] = {k: snapshot[k] for k in
        ('total_gaussians','compatibility_mature_count','elastic_eligible_count','maturity_mean','maturity_p10','maturity_p50','maturity_p90') if k in snapshot}
    row['missing_observability'] = ['per-step mature population', 'online selected global ID sequence',
                                    'per-stage isolated Selector times inside live Mapping']
    runs.append(row)

comparisons = []


def first_difference(reference, candidate, fields):
    for a, b in zip(signatures[reference], signatures[candidate]):
        changed = [k for k in fields if a.get(k) != b.get(k)]
        if changed:
            return dict(reference=a, candidate=b, differing_fields=changed)
    return None


for scene in ('room', 'desk', 'xyz', 'long'):
    a = [r for r in runs if r['scene'] == scene and r['backend'] == 'legacy']
    b = [r for r in runs if r['scene'] == scene and r['backend'] == 'cuda']
    if not a or not b:
        continue
    metrics = {k: ([r['mapping'][k] for r in a], [r['mapping'][k] for r in b]) for k in
               ('complete_mapping_mean_ms', 'complete_mapping_p95_ms', 'accumulated_mapping_wall_ms') if all(k in r['mapping'] for r in a+b)}
    metrics.update({k: ([r['shutdown'][k] for r in a], [r['shutdown'][k] for r in b]) for k in
                    ('mapping_cycles', 'selector_updates', 'tf_updates', 'gaussian_total') if all(k in r['shutdown'] for r in a+b)})
    metrics.update({k: ([r['trajectory'][k] for r in a], [r['trajectory'][k] for r in b]) for k in
                    ('translation_rmse', 'rotation_rmse') if all(k in r['trajectory'] for r in a+b)})
    metrics.update({k: ([r['shutdown_population'][k] for r in a], [r['shutdown_population'][k] for r in b]) for k in
                    ('compatibility_mature_count', 'elastic_eligible_count', 'maturity_mean') if all(k in r['shutdown_population'] for r in a+b)})
    for ratio in a[0]['quality']:
        for metric in ('psnr', 'ssim', 'lpips'):
            if all(metric in r['quality'].get(ratio, {}) for r in a+b):
                metrics[f'{ratio}/{metric}'] = ([r['quality'][ratio][metric] for r in a], [r['quality'][ratio][metric] for r in b])
    rows = {}
    for key, (legacy, cuda) in metrics.items():
        la, cb = distribution(legacy), distribution(cuda)
        delta = cb['mean']-la['mean']
        rows[key] = dict(legacy=la, cuda=cb, cuda_minus_legacy_mean=delta,
                         delta_div_legacy_std=delta/la['std'] if la['std'] else None,
                         cuda_mean_inside_observed_legacy_range=la['min'] <= cb['mean'] <= la['max'],
                         all_cuda_runs_inside_observed_legacy_range=min(cuda) >= min(legacy) and max(cuda) <= max(legacy))
    trace = {}
    for candidate in ([a[1]] if len(a)>1 else [])+b[:1]:
        trace[candidate['name']] = dict(reference=a[0]['name'],
            first_logged_input_difference=first_difference(a[0]['name'],candidate['name'],('keyframe_id','ratio','total_gaussians')),
            first_logged_output_difference=first_difference(a[0]['name'],candidate['name'],('active_gaussians','loss')))
    comparisons.append(dict(scene=scene, metrics=rows, repeated_runs_complete=len(a) == len(b) == 3, log_divergence=trace))

summary = dict(production_accepted=False, scope=status.get('scope'), runs=runs, comparisons=comparisons,
               complete=status.get('complete', False), active=status.get('active'), error=status.get('error'),
               variance_method='Three independent seed0 processes per backend. Report sample std, observed min/max, and CUDA-Legacy delta/std; no arbitrary quality tolerance. Three repeats do not establish statistical equivalence.')
(output/'online_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(completed=len(runs), active=summary['active'], complete=summary['complete']), ensure_ascii=False))

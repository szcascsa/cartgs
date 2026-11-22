"""配对 bootstrap 为主，独立 resampling 为敏感性；不使用 min/max gate。"""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

dest = Path(__file__).resolve().parent
root = dest.parents[2]
output = root/'results/selector_revised_acceptance_20261015'
status = json.loads((output/'status.json').read_text())
protocol = status['protocol']
records = []
for run in status['runs']:
    if not run.get('complete'):
        continue
    path = output/run['name']
    log = (path/'train.log').read_text(errors='replace')
    timing_lines = re.findall(r'^\[Mapping PERF FINAL\] (.*)', log, re.M)
    timing = dict(re.findall(r'(\w+)=([\d.e+-]+)', timing_lines[-1])) if timing_lines else {}
    tf_lines = re.findall(r'^\[Continuous TF PERF\] (.*)', log, re.M)
    tf = dict(re.findall(r'(\w+)=([\d.e+-]+)', tf_lines[-1])) if tf_lines else {}
    row = dict(phase=run['phase'], scene=run['scene'], backend=run['backend'], seed=run['algorithm_seed'],
        run_id=run['run_id'], pid=run['train']['pid'], process_wall_s=run['train']['seconds'])
    for key, value in run['counters'].items():
        if key != 'selector_timing_scope':
            row[key] = float(value)
    row.update(selector_updates=run['transform_state']['selector_step'], tf_updates=run['transform_state']['transform_step'],
        actual_gaussian_commits=run['actual_gaussian_commits'],
        mapping_mean_ms=float(timing.get('complete_mapping_mean_ms', 'nan')),
        mapping_p95_ms=float(timing.get('complete_mapping_p95_ms', 'nan')),
        tf_host_total_ms=float(tf.get('total_tf_ms_host', 'nan')))
    trajectory, = [v for k, v in run['trajectory_metrics'].items() if '/full/' in k]
    trans, rot = re.findall(r'rmse\s+([\d.e+-]+)', trajectory)
    row.update(translation_rmse=float(trans), rotation_rmse=float(rot))
    for summary in run['evaluation_summary']:
        ratio = summary['ratio']
        scope = 'full' if summary['phase'] == 'FULL' else f"ratio_{float(ratio):.2f}"
        for key in ['psnr', 'ssim', 'lpips', 'total_gaussians', 'active_gaussians', 'compatibility_mature_count']:
            if summary.get(key):
                row[f'{scope}/{key}'] = float(summary[key])
        if scope != 'full':
            row[f'{scope}/selected_ratio'] = float(summary['active_gaussians'])/float(summary['total_gaussians'])
    records.append(row)
raw = pd.DataFrame(records)
raw.to_csv(dest/'online_runs.csv', index=False)
rng = np.random.default_rng(protocol['bootstrap']['seed'])
N = protocol['bootstrap']['resamples']
statistics = []
for scene in protocol['scenes']:
    pair = raw[(raw.phase == 'statistical') & (raw.scene == scene)]
    a = pair[pair.backend == 'legacy'].set_index('seed').sort_index()
    b = pair[pair.backend == 'cuda'].set_index('seed').sort_index()
    assert list(a.index) == list(b.index) == sorted(protocol['algorithm_seeds']), f'Incomplete cohort {scene}'
    for metric in [c for c in raw.columns if c not in ['phase', 'scene', 'backend', 'seed', 'run_id', 'pid']]:
        if a[metric].isna().any() or b[metric].isna().any():
            continue
        x, y = a[metric].to_numpy(float), b[metric].to_numpy(float)
        sx, sy = x.std(ddof=1), y.std(ddof=1)
        delta = float(y.mean()-x.mean())
        index = rng.integers(0, len(x), size=(N, len(x)))
        diff = (y-x)[index].mean(axis=1)
        independent = y[rng.integers(0,len(y),size=(N,len(y)))].mean(axis=1)-x[rng.integers(0,len(x),size=(N,len(x)))].mean(axis=1)
        pooled = np.sqrt((sx*sx+sy*sy)/2)
        entry = dict(scene=scene, metric=metric, n_legacy=len(x), n_cuda=len(y),
            legacy_mean=float(x.mean()), legacy_std=float(sx), cuda_mean=float(y.mean()), cuda_std=float(sy),
            difference=delta, paired_ci95=np.quantile(diff,[.025,.975]).tolist(),
            independent_ci95=np.quantile(independent,[.025,.975]).tolist(),
            normalized_difference=delta/sx if sx else None,
            hedges_g=(1-3/(4*(len(x)+len(y))-9))*delta/pooled if pooled else None)
        key = metric.split('/')[-1]
        margins = protocol['online_noninferiority_margins']
        if key in ['psnr', 'ssim', 'lpips']:
            margin = margins[key]
            lower, upper = entry['paired_ci95']
            passed = upper <= margin if key == 'lpips' else lower >= -margin
            sensitivity = entry['independent_ci95'][1] <= margin if key == 'lpips' else entry['independent_ci95'][0] >= -margin
            entry.update(noninferiority_margin=margin, noninferiority_passed=bool(passed), independent_sensitivity_passed=bool(sensitivity))
        elif metric in ['translation_rmse', 'rotation_rmse']:
            epsilon = margins['translation_epsilon_m' if metric.startswith('translation') else 'rotation_epsilon_unitless']
            adjusted = y-(1+margins['trajectory_relative'])*x-epsilon
            ci = np.quantile(adjusted[index].mean(axis=1),[.025,.975]).tolist()
            entry.update(noninferiority_margin_formula=f'CUDA <= 1.05*Legacy + {epsilon}',
                adjusted_difference_ci95=ci, noninferiority_passed=bool(ci[1] <= 0))
        statistics.append(entry)
gates = [r for r in statistics if 'noninferiority_passed' in r]
summary = dict(statistics=statistics, noninferiority_checks=len(gates),
    noninferiority_passed=all(r['noninferiority_passed'] for r in gates),
    failed_or_inconclusive=[dict(scene=r['scene'], metric=r['metric'], difference=r['difference'], ci95=r['paired_ci95'])
                           for r in gates if not r['noninferiority_passed']],
    interpretation='CI crossing the registered margin is insufficient evidence of non-inferiority; it does not prove a numerical backend defect. Equal-work/time are separate cohorts.')
(dest/'online_statistics.json').write_text(json.dumps(summary, indent=2)+'\n')
pd.json_normalize(statistics).to_csv(dest/'online_statistics.csv', index=False)
contrasts = []
for phase in ['equal_work', 'equal_time']:
    for scene in protocol['scenes']:
        pair = raw[(raw.phase == phase) & (raw.scene == scene)].set_index('backend')
        assert set(pair.index) == {'legacy', 'cuda'}
        a, b = pair.loc['legacy'], pair.loc['cuda']
        contrasts.append(dict(phase=phase, scene=scene,
            process_wall_speedup=float(a.process_wall_s/b.process_wall_s),
            mapping_work_speedup=float(a.mapping_host_work_ms/b.mapping_host_work_ms),
            mapping_cycle_speedup=float(a.mapping_mean_ms/b.mapping_mean_ms),
            legacy=a.to_dict(), cuda=b.to_dict(),
            deltas={k:float(b[k]-a[k]) for k in raw.columns if k not in ['phase','scene','backend','seed','run_id','pid']}))
(dest/'online_budget_contrasts.json').write_text(json.dumps(contrasts, indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='statistics'}, indent=2))

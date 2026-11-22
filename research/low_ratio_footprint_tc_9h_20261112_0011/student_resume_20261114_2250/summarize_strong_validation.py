"""弱尾与溢出候选逐场景汇总；同输入配对，保留两轮、慢例和全部质量反例。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIMARY = [f'local{x:g}' for x in [.1, .2, .3, .4, .5]]


def main():
    root = HERE / 'reports'
    state = json.loads((root / 'RELEASE_RENDER_20261115_CONTROLLER.json').read_text())
    assert state['state'] == 'complete', state['state']
    merged = {}
    for workload, prefix, expected in [('student_lod', 'release_render_formal_', 18),
                                       ('nested', 'release_render_nested_', 32)]:
        jobs = [name for name in state['accepted'] if name.startswith(prefix)]
        assert len(jobs) == expected, (workload, len(jobs))
        for name in jobs:
            job = json.loads((root / 'jobs' / (name + '.json')).read_text())
            assert job['measurement_valid'] and job['returncode'] == 0, name
            for kind in ['samples', 'events', 'quality', 'checks', 'workloads', 'memory']:
                path = root / name / (kind + '.csv')
                if path.exists() and path.stat().st_size > 1:
                    data = pd.read_csv(path)
                    data['workload'], data['job'] = workload, name
                    merged.setdefault(kind, []).append(data)
    data = {name: pd.concat(parts, ignore_index=True) for name, parts in merged.items()}
    for name, frame in data.items():
        frame.to_csv(root / f'STRONG_FINAL_{name.upper()}.csv.gz', index=False, compression='gzip')
    samples = data['samples']
    keys = ['workload', 'scene', 'condition', 'round', 'variant']
    stats = samples.groupby(keys).agg(p50_ms=('ms', 'median'), p95_ms=('ms', lambda x: x.quantile(.95)),
        mean_ms=('ms', 'mean'), K=('K', 'median'), K_min=('K', 'min'), K_max=('K', 'max'),
        N=('N', 'first'), samples=('ms', 'size')).reset_index()
    for variant, label in [('F2', 'F2'), ('tile_safe_bitexp_stop', 'old38'), ('previous182', 'previous182')]:
        ref = stats[stats.variant == variant][keys[:-1] + ['p50_ms']].rename(columns={'p50_ms': label + '_p50_ms'})
        stats = stats.merge(ref, on=keys[:-1])
        stats[label + '_speedup'] = stats[label + '_p50_ms'] / stats.p50_ms
    stats.to_csv(root / 'STRONG_SCENE_RATIO.csv', index=False)
    stats[(stats.F2_speedup < 1) | (stats.previous182_speedup < 1)].to_csv(root / 'STRONG_ALL_SLOW_CASES.csv', index=False)
    stats[stats.condition == 'full'].to_csv(root / 'STRONG_FULL_ACCEPTANCE.csv', index=False)
    events = data['events'].groupby(keys).agg(event_p50_ms=('cuda_ms', 'median'),
        event_p95_ms=('cuda_ms', lambda x: x.quantile(.95))).reset_index()
    events.to_csv(root / 'STRONG_CUDA_EVENT.csv', index=False)
    quality = data['quality'].groupby(['workload', 'scene', 'condition', 'variant']).agg(
        psnr_loss=('psnr_loss', 'mean'), worst_frame_loss=('psnr_loss', 'max'), psnr=('psnr', 'mean'),
        baseline_psnr=('baseline_psnr', 'mean'), ssim=('ssim', 'mean'), baseline_ssim=('baseline_ssim', 'mean'),
        lpips=('lpips', 'mean'), baseline_lpips=('baseline_lpips', 'mean')).reset_index()
    quality.to_csv(root / 'STRONG_QUALITY_CONDITIONS.csv', index=False)
    quality[quality.psnr_loss > .5].to_csv(root / 'STRONG_QUALITY_FAILURES.csv', index=False)
    overview = []
    for (workload, variant), frame in stats[stats.condition.isin(PRIMARY)].groupby(['workload', 'variant']):
        q = quality[(quality.workload == workload) & (quality.variant == variant)]
        overview.append(dict(workload=workload, variant=variant, p50_us=frame.p50_ms.mean() * 1000,
            p95_us=frame.p95_ms.mean() * 1000,
            F2_speedup=np.exp(np.log(frame.F2_speedup).mean()), old38_speedup=np.exp(np.log(frame.old38_speedup).mean()),
            previous182_speedup=np.exp(np.log(frame.previous182_speedup).mean()),
            slow_vs_F2=int((frame.F2_speedup < 1).sum()), slow_vs_previous182=int((frame.previous182_speedup < 1).sum()),
            psnr_loss=q.psnr_loss.mean() if len(q) else 0., worst_condition=q.psnr_loss.max() if len(q) else 0.,
            worst_frame=q.worst_frame_loss.max() if len(q) else 0.,
            quality_pass=bool(not len(q) or (q.psnr_loss.mean() <= .2 and q.psnr_loss.max() <= .5))))
    pd.DataFrame(overview).to_csv(root / 'STRONG_OVERVIEW.csv', index=False)
    per = samples.groupby(['workload', 'scene', 'round', 'condition', 'frame', 'motion', 'variant']).agg(
        K=('K', 'first'), T=('ms', 'median')).reset_index()
    full = per[per.condition == 'full'][['workload', 'scene', 'round', 'frame', 'variant', 'K', 'T']].rename(columns={'K': 'K_ref', 'T': 'T_ref'})
    per = per.merge(full, on=['workload', 'scene', 'round', 'frame', 'variant'])
    per['normalized_K'], per['normalized_T'] = per.K / per.K_ref, per['T'] / per.T_ref
    per['ideal_gap'] = per.normalized_T - per.normalized_K
    per.to_csv(root / 'STRONG_PER_FRAME.csv.gz', index=False, compression='gzip')
    fits, inversions = [], []
    for key, frame in per[(per.condition.str.startswith('local')) | (per.condition == 'full')].groupby(['workload', 'scene', 'round', 'frame', 'variant']):
        frame = frame.sort_values('K')
        k, t = frame.K.to_numpy(), frame['T'].to_numpy() * 1000
        if len(np.unique(k)) < 2:
            continue
        b, a = np.linalg.lstsq(np.stack([np.ones_like(k), k], axis=1), t, rcond=None)[0]
        residual, total = ((t - (b + a * k)) ** 2).sum(), ((t - t.mean()) ** 2).sum()
        base = dict(zip(['workload', 'scene', 'round', 'frame', 'variant'], key))
        fits.append(dict(**base, intercept_us=b, slope_us_per_1k=a * 1000,
            R2=1 - residual / total if total else 1., mean_ideal_gap=frame.ideal_gap.mean(),
            inversions_gt2pct=int((t[1:] < t[:-1] * .98).sum())))
        for index in range(len(k) - 1):
            if k[index + 1] > k[index] and t[index + 1] < t[index]:
                inversions.append(dict(**base, lower_K=k[index], higher_K=k[index + 1],
                    lower_us=t[index], higher_us=t[index + 1], drop_pct=(1 - t[index + 1] / t[index]) * 100))
    pd.DataFrame(fits).to_csv(root / 'STRONG_SCALING_FIT.csv', index=False)
    pd.DataFrame(inversions).to_csv(root / 'STRONG_INVERSIONS.csv', index=False)
    checks = data['checks']
    assert checks.radii_bitwise.all()
    print(pd.DataFrame(overview).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

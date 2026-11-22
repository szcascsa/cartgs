"""最终Rasterizer与起点最快纯渲染入口同进程配对；质量仅在逐位证明后复用。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    r = HERE / 'reports'
    state = json.loads((r / 'RELEASE_JOINT_20261115_MERGED.json').read_text())
    assert state['state'] == 'complete'
    names = [name for name in state['accepted'] if name.startswith('release_thin_')]
    assert len(names) == 50, len(names)
    data = {}
    for kind in ['samples', 'events', 'checks']:
        parts = []
        for name in names:
            job = json.loads((r / 'jobs' / (name + '.json')).read_text())
            assert job['returncode'] == 0 and job['measurement_valid']
            frame = pd.read_csv(r / name / (kind + '.csv'))
            frame['job'] = name
            parts.append(frame)
        data[kind] = pd.concat(parts, ignore_index=True)
        data[kind].to_csv(r / ('RELEASE_THIN_' + kind.upper() + '.csv.gz'), index=False, compression='gzip')
    assert data['checks'].rgb_bitwise.all() and data['checks'].radii_bitwise.all()
    keys = ['workload', 'scene', 'condition', 'round', 'variant']
    stats = data['samples'].groupby(keys).agg(p50_ms=('ms', 'median'), p95_ms=('ms', lambda x: x.quantile(.95)),
        K=('K', 'median'), K_min=('K', 'min'), K_max=('K', 'max'), N=('N', 'first'), samples=('ms', 'size')).reset_index()
    for variant in ['F2', 'previous182', 'thin182', 'new_regular']:
        baseline = stats[stats.variant.eq(variant)][keys[:-1] + ['p50_ms']].rename(columns={'p50_ms': variant + '_p50_ms'})
        stats = stats.merge(baseline, on=keys[:-1], validate='many_to_one')
        stats[variant + '_speedup'] = stats[variant + '_p50_ms'] / stats.p50_ms
    stats.to_csv(r / 'RELEASE_THIN_SCENE_RATIO.csv', index=False)
    data['events'].groupby(keys).agg(event_p50_ms=('cuda_ms', 'median'),
        event_p95_ms=('cuda_ms', lambda x: x.quantile(.95))).reset_index().to_csv(r / 'RELEASE_THIN_CUDA_EVENT.csv', index=False)
    per = data['samples'].groupby(['workload', 'scene', 'round', 'condition', 'frame', 'motion', 'variant']).agg(
        K=('K', 'first'), T=('ms', 'median')).reset_index()
    full = per[per.condition.eq('full')][['workload', 'scene', 'round', 'frame', 'variant', 'K', 'T']].rename(columns={'K': 'K_ref', 'T': 'T_ref'})
    per = per.merge(full, on=['workload', 'scene', 'round', 'frame', 'variant'], validate='many_to_one')
    per['normalized_K'], per['normalized_T'] = per.K / per.K_ref, per['T'] / per.T_ref
    per['ideal_gap'] = per.normalized_T - per.normalized_K
    per.to_csv(r / 'RELEASE_THIN_PER_FRAME.csv.gz', index=False, compression='gzip')
    rows = []
    primary = [f'local{x:g}' for x in [.1, .2, .3, .4, .5]]
    for (workload, variant), frame in stats[stats.condition.isin(primary)].groupby(['workload', 'variant']):
        rows.append(dict(workload=workload, variant=variant, p50_us=frame.p50_ms.mean() * 1000,
            p95_us=frame.p95_ms.mean() * 1000, F2_speedup=np.exp(np.log(frame.F2_speedup).mean()),
            previous182_speedup=np.exp(np.log(frame.previous182_speedup).mean()),
            thin182_speedup=np.exp(np.log(frame.thin182_speedup).mean()),
            slower_F2=int(frame.F2_speedup.lt(1).sum()), slower_thin182=int(frame.thin182_speedup.lt(1).sum())))
    pd.DataFrame(rows).to_csv(r / 'RELEASE_THIN_OVERVIEW.csv', index=False)
    stats[stats.condition.eq('full')].to_csv(r / 'RELEASE_THIN_ALL_FULL.csv', index=False)
    stats[(stats.F2_speedup < 1) | (stats.thin182_speedup < 1)].to_csv(r / 'RELEASE_THIN_ALL_SLOW_CASES.csv', index=False)
    (r / 'RELEASE_THIN_SUMMARY.json').write_text(json.dumps(dict(passed=True, jobs=len(names),
        samples=len(data['samples']), checks=len(data['checks']), exact_dispatch=True), indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()

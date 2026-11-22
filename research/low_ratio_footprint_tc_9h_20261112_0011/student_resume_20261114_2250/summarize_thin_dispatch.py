"""汇总已完成的轻量入口复验，图像相同的质量指标直接复用，不重复渲染。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIMARY = [f'local{x:g}' for x in [.1, .2, .3, .4, .5]]


def main():
    root = HERE / 'reports'
    state = json.loads((root / 'RESEARCH_ACCEPTANCE_CONTROLLER.json').read_text())
    assert state['state'] == 'failed' and state['current_task'] == 'student_execution_identities'
    files = {}
    for workload, prefix, count in [('student_lod', 'student_thin_formal_', 18),
                                     ('nested', 'student_thin_nested_', 32)]:
        jobs = [name for name in state['accepted'] if name.startswith(prefix)]
        assert len(jobs) == count, (workload, len(jobs))
        for name in jobs:
            job = json.loads((root / 'jobs' / (name + '.json')).read_text())
            assert job['measurement_valid'] and job['returncode'] == 0
            assert json.loads((root / name / 'DONE.json').read_text())['passed']
            for kind in ['samples', 'events', 'checks', 'memory']:
                path = root / name / (kind + '.csv')
                if path.exists() and path.stat().st_size > 1:
                    data = pd.read_csv(path)
                    data['workload'] = workload
                    data['job'] = name
                    files.setdefault(kind, []).append(data)
    data = {name: pd.concat(parts, ignore_index=True) for name, parts in files.items()}
    keys = ['workload', 'scene', 'condition', 'round', 'variant']
    samples = data['samples']
    stats = samples.groupby(keys).agg(p50_ms=('ms', 'median'), p95_ms=('ms', lambda x: x.quantile(.95)),
        K=('K', 'median'), N=('N', 'first'), samples=('ms', 'size')).reset_index()
    for variant, label in [('F2', 'F2'), ('tile_safe_bitexp_stop', 'old38'), ('encoded_signed_square', 'previous182')]:
        ref = stats[stats.variant == variant][keys[:-1] + ['p50_ms']].rename(columns={'p50_ms': label + '_p50_ms'})
        stats = stats.merge(ref, on=keys[:-1])
        stats[label + '_speedup'] = stats[label + '_p50_ms'] / stats.p50_ms
    stats.to_csv(root / 'THIN_SCENE_RATIO.csv', index=False)
    stats[(stats.F2_speedup < 1) | (stats.previous182_speedup < 1)].to_csv(root / 'THIN_ALL_SLOW_CASES.csv', index=False)
    stats[(stats.condition == 'full') & (stats.variant == 'thin_dispatch')].to_csv(root / 'THIN_FULL_ACCEPTANCE.csv', index=False)
    event_stats = data['events'].groupby(keys).agg(event_p50_ms=('cuda_ms', 'median'),
        event_p95_ms=('cuda_ms', lambda x: x.quantile(.95))).reset_index()
    event_stats.to_csv(root / 'THIN_CUDA_EVENT.csv', index=False)
    overview = []
    for (workload, variant), part in stats[stats.condition.isin(PRIMARY)].groupby(['workload', 'variant']):
        overview.append(dict(workload=workload, variant=variant, p50_us=part.p50_ms.mean() * 1000,
            p95_us=part.p95_ms.mean() * 1000,
            F2_speedup=np.exp(np.log(part.F2_speedup).mean()),
            old38_speedup=np.exp(np.log(part.old38_speedup).mean()),
            previous182_speedup=np.exp(np.log(part.previous182_speedup).mean()),
            slow_vs_F2=int((part.F2_speedup < 1).sum()),
            slow_vs_previous182=int((part.previous182_speedup < 1).sum())))
    overview = pd.DataFrame(overview)
    overview.to_csv(root / 'THIN_OVERVIEW.csv', index=False)
    for kind, frame in data.items():
        frame.to_csv(root / f'THIN_{kind.upper()}.csv.gz', index=False, compression='gzip')
    (root / 'THIN_SUMMARY.json').write_text(json.dumps(dict(jobs=50, samples=len(samples),
        workloads={key: int(value.scene.nunique()) for key, value in stats.groupby('workload')},
        quality_reused_from='RENDER_QUALITY_CONDITIONS.csv',
        quality_reuse_basis='同一182二进制；正式脚本每帧先断言Thin与182的RGB/radii逐位相同',
        no_images=True), ensure_ascii=False, indent=2))
    print(overview.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

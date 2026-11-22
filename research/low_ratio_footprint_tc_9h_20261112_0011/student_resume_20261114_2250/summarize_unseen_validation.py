"""冻结配置在未参与开发的新视角上验收；不隐藏画质或性能失败项。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIMARY = [f'local{x:g}' for x in [.1, .2, .3, .4, .5]]


def main():
    r = HERE / 'reports'
    state = json.loads((r / 'RELEASE_UNSEEN_20261115_MERGED.json').read_text())
    assert state['state'] == 'complete'
    groups = {}
    for name in state['accepted']:
        if not name.startswith('unseen_'):
            continue
        job = json.loads((r / 'jobs' / (name + '.json')).read_text())
        assert job['returncode'] == 0 and job['measurement_valid']
        kind = 'nested' if name.startswith('unseen_nested_quality_') else 'thin' if name.startswith('unseen_thin_') else 'joint'
        for metric in ['samples', 'events', 'quality', 'checks']:
            path = r / name / (metric + '.csv')
            if path.exists() and path.stat().st_size > 1:
                frame = pd.read_csv(path)
                frame['job'] = name
                groups.setdefault((kind, metric), []).append(frame)
    merged = {key: pd.concat(value, ignore_index=True) for key, value in groups.items()}
    for (kind, metric), frame in merged.items():
        frame.to_csv(r / ('UNSEEN_' + kind.upper() + '_' + metric.upper() + '.csv.gz'), index=False, compression='gzip')
    summaries = {}
    for kind in ['thin', 'joint']:
        data = merged[kind, 'samples']
        keys = (['workload'] if kind == 'thin' else []) + ['scene', 'condition', 'round', 'variant']
        stats = data.groupby(keys).agg(p50_ms=('ms', 'median'), p95_ms=('ms', lambda x: x.quantile(.95)),
            K=('K', 'median'), N=('N', 'first'), samples=('ms', 'size')).reset_index()
        refs = ['F2', 'thin182', 'previous182'] if kind == 'thin' else ['A_fp32_F2', 'C_native_F2', 'I_previous']
        for variant in refs:
            frame = stats[stats.variant.eq(variant)][keys[:-1] + ['p50_ms']].rename(columns={'p50_ms': variant + '_p50_ms'})
            stats = stats.merge(frame, on=keys[:-1], validate='many_to_one')
            stats[variant + '_speedup'] = stats[variant + '_p50_ms'] / stats.p50_ms
        stats.to_csv(r / ('UNSEEN_' + kind.upper() + '_SCENE_RATIO.csv'), index=False)
        stats[stats.condition.eq('full')].to_csv(r / ('UNSEEN_' + kind.upper() + '_ALL_FULL.csv'), index=False)
        stats[np.logical_or.reduce([stats[name + '_speedup'].lt(1) for name in refs])].to_csv(r / ('UNSEEN_' + kind.upper() + '_ALL_SLOW_CASES.csv'), index=False)
        check = merged[kind, 'checks']
        assert check.rgb_bitwise.all() and check.radii_bitwise.all()
        if kind == 'joint':
            assert check.same_ids.all()
        group = (['workload'] if kind == 'thin' else []) + ['variant']
        rows = []
        for key, values in stats[stats.condition.isin(PRIMARY)].groupby(group):
            base = dict(zip(group, key if isinstance(key, tuple) else (key,)))
            base.update(p50_us=values.p50_ms.mean() * 1000, p95_us=values.p95_ms.mean() * 1000)
            for reference in refs:
                base[reference + '_speedup'] = float(np.exp(np.log(values[reference + '_speedup']).mean()))
                base[reference + '_slow_cases'] = int(values[reference + '_speedup'].lt(1).sum())
            rows.append(base)
        pd.DataFrame(rows).to_csv(r / ('UNSEEN_' + kind.upper() + '_OVERVIEW.csv'), index=False)
        summaries[kind] = dict(samples=len(data), checks=len(check), scenes=stats.scene.nunique(),
            rounds=stats['round'].nunique(), conditions=stats.condition.nunique())
    quality = merged['joint', 'quality'].groupby(['scene', 'condition', 'variant']).agg(
        psnr_loss=('psnr_loss', 'mean'), worst_frame_loss=('psnr_loss', 'max'),
        psnr=('psnr', 'mean'), baseline_psnr=('baseline_psnr', 'mean'),
        ssim=('ssim', 'mean'), baseline_ssim=('baseline_ssim', 'mean'),
        lpips=('lpips', 'mean'), baseline_lpips=('baseline_lpips', 'mean'),
        id_overlap=('id_overlap', 'mean'), worst_id_overlap=('id_overlap', 'min')).reset_index()
    quality.to_csv(r / 'UNSEEN_JOINT_QUALITY_CONDITIONS.csv', index=False)
    final = quality[quality.variant.eq('D_final')]
    nested = merged['nested', 'quality'].groupby(['scene', 'condition']).agg(
        psnr_loss=('psnr_loss', 'mean'), worst_frame_loss=('psnr_loss', 'max'),
        psnr=('psnr', 'mean'), baseline_psnr=('baseline_psnr', 'mean'), ssim=('ssim', 'mean'),
        baseline_ssim=('baseline_ssim', 'mean'), lpips=('lpips', 'mean'), baseline_lpips=('baseline_lpips', 'mean')).reset_index()
    nested.to_csv(r / 'UNSEEN_NESTED_QUALITY_CONDITIONS.csv', index=False)
    result = dict(**summaries, mean_psnr_loss=float(final.psnr_loss.mean()),
        worst_condition_loss=float(final.psnr_loss.max()), worst_frame_loss=float(final.worst_frame_loss.max()),
        quality_pass=bool(final.psnr_loss.mean() <= .2 and final.psnr_loss.max() <= .5),
        nested_mean_psnr_loss=float(nested.psnr_loss.mean()), nested_worst_condition=float(nested.psnr_loss.max()),
        nested_worst_frame=float(nested.worst_frame_loss.max()),
        nested_quality_pass=bool(nested.psnr_loss.mean() <= .2 and nested.psnr_loss.max() <= .5),
        no_parameter_tuning=True, accepted=state['accepted'], rejected=state['rejected'])
    (r / 'UNSEEN_SUMMARY.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ['accepted', 'rejected']}, indent=2))


if __name__ == '__main__':
    main()

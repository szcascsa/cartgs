"""隔离LoD内部缩放与LoD/Full跨域比较；保存全部反序，不强制线性或非负截距。"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE / 'reports'


def analyze(data, keys, prefix):
    fits, inversions = [], []
    for scope in ['ratios_only', 'ratios_and_full']:
        mask = data.condition.str.startswith('local')
        if scope == 'ratios_and_full':
            mask |= data.condition.eq('full')
        for key, frame in data[mask].groupby(keys):
            base = dict(zip(keys, key))
            frame = frame.sort_values(['K', 'condition'])
            k, t = frame.K.to_numpy(), frame['T'].to_numpy() * 1000.
            if len(np.unique(k)) < 2:
                continue
            b, a = np.linalg.lstsq(np.column_stack([np.ones_like(k), k]), t, rcond=None)[0]
            residual = np.square(t - (b + a * k)).sum()
            total = np.square(t - t.mean()).sum()
            for index in range(len(k) - 1):
                if k[index + 1] > k[index] and t[index + 1] < t[index]:
                    inversions.append(dict(**base, scope=scope, lower_condition=frame.iloc[index].condition,
                        higher_condition=frame.iloc[index + 1].condition, lower_K=k[index], higher_K=k[index + 1],
                        lower_us=t[index], higher_us=t[index + 1], reduction_pct=(1 - t[index + 1] / t[index]) * 100.))
            significant = sum(k[i + 1] > k[i] and t[i + 1] < .98 * t[i] for i in range(len(k) - 1))
            fits.append(dict(**base, scope=scope, points=len(k), intercept_us=b, slope_us_per_1k=a * 1000,
                R2=1 - residual / total if total else 1., inversions_gt2pct=significant,
                normalized_T_median=frame.normalized_T.median(), normalized_K_median=frame.normalized_K.median(),
                mean_ideal_gap=(frame.normalized_T - frame.normalized_K).mean()))
    f = pd.DataFrame(fits)
    f.to_csv(R / (prefix + '_SCALING_FIT.csv'), index=False)
    pd.DataFrame(inversions).to_csv(R / (prefix + '_ALL_INVERSIONS.csv'), index=False)
    groups = [key for key in ['workload', 'variant', 'scope'] if key in f]
    f.groupby(groups).agg(intercept_us=('intercept_us', 'median'), slope_us_per_1k=('slope_us_per_1k', 'median'),
        R2=('R2', 'median'), mean_ideal_gap=('mean_ideal_gap', 'median'),
        series=('R2', 'size'), series_with_inversions=('inversions_gt2pct', lambda x: int(x.gt(0).sum())),
        total_inversions_gt2pct=('inversions_gt2pct', 'sum')).reset_index().to_csv(R / (prefix + '_SCALING_OVERVIEW.csv'), index=False)
    data[data.condition.str.startswith('local')].groupby(groups[:-1] + ['condition']).agg(K=('K', 'median'),
        T_us=('T', lambda x: x.median() * 1000), normalized_K=('normalized_K', 'median'),
        normalized_T=('normalized_T', 'median')).reset_index().to_csv(R / (prefix + '_NORMALIZED_BY_RATIO.csv'), index=False)


def main():
    pure = pd.read_csv(R / 'RELEASE_THIN_PER_FRAME.csv.gz')
    analyze(pure, ['workload', 'scene', 'round', 'frame', 'variant'], 'FINAL_RASTER')
    joint = pd.read_csv(R / 'RELEASE_JOINT_PER_FRAME.csv.gz')
    analyze(joint, ['scene', 'round', 'frame', 'variant'], 'FINAL_JOINT')
    print('RELEASE_SCALING_COMPLETE')


if __name__ == '__main__':
    main()

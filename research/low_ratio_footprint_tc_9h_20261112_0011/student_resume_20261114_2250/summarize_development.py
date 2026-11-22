"""保留所有开发候选、独立轮次、逐条件速度与正确性计数。"""
from pathlib import Path
import argparse
import json
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage')
    args = parser.parse_args()
    state = json.loads((HERE/'reports'/(args.stage+'_CONTROLLER.json')).read_text())
    jobs = [name for name in state['accepted'] if (HERE/'reports'/name/'samples.csv').exists()]
    samples = pd.concat([pd.read_csv(HERE/'reports'/name/'samples.csv') for name in jobs])
    stats = samples.groupby(['scene', 'condition', 'round', 'variant']).ms.agg(p50='median', p95=lambda x:x.quantile(.95)).reset_index()
    base = stats[stats.variant=='G_cpp_pose'].drop(columns='variant').rename(columns={'p50':'reference_p50','p95':'reference_p95'})
    stats = stats.merge(base, on=['scene','condition','round'])
    stats['speedup'] = stats.reference_p50 / stats.p50
    stats.to_csv(HERE/'reports'/(args.stage+'_SUMMARY.csv'), index=False)
    checks = pd.concat([pd.read_csv(HERE/'reports'/name/'checks.csv') for name in jobs])
    counts = checks.groupby('variant').agg(checks=('same_ids','size'), same_ids=('same_ids','sum'),
        rgb_bitwise=('rgb_bitwise','sum'),radii_bitwise=('radii_bitwise','sum')).reset_index()
    counts.to_csv(HERE/'reports'/(args.stage+'_CHECKS.csv'), index=False)
    primary = stats[stats.condition.isin([f'local{x:g}' for x in [.1,.2,.3,.4,.5]])]
    print(primary.groupby('variant').speedup.agg(['mean','min','max']).to_string())
    print(counts.to_string(index=False))


if __name__ == '__main__':
    main()

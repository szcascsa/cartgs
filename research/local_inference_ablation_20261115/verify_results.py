"""从逐帧记录独立复核局部消融指标、预算和历史选点。"""
import argparse
import csv
import math
from pathlib import Path
from common import ROOT, RATIOS, VARIANTS, load, write, records


def main(final=False):
    reference = {(row['group'], float(row['ratio'])): row for row in records()}
    completed = sorted((ROOT/'runs').rglob('GROUP_COMPLETE.json'))
    checks = []
    for marker in completed:
        folder = Path(load(marker)['output'])
        completion = load(folder/'COMPLETE.json')
        group = completion['group']
        assert completion['complete'] and completion['model_state_bitwise_unchanged']
        assert completion['source_unchanged'] and completion['all_frame_ids_and_K_verified']
        assert len(completion['results']) == 22
        assert len(completion['packet_checks']) == 66 and len(completion['reference_checks']) == 33
        assert load(folder/'INPUTS_BEFORE.json') == load(folder/'INPUTS_AFTER.json')
        group_rows = 0
        max_error = {name: 0. for name in ['psnr', 'ssim', 'lpips']}
        for ratio in RATIOS:
            item = reference[group, ratio]
            historical = Path(item['source'])
            hashes = load(historical/'selection_hashes.json')
            with (historical/'anchor_frustum_per_view.csv').open() as stream:
                prior = list(csv.DictReader(stream))
            variants = []
            for variant in VARIANTS:
                output = folder/variant/f'ratio_{ratio:g}'
                summary = load(output/'metric_summary.json')
                with (output/'per_frame_metrics.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                assert len(rows) == int(item['frames']) == summary['frames'] == len(prior)
                assert summary['group'] == group and summary['variant'] == variant
                assert summary['N'] == int(item['N']) and summary['ratio'] == ratio
                assert summary['training_steps'] == summary['images_saved'] == 0
                for index, (row, old) in enumerate(zip(rows, prior)):
                    assert row['group'] == group and row['variant'] == variant
                    assert int(row['frame']) == index and float(row['ratio']) == ratio
                    for key, old_key in [('trajectory_index','trajectory_index'),('rgb_index','gt_index'),
                        ('candidate_N','candidate_gaussians'),('eligible_N','eligible_candidates'),
                        ('requested_K','requested_budget'),('K','active_gaussians')]:
                        assert int(row[key]) == int(float(old[old_key])), (group, ratio, variant, index, key)
                    assert int(row['K']) == min(int(row['requested_K']), int(row['eligible_N']))
                    assert int(row['budget_shortfall']) == int(row['requested_K'])-int(row['K'])
                    assert all(math.isfinite(float(row[name])) for name in ['psnr','ssim','lpips'])
                    if variant == 'selector_no_comp':
                        assert row['selection_sha256'] == hashes[index]
                for name in ['psnr','ssim','lpips']:
                    mean = math.fsum(float(row[name]) for row in rows)/len(rows)
                    error = abs(mean-summary[name])
                    assert error < 1e-9
                    assert summary['baseline_'+name] == float(item[name])
                    assert abs(summary['delta_'+name]-(summary[name]-float(item[name]))) < 1e-12
                    max_error[name] = max(max_error[name], error)
                assert abs(math.fsum(int(row['K']) for row in rows)/len(rows)-float(item['K_mean'])) < 1e-9
                variants.append(rows)
                group_rows += len(rows)
            for left, right in zip(*variants):
                for key in ['frame','trajectory_index','rgb_index','rgb_timestamp','N',
                            'candidate_N','eligible_N','requested_K','K','budget_shortfall']:
                    assert left[key] == right[key]
        checks.append(dict(group=group, evaluations=22, per_frame_rows=group_rows,
                           mean_reaggregation_max_error=max_error))
        print('VERIFIED_GROUP', group, group_rows, flush=True)
    if final:
        assert len(checks) == 32
    write(ROOT/'reports'/('PER_FRAME_VERIFIED.json' if final else 'PER_FRAME_VERIFIED_PARTIAL.json'),
          dict(complete=len(checks)==32,groups=len(checks),evaluations=22*len(checks),
               per_frame_rows=sum(item['per_frame_rows'] for item in checks),checks=checks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    main(args.final)

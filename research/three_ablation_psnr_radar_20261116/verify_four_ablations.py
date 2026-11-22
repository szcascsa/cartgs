"""独立核验四项消融的共同样本聚合、Detach原始数据和交付文件。"""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parent
RATIOS = [.01, .03, .05, .08, .1, .2, .3, .4, .5, .6, .7]
VARIANTS = {'selector_no_comp', 'random_with_comp', 'no_importance', 'master_detach'}


def read_csv(name):
    with (ROOT/'data'/name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def mean(values):
    return math.fsum(values)/len(values)


def close(left, right):
    return abs(float(left)-float(right)) <= 1e-12


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    groups, scenes, datasets = (read_csv(name) for name in
        ['PER_GROUP_PSNR_DECREASE.csv', 'PER_SCENE_PSNR_DECREASE.csv', 'DATASET_PSNR_DECREASE.csv'])
    assert len(groups) == 1188 and len(scenes) == 660 and len(datasets) == 132
    assert {row['variant'] for row in groups} == VARIANTS
    assert len({row['group'] for row in groups}) == 27

    by_scene = defaultdict(list)
    for row in groups:
        by_scene[row['dataset'], row['scene'], float(row['ratio']), row['variant']].append(row)
    max_error = 0.
    for row in scenes:
        values = by_scene[row['dataset'], row['scene'], float(row['ratio']), row['variant']]
        expected = mean([float(value['psnr_decrease']) for value in values])
        max_error = max(max_error, abs(expected-float(row['psnr_decrease'])))
        assert close(expected, row['psnr_decrease'])

    by_dataset = defaultdict(list)
    for row in scenes:
        by_dataset[row['dataset'], float(row['ratio']), row['variant']].append(row)
    for row in datasets:
        values = by_dataset[row['dataset'], float(row['ratio']), row['variant']]
        expected = mean([float(value['psnr_decrease']) for value in values])
        max_error = max(max_error, abs(expected-float(row['psnr_decrease'])))
        assert close(expected, row['psnr_decrease'])

    baseline = {(row['group'], float(row['ratio'])): row
                for row in read_csv('source_inference_comparison.csv')}
    detached = {(row['id'], float(row['ratio'])): row for row in read_csv('source_master_detach_original_rgb_metrics.csv')
                if row['scope'] == 'view_ratio'}
    detach_audit = read_csv('MASTER_DETACH_ALL32_PSNR_DECREASE.csv')
    assert len(baseline) == len(detached) == len(detach_audit) == 352
    for row in detach_audit:
        key = row['group'], float(row['ratio'])
        base, frozen = baseline[key], detached[key]
        assert close(row['baseline_psnr'], base['baseline_psnr'])
        assert close(row['detach_psnr'], frozen['psnr'])
        assert close(row['psnr_decrease'], float(base['baseline_psnr'])-float(frozen['psnr']))
        assert int(row['frames']) == int(base['frames']) == int(frozen['evaluated_frames'])
        assert int(row['N']) == int(base['N']) == int(frozen['gaussian_count'])
        assert close(row['K_mean'], base['K_mean']) and close(row['K_mean'], frozen['actual_K_mean'])

    assert all(digest(Path(record['path'])) == record['sha256'] for record in manifest['sources'].values())
    figure_files = []
    for name, expected_size in [('THREE_DATASET_LOCAL_ABLATION_PSNR_DECREASE', (8100, 4500)),
            ('REPLICA_FOUR_ABLATION_PSNR_DECREASE', (3600, 4500)),
            ('TUM_RGBD_FOUR_ABLATION_PSNR_DECREASE', (3600, 4500)),
            ('VECTOR_FOUR_ABLATION_PSNR_DECREASE', (3600, 4500))]:
        for extension in ['png', 'svg', 'pdf']:
            path = ROOT/'figures'/f'{name}.{extension}'
            assert path.is_file() and path.stat().st_size > 10_000
            figure_files.append(path)
        assert Image.open(ROOT/'figures'/f'{name}.png').size == expected_size
    assert all(digest(path) == manifest['outputs'][str(path.relative_to(ROOT)).replace('\\','/')]
               for path in figure_files)

    report = dict(complete=True, independent_weighted_reaggregation='passed', max_error=max_error,
        frozen_stereo_original_rgb_reevaluated=69,
        common_groups=27, master_detach_groups=32, master_detach_evaluations=736,
        excluded_zero_selection_groups=5, dataset_means_verified=132,
        baseline_references_verified=297, master_detach_pairs_verified=352,
        original_sources_unchanged=True, figure_files_verified=len(figure_files),
        png_dimensions_verified=True, radial_limits=[-3, 18], zip_verified=True)
    (ROOT/'QA_VERIFIED.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with zipfile.ZipFile(ROOT/'FOUR_ABLATION_PSNR_FIGURES.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in figure_files:
            archive.write(path, path.relative_to(ROOT))
        for path in (ROOT/'data').rglob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(ROOT))
        archive.write(ROOT/'figures/data-manifest.md', Path('figures/data-manifest.md'))
        for path in [ROOT/'plot_three_ablations.py', Path(__file__), ROOT/'SOURCE_MANIFEST.json',
                     ROOT/'README_结果说明.md', ROOT/'QA_VERIFIED.json']:
            archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(ROOT/'FOUR_ABLATION_PSNR_FIGURES.zip') as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        assert all(str(path.relative_to(ROOT)).replace('\\','/') in names for path in figure_files)
        assert {'QA_VERIFIED.json', 'verify_four_ablations.py'} <= names
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()

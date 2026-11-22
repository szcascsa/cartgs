"""将已核验的逐场景质量与去重点数整理为三个数据集表；不改原评价。"""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = PROJECT / 'research/all_dataset_ratio_radar_global_local_20261114/unique_union'
OUTPUT = PROJECT.parent / 'outputs/scene_ratio_quality_tables_20261115'
RATIOS = [.01, .03, .05, .08, .1, .2, .3, .4, .5, .6, .7]
METHODS = [
    ('cartgs_random', 'CaRtGS随机删除'),
    ('cartgs_opacity', 'CaRtGS不透明度删除'),
    ('ours_global', '我们：全局Ratio'),
    ('ours_view_ratio', '我们：局部Ratio'),
]
MODE_LABELS = {'mono': '单目（Mono）', 'rgbd': 'RGB-D', 'stereo': '双目（Stereo）'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig')))


def main():
    receipt = json.loads((SOURCE / 'QA_RECEIPT.json').read_text(encoding='utf-8'))
    assert receipt['complete'] and receipt['quality_and_original_K_unchanged']
    master = PROJECT / 'research/cartgs_pruning_baselines_20261112/reports/METHODS_ALL_METRICS.csv'
    assert sha(master) == receipt['source_sha256']
    original = {(r['group'], r['method'], float(r['ratio'])): r for r in load_rows(master)}
    union_path = SOURCE / 'data/LOCAL_UNION_COUNTS.csv'
    unions = {(r['group'], float(r['ratio'])): r for r in load_rows(union_path)}
    assert len(unions) == 352
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'data').mkdir(exist_ok=True)
    sources = {str(master): sha(master), str(union_path): sha(union_path)}
    datasets = []
    metric_rows = 0
    for tag in ['REPLICA', 'TUM', 'VECTOR']:
        folder = SOURCE / tag / 'data'
        manifest = json.loads((folder / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
        path = folder / 'PER_SCENE.csv'
        sources[str(path)] = sha(path)
        assert manifest['union_source_sha256'] == sha(union_path)
        records = load_rows(path)
        lookup = {(r['scene'], r['mode'], r['method'], float(r['ratio'])): r for r in records}
        expected = {(s, m, method, ratio) for s in manifest['scenes'] for m in manifest['modes']
                    for method, _ in METHODS for ratio in RATIOS}
        assert len(lookup) == len(records) and set(lookup) == expected
        groups = []
        wide_rows = []
        flat_rows = []
        for mode in manifest['modes']:
            for scene in manifest['scenes']:
                group = dict(scene=scene, mode=mode, mode_label=MODE_LABELS[mode], methods=[])
                for method, label in METHODS:
                    values = []
                    wide = dict(相机模式=MODE_LABELS[mode], 场景=scene, 方法=label)
                    for ratio in RATIOS:
                        row = lookup[scene, mode, method, ratio]
                        base = original[row['group'], method, ratio]
                        assert all(row[k] == base[k] for k in ['psnr', 'ssim', 'lpips', 'N', 'frames', 'source'])
                        assert float(row['K']) == float(base['K_mean'])
                        count = float(row['unique_total'])
                        assert count.is_integer() and 0 <= count <= int(row['N'])
                        if method == 'ours_view_ratio':
                            u = unions[row['group'], ratio]
                            assert int(count) == int(u['local_union_count'])
                            assert u['all_view_hashes_match'] == 'True'
                            assert float(row['K']) == float(u['local_mean_K'])
                        else:
                            assert count == float(row['K'])
                        result = dict(ratio=ratio, psnr=float(row['psnr']), ssim=float(row['ssim']),
                                      lpips=float(row['lpips']), gs=int(count))
                        assert all(math.isfinite(result[k]) for k in ['psnr', 'ssim', 'lpips'])
                        values.append(result)
                        prefix = f'{ratio*100:g}%'
                        wide.update({prefix + ' PSNR(dB)': result['psnr'], prefix + ' SSIM': result['ssim'],
                                     prefix + ' LPIPS': result['lpips'], prefix + ' #GS': result['gs']})
                        flat_rows.append(dict(dataset=manifest['dataset'], **row))
                        metric_rows += 1
                    full_method = 'ours_full' if method.startswith('ours_') else 'cartgs_full'
                    full = original[f"{manifest['dataset']}/{scene}/{mode}", full_method, 1.0]
                    assert full['N'] == row['N'] and full['frames'] == row['frames']
                    assert full['N'] == full['K_mean']
                    full_result = dict(ratio=1.0, psnr=float(full['psnr']), ssim=float(full['ssim']),
                                       lpips=float(full['lpips']), gs=int(full['N']))
                    values.append(full_result)
                    wide.update({'Full PSNR(dB)': full_result['psnr'], 'Full SSIM': full_result['ssim'],
                                 'Full LPIPS': full_result['lpips'], 'Full #GS': full_result['gs']})
                    flat_rows.append(dict(dataset=manifest['dataset'], group=full['group'],scene=scene,
                                          mode=mode,method=method,ratio=1.0,N=full['N'],K=full['K_mean'],
                                          frames=full['frames'],source=full['source'],psnr=full['psnr'],
                                          ssim=full['ssim'],lpips=full['lpips'],unique_total=full['N']))
                    group['methods'].append(dict(method=method, label=label, values=values))
                    wide_rows.append(wide)
                groups.append(group)
        for name, rows in [(f'{tag}_全部场景表.csv', wide_rows), (f'data/{tag}_逐项数据.csv', flat_rows)]:
            with (OUTPUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        datasets.append(dict(tag=tag, name=manifest['dataset'], scenes=manifest['scenes'],
                             modes=manifest['modes'], groups=groups, data_rows=len(wide_rows)))
    assert metric_rows == 1408 and sum(d['data_rows'] for d in datasets) == 128
    payload = dict(output=str(OUTPUT), ratios=RATIOS, methods=METHODS, datasets=datasets,
                   column_groups=[*RATIOS, 1.0],full_source_records=64,
                   sources=sources, metric_rows=metric_rows,
                   count_definition='局部#GS=所有评价视角选中Master Gaussian ID的并集基数；重复出现只计一次。全局及删除方法#GS=固定保留集大小。')
    (ROOT / 'table_payload.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('TABLE_DATA_VERIFIED', 'LoD',metric_rows, 'Full_display_rows',128,'numeric_values',(metric_rows+128)*4,
          'dataset_rows', {d['tag']: d['data_rows'] for d in datasets}, flush=True)


if __name__ == '__main__':
    main()

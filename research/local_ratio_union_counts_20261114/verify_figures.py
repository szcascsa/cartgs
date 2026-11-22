"""独立核查并集图的数据、SVG坐标、旧文件完整性与交付包。"""
from decimal import Decimal
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT.parent / 'all_dataset_ratio_radar_global_local_20261114'
OUTPUT = PLOTS / 'unique_union'
RATIOS = ['0.01', '0.03', '0.05', '0.08', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7']
METHODS = ['cartgs_random', 'cartgs_opacity', 'ours_global', 'ours_view_ratio']
METRICS = ['psnr', 'ssim', 'lpips']
NS = {'s': 'http://www.w3.org/2000/svg'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig')))


def mean(values):
    return sum(values) / Decimal(len(values))


def main():
    statistics = json.loads((ROOT / 'UNION_VERIFIED.json').read_text(encoding='utf-8'))
    assert statistics['complete'] and statistics['ratios'] == 352
    counts = {(r['group'], r['ratio']): r for r in rows(ROOT / 'LOCAL_UNION_COUNTS.csv')}
    assert len(counts) == 352
    previous = json.loads((ROOT / 'OLD_ARTIFACTS_BEFORE.json').read_text(encoding='utf-8'))
    assert all(digest(Path(p)) == value for p, value in previous.items())
    receipt = json.loads((OUTPUT / 'VERIFIED.json').read_text(encoding='utf-8'))
    assert receipt['complete'] and receipt['source_unchanged']
    scene_rows = mean_rows = svg_paths = 0
    for tag, config in receipt['datasets'].items():
        directory = OUTPUT / tag
        old_scene = rows(PLOTS / tag / 'data/PER_SCENE.csv')
        new_scene = rows(directory / 'data/PER_SCENE.csv')
        old_means = rows(PLOTS / tag / 'data/RADAR_MEANS.csv')
        new_means = rows(directory / 'data/RADAR_MEANS.csv')
        assert len(old_scene) == len(new_scene) and len(old_means) == len(new_means) == 132
        for before, after in zip(old_scene, new_scene):
            assert all(after[key] == value for key, value in before.items())
            expected = (counts[after['group'], after['ratio']]['local_union_count']
                        if after['method'] == 'ours_view_ratio' else after['K'])
            assert Decimal(after['unique_total']) == Decimal(expected)
            assert Decimal(after['K']) <= Decimal(after['unique_total']) <= Decimal(after['N'])
        for before, after in zip(old_means, new_means):
            assert all(after[key] == value for key, value in before.items())
            modes = config['modes'] if after['mode'] == 'all' else [after['mode']]
            for field in ['unique_total', 'psnr', 'ssim', 'lpips']:
                per_scene = []
                for scene in config['scenes']:
                    selected = [r for r in new_scene if r['scene'] == scene and r['mode'] in modes
                                and r['method'] == after['method'] and r['ratio'] == after['ratio']]
                    assert len(selected) == len(modes)
                    per_scene.append(mean([Decimal(r[field]) for r in selected]))
                assert abs(Decimal(after[field]) - mean(per_scene)) < Decimal('1e-8')
        lookup = {(r['mode'], r['method'], r['ratio']): r for r in new_means}
        for metric in METRICS:
            for kind, modes in [('RADAR', ['all']), ('BY_MODE', config['modes'])]:
                svg_paths += check_svg(directory / f'figures/{tag}_{metric.upper()}_{kind}.svg',
                                       lookup, [(m, metric) for m in modes], config['K_axis_max'])
        svg_paths += check_svg(directory / f'figures/{tag}_RADAR_OVERVIEW.svg', lookup,
                               [('all', m) for m in METRICS], config['K_axis_max'])
        for extension in ['png', 'svg', 'pdf']:
            files = list((directory / 'figures').glob('*.' + extension))
            assert len(files) == 7
            if extension == 'png':
                for file in files:
                    with Image.open(file) as image:
                        assert all(abs(d - 450) < 0.1 for d in image.info['dpi'])
                        image.verify()
        scene_rows += len(new_scene)
        mean_rows += len(new_means)
    assert scene_rows == 1408 and mean_rows == 396 and svg_paths == 288
    for path, expected in receipt['figures'].items():
        assert digest(OUTPUT / path) == expected
    for name in ['UNION_VERIFIED.json', 'FINISHED.json']:
        (OUTPUT / 'data' / name).write_bytes((ROOT / name).read_bytes())
    result = dict(complete=True,scene_rows=scene_rows,mean_rows=mean_rows,
                  svg_paths_verified=svg_paths,svg_coordinate_tolerance=1e-5,
                  count_table_cells_verified=svg_paths//8*44,
                  unchanged_old_files=len(previous),quality_and_original_K_unchanged=True,
                  count_means_verified_with_decimal=True,png_dpi=450,
                  source_sha256=config['source_sha256'],statistics=statistics)
    (OUTPUT / 'QA_RECEIPT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive_path = OUTPUT / 'ALL_DATASETS_GLOBAL_LOCAL_UNION_RADAR.zip'
    with zipfile.ZipFile(archive_path, 'a', zipfile.ZIP_DEFLATED) as archive:
        archive.write(OUTPUT / 'QA_RECEIPT.json', 'QA_RECEIPT.json')
        for name in ['UNION_VERIFIED.json', 'FINISHED.json']:
            archive.write(OUTPUT / 'data' / name, 'data/' + name)
        for name in ['collect_union.py', 'run_queue.py', 'verify_union.py', 'verify_figures.py', 'PROTOCOL.md']:
            archive.write(ROOT / name, 'statistics/' + name)
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert archive.testzip() is None
        assert all(len([n for n in archive.namelist() if n.endswith('.'+ext)]) == 21
                   for ext in ['png', 'svg', 'pdf'])
    print('FIGURE_QA_COMPLETE',scene_rows,mean_rows,svg_paths,'old_files',len(previous),flush=True)


def check_svg(path, lookup, panels, count_limit):
    root = ET.parse(path).getroot()
    verified = 0
    for method in METHODS:
        for kind in ['quality_', 'retained_K_']:
            groups = root.findall(f".//s:g[@id='{kind + method}']", NS)
            assert len(groups) == len(panels), (path, kind, method)
            for group, (mode, metric) in zip(groups, panels):
                line = group.find('s:path', NS)
                coordinates = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?(?:e[-+]?\d+)?',line.attrib['d'])]
                assert len(coordinates) == 24
                clip = re.fullmatch(r'url\(#(.+)\)', line.attrib['clip-path']).group(1)
                rect = root.find(f".//s:clipPath[@id='{clip}']/s:rect", NS).attrib
                x, y, width, height = [float(rect[key]) for key in ['x','y','width','height']]
                expected = []
                for index, ratio in enumerate(RATIOS + RATIOS[:1]):
                    value = lookup[mode, method, ratio]
                    if kind == 'retained_K_': radius = float(value['unique_total']) / count_limit
                    elif metric == 'psnr': radius = float(value[metric]) / 40
                    elif metric == 'lpips': radius = 1 - float(value[metric])
                    else: radius = float(value[metric])
                    angle = math.pi/2 - (index % 11) * 2 * math.pi/11
                    expected.extend([x+(radius*math.cos(angle)+1.3)/2.6*width,
                                     y+(1.3-radius*math.sin(angle))/2.6*height])
                assert max(abs(a-b) for a,b in zip(coordinates,expected)) < 1e-5, (path,mode,kind,method)
                verified += 1
    tables = [g for g in root.findall('.//s:g', NS) if g.attrib.get('id', '').startswith('table_')]
    assert len(tables) == len(panels)
    for table, (mode, _) in zip(tables, panels):
        labels = [''.join(t.itertext()) for t in table.findall('.//s:text', NS)]
        expected = ['Ratio'] + [f'{float(r)*100:g}%' for r in RATIOS]
        for method, label in zip(METHODS, ['随机删除', 'Opacity删除', '我们全局', '我们局部']):
            expected += [label] + [f"{float(lookup[mode,method,r]['unique_total'])/1000:.2f}" for r in RATIOS]
        assert labels == expected, (path, mode, '点数表不符')
    return verified


if __name__ == '__main__':
    main()

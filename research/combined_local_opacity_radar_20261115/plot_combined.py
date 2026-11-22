"""三数据集全部32组等权平均：局部方法与Opacity Top-K；质量和点数分别出图。"""
from decimal import Decimal
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
PREVIOUS = RESEARCH / 'all_dataset_ratio_radar_global_local_20261114/unique_union'
STYLE = RESEARCH / 'replica_ratio_radar_20261113/plot_replica_radar.py'
if not STYLE.exists():
    STYLE = ROOT / 'plot_replica_radar.py'
sys.path.insert(0, str(STYLE.parent))
import plot_replica_radar as base
import numpy as np
from PIL import Image

METHODS = ['cartgs_opacity', 'ours_view_ratio']
LABELS = {'cartgs_opacity': 'CaRtGS：Opacity Top-K', 'ours_view_ratio': '我们：局部视角Ratio'}
COLORS = {'cartgs_opacity': '#C99B2E', 'ours_view_ratio': '#319B80'}
SOURCES = {'REPLICA': 16, 'TUM': 10, 'VECTOR': 6}
FIELDS = ['psnr', 'ssim', 'lpips', 'unique_total', 'K', 'N']
SOURCE_SHA = 'e56cc01dfe0dcfb1613f06d4211ce5d7e137b6c6ce341c2667db200cc027c891'


def read_csv(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig')))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def aggregate():
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    inputs = [STYLE]
    raw, groups = [], set()
    for tag, count in SOURCES.items():
        source = PREVIOUS / tag / 'data/PER_SCENE.csv'
        if not source.exists():
            source = data / f'SOURCE_{tag}.csv'
        inputs.append(source)
        selected = [r for r in read_csv(source) if r['method'] in METHODS]
        assert len(selected) == count * 2 * 11
        assert len({r['group'] for r in selected}) == count
        groups.update(r['group'] for r in selected)
        raw.extend(dict(dataset=tag, **r) for r in selected)
        destination = data / f'SOURCE_{tag}.csv'
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    assert len(groups) == 32 and len(raw) == 704
    keys = [(r['group'], r['method'], float(r['ratio'])) for r in raw]
    assert len(set(keys)) == len(keys)
    assert set(keys) == {(g, m, r) for g in groups for m in METHODS for r in base.RATIOS}
    union_source = RESEARCH / 'local_ratio_union_counts_20261114/LOCAL_UNION_COUNTS.csv'
    if not union_source.exists():
        union_source = data / 'LOCAL_UNION_COUNTS.csv'
    inputs.append(union_source)
    unions = {(r['group'], float(r['ratio'])): r for r in read_csv(union_source)}
    assert len(unions) == 352
    metrics_source = RESEARCH / 'cartgs_pruning_baselines_20261112/reports/METHODS_ALL_METRICS.csv'
    if not metrics_source.exists():
        metrics_source = data / 'METHODS_ALL_METRICS.csv'
    inputs.append(metrics_source)
    assert digest(metrics_source) == SOURCE_SHA
    original = {(r['group'], r['method'], float(r['ratio'])): r for r in read_csv(metrics_source)}
    for row in raw:
        old = original[row['group'], row['method'], float(row['ratio'])]
        assert all(row[f] == old[f] for f in ['psnr', 'ssim', 'lpips', 'frames', 'N', 'source'])
        assert Decimal(row['K']) == Decimal(old['K_mean'])
        assert Decimal(row['K']) <= Decimal(row['unique_total']) <= Decimal(row['N'])
        if row['method'] == 'ours_view_ratio':
            union = unions[row['group'], float(row['ratio'])]
            assert row['unique_total'] == union['local_union_count']
            assert union['all_view_hashes_match'] == 'True'
            assert row['source'] == union['source'] and row['frames'] == union['views']
        else:
            assert Decimal(row['unique_total']) == Decimal(row['K'])
            assert int(Decimal(row['K'])) == int(Decimal(row['ratio']) * Decimal(row['N']))
    for source in [union_source, metrics_source]:
        destination = data / source.name
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    means, max_error = [], Decimal(0)
    for method in METHODS:
        for ratio in base.RATIOS:
            selected = [r for r in raw if r['method'] == method and float(r['ratio']) == ratio]
            assert len(selected) == 32
            row = dict(mode='all', method=method, ratio=ratio, scene_mode_groups=32,
                       weighting='equal_32_scene_mode_groups', evaluated_frames=sum(int(r['frames']) for r in selected))
            for field in FIELDS:
                row[field] = math.fsum(float(r[field]) for r in selected) / 32
                independent = sum(Decimal(r[field]) for r in selected) / Decimal(32)
                error = abs(Decimal(str(row[field])) - independent)
                assert error < Decimal('1e-8')
                max_error = max(max_error, error)
            means.append(row)
    base.write_csv(data / 'COMBINED_PER_GROUP.csv', raw)
    base.write_csv(data / 'COMBINED_MEANS.csv', means)
    return means, inputs, str(max_error)


def draw(ax, rows, metric, small=False):
    base.radar(ax, rows, metric, 'all', small=small)
    for method in METHODS:
        line = next(line for line in ax.lines if line.get_color() == COLORS[method]
                    and len(line.get_xdata()) == 12)
        line.set_gid(f'{metric}_{method}')
        line.set_linewidth(2.0)


def table(fig, rows, metric, bounds, fontsize=8.5):
    ax = fig.add_axes(bounds)
    ax.axis('off')
    labels = ['CaRtGS Top-K', '我们局部']
    cells = []
    for method, label in zip(METHODS, labels):
        values = [next(r[metric] for r in rows if r['method'] == method and r['ratio'] == ratio)
                  for ratio in base.RATIOS]
        formatted = ([f'{v/1000:.2f}' for v in values] if metric == 'unique_total'
                     else [f'{v:.2f}' if metric == 'psnr' else f'{v:.3f}' for v in values])
        cells.append([label] + formatted)
    result = ax.table(cellText=cells, colLabels=['Ratio'] + [f'{r*100:g}%' for r in base.RATIOS],
                      cellLoc='center', colWidths=[.15] + [.85/11]*11, bbox=[0, 0, 1, 1])
    result.auto_set_font_size(False)
    result.set_fontsize(fontsize)
    for (row, column), cell in result.get_celld().items():
        cell.set_edgecolor('#E1E6EA')
        cell.set_linewidth(.5)
        cell.set_facecolor('#F1F4F6' if row == 0 else ('white' if row == 1 else '#FAFBFC'))
        if row and column == 0:
            cell.get_text().set_color(COLORS[METHODS[row-1]])
            cell.get_text().set_fontweight('bold')


def make_figures(rows):
    base.METHODS = METHODS
    base.LABELS.update(LABELS)
    base.COLORS.update(COLORS)
    base.MARKERS['ours_view_ratio'] = '^'
    base.configure_font()
    limit = math.ceil(max(r['unique_total'] for r in rows)/10000)*10000
    base.METRICS['unique_total'] = ('Gaussian总数U', '点', limit,
                                   list(np.linspace(limit/5, limit, 5)))
    figures = ROOT / 'figures'
    figures.mkdir(exist_ok=True)
    (figures / 'data-manifest.md').write_text('# 数据清单\n\n'
        '|图|数据|真实／模拟|统计|输出|\n|---|---|---|---|---|\n'
        '|PSNR、SSIM、LPIPS、Gaussian总数各一张及总览|../data/COMBINED_MEANS.csv|真实|32个场景／模式等权平均|450 DPI PNG、SVG、PDF|\n\n'
        '源数据：Replica 16组、TUM 10组、VECtor 6组。仅保留ours_view_ratio和cartgs_opacity。'
        '质量不重新渲染；局部U为每个场景／模式／Ratio全部评价视角选中母Gaussian ID并集大小；Top-K取固定保留集大小。\n', encoding='utf-8')
    for metric in ['psnr', 'ssim', 'lpips', 'unique_total']:
        fig = base.plt.figure(figsize=(7.8, 9.5))
        ax = fig.add_axes([.035, .270, .93, .62])
        draw(ax, rows, metric)
        if metric == 'unique_total':
            for label, tick in zip(ax.texts[:5], base.METRICS[metric][3]):
                label.set_text(f'{tick/1000:g}k')
        title = base.METRICS[metric][0]
        fig.suptitle('三数据集合并 · ' + title, y=.976, fontsize=21, fontweight='bold', color='#22313B')
        fig.text(.5, .927, 'Replica 16组 + TUM 10组 + VECtor 6组 · 全部32组等权平均',
                 ha='center', fontsize=10.2, color='#52636F')
        fig.legend(handles=base.legend_handles(), loc='lower center', bbox_to_anchor=(.5, .229),
                   ncol=2, frameon=False, fontsize=11)
        fig.text(.5, .215, '跨视角去重总数U（千点）' if metric == 'unique_total' else '各Ratio平均' + title,
                 ha='center', fontsize=10, color='#33434F')
        table(fig, rows, metric, [.035, .110, .93, .086])
        note = ('局部U：所有评价视角选中ID并集；Top-K：固定保留总数；单位k=千点。'
                if metric == 'unique_total' else '实线仅表示渲染质量；沿用原等距刻度，质量越外越好。')
        fig.text(.5, .067, note, ha='center', fontsize=8.8, color='#52636F')
        fig.text(.5, .037, '每组权重1/32；不按帧数加权；CaRtGS按不透明度排序保留Top-K。',
                 ha='center', fontsize=8.8, color='#52636F')
        base.save(fig, figures / f'COMBINED_{metric.upper()}_RADAR')
    # 总览复用已核验的四张单图，不另外生成一组曲线。
    overview = base.plt.figure(figsize=(15.6, 19))
    for index, metric in enumerate(['psnr', 'ssim', 'lpips', 'unique_total']):
        ax = overview.add_axes([index%2*.5, 1-(index//2+1)*.5, .5, .5])
        ax.imshow(Image.open(figures / f'COMBINED_{metric.upper()}_RADAR.png'))
        ax.axis('off')
    base.save(overview, figures / 'COMBINED_OVERVIEW')
    return limit


def verify_svg(rows, limit):
    ns = {'s': 'http://www.w3.org/2000/svg'}
    for metric in ['psnr', 'ssim', 'lpips', 'unique_total']:
        root = ET.parse(ROOT / f'figures/COMBINED_{metric.upper()}_RADAR.svg').getroot()
        for method in METHODS:
            path = root.find(f".//s:g[@id='{metric}_{method}']/s:path", ns)
            actual = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?(?:e[-+]?\d+)?', path.attrib['d'])]
            clip = re.fullmatch(r'url\(#(.+)\)', path.attrib['clip-path']).group(1)
            box = root.find(f".//s:clipPath[@id='{clip}']/s:rect", ns).attrib
            x, y, w, h = [float(box[k]) for k in ['x', 'y', 'width', 'height']]
            expected = []
            for index, ratio in enumerate(base.RATIOS + base.RATIOS[:1]):
                value = next(r[metric] for r in rows if r['method'] == method and r['ratio'] == ratio)
                radius = 1-value if metric == 'lpips' else value/(limit if metric == 'unique_total' else (40 if metric == 'psnr' else 1))
                angle = math.pi/2 - (index%11)*2*math.pi/11
                expected.extend([x+(radius*math.cos(angle)+1.3)/2.6*w,
                                 y+(1.3-radius*math.sin(angle))/2.6*h])
            assert len(actual) == 24 and max(abs(a-b) for a,b in zip(actual,expected)) < 1e-5
        with Image.open(ROOT / f'figures/COMBINED_{metric.upper()}_RADAR.png') as png:
            assert all(abs(dpi-450) < .1 for dpi in png.info['dpi'])
            png.verify()


def main():
    rows, inputs, decimal_error = aggregate()
    before = {str(path): digest(path) for path in inputs}
    limit = make_figures(rows)
    verify_svg(rows, limit)
    assert before == {str(path): digest(path) for path in inputs}
    receipt = dict(complete=True, source_hashes=before, original_metrics_sha256=SOURCE_SHA,
                   methods=LABELS, group_counts=SOURCES, groups=32, per_group_rows=704, mean_rows=22,
                   weighting='全部32个场景／模式各占1/32，不先对三个数据集等权平均，不按轨迹帧数加权',
                   dataset_weights={'Replica': .5, 'TUM': .3125, 'VECtor': .1875},
                   max_decimal_mean_error=decimal_error, svg_curve_paths_verified=8,
                   count_definition='局部：同场景／模式／Ratio所有评价视角选中母Gaussian ID并集；Top-K：固定保留集大小',
                   count_axis_max=limit, quality_scale='PSNR 0–40，SSIM 0–1，LPIPS反向1–0；等距刻度',
                   new_training=False, new_rendering=False, png_dpi=450,
                   artifacts={str(p.relative_to(ROOT)): digest(p) for p in (ROOT/'figures').glob('*') if p.is_file()})
    write_json(ROOT / 'VERIFIED.json', receipt)
    (ROOT / 'README_结果说明.md').write_text('# 三数据集合并的局部方法与CaRtGS Top-K对比\n\n'
        'Replica 8场景×2模式=16组，TUM 5场景×2模式=10组，VECtor 3场景×2模式=6组，共32组。'
        '每组先使用既有全轨迹平均指标，再对32组直接算术平均。每组权重1/32；数据集贡献分别为50%、31.25%、18.75%。'
        '没有按帧数加权，也没有将三个数据集均值各赋1/3权重。PSNR按已有dB值平均。\n\n'
        '只展示两个方法：金色CaRtGS Opacity Top-K保留，绿色我们局部视角Ratio。'
        'Top-K保留最高不透明度的K个点，K=floor(ratio×CaRtGS自身地图N)。质量沿用未校正原图的既有评价。'
        '同Ratio下各方法实际点数不同，各自估计轨迹也不同，因此不是固定相机位姿、同K对照。\n\n'
        'PSNR、SSIM、LPIPS各一张独立扇形雷达图，高斯点数单独一张。质量保持原等距刻度：'
        'PSNR 0–40，SSIM 0–1，LPIPS反向1–0；越外质量越好。点数图使用线性刻度，越外仅表示点数更多。\n\n'
        '局部总数U=|∪ᵥ Sᵥ|：每个场景／相机模式／Ratio分别对所有评价视角选中的Master Gaussian ID去重，'
        '重复出现或补偿坐标改变仍只算一个。Top-K使用固定保留集合的大小。随后将32组总数平均。'
        'U不是单帧平均渲染量，也不是Alpha实际贡献点数。原逐视角平均K和地图N保留在CSV中。\n\n'
        '输出4张单图和1张总览；PNG为450 DPI，另有SVG和PDF。逐值检查原始指标、352项并集统计来源、'
        '32组完整覆盖、Decimal独立均值及8条SVG曲线坐标。全部旧文件只读复用，没有训练或重新渲染。\n', encoding='utf-8')
    archive_path = ROOT / 'COMBINED_LOCAL_VS_CARTGS_TOPK.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for folder in ['figures', 'data']:
            for path in (ROOT/folder).rglob('*'):
                if path.is_file():
                    archive.write(path, path.relative_to(ROOT))
        for path in [Path(__file__), ROOT/'VERIFIED.json', ROOT/'README_结果说明.md', STYLE]:
            archive.write(path, path.name)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
    print('COMBINED_RADAR_COMPLETE', len(rows), 'means; 4 standalone charts + overview;', 'decimal_error', decimal_error, flush=True)


if __name__ == '__main__':
    main()

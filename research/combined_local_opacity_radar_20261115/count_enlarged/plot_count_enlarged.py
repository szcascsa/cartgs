"""放大Gaussian总数图，附低Ratio局部放大；两个面板均使用线性点数刻度。"""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import plot_combined as prior
import numpy as np

base = prior.base
RATIOS = list(base.RATIOS)
LIMITS = {'all': 42000, 'zoom': 6000}


def draw(ax, rows, ratios, limit, panel):
    base.RATIOS = ratios
    ticks = [10000, 20000, 30000, 40000] if panel == 'all' else [1000, 2000, 3000, 4000, 5000, 6000]
    base.METRICS['unique_total'] = ('Gaussian总数U', '点', limit, ticks)
    base.radar(ax, rows, 'unique_total', 'all')
    for index, text in enumerate(ax.texts):
        if index < len(ticks):
            text.set_text(f'{ticks[index]/1000:g}k')
            text.set_fontsize(12 if panel == 'all' else 11)
        else:
            text.set_fontsize(15 if panel == 'all' else 13)
    for method in prior.METHODS:
        line = next(line for line in ax.lines if line.get_color() == prior.COLORS[method]
                    and len(line.get_xdata()) == len(ratios)+1)
        line.set_gid(f'count_{panel}_{method}')
        line.set_linewidth(2.8)
        line.set_markersize(7)
        line.set_markeredgewidth(1.7)
    ax.set_title('全部Ratio · 外圈42k' if panel == 'all' else '1%–10%局部放大 · 外圈6k',
                 fontsize=16 if panel == 'all' else 14, fontweight='bold', pad=9, color='#26343F')


def verify_svg(path, rows):
    ns = {'s': 'http://www.w3.org/2000/svg'}
    root = ET.parse(path).getroot()
    for panel, ratios in [('all', RATIOS), ('zoom', RATIOS[:5])]:
        for method in prior.METHODS:
            line = root.find(f".//s:g[@id='count_{panel}_{method}']/s:path", ns)
            actual = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?(?:e[-+]?\d+)?', line.attrib['d'])]
            clip = re.fullmatch(r'url\(#(.+)\)', line.attrib['clip-path']).group(1)
            box = root.find(f".//s:clipPath[@id='{clip}']/s:rect", ns).attrib
            x, y, width, height = [float(box[k]) for k in ['x', 'y', 'width', 'height']]
            expected = []
            for index, ratio in enumerate(ratios + ratios[:1]):
                value = next(r['unique_total'] for r in rows if r['method'] == method and r['ratio'] == ratio)
                radius = value / LIMITS[panel]
                assert 0 <= radius <= 1
                angle = math.pi/2-(index % len(ratios))*2*math.pi/len(ratios)
                expected.extend([x+(radius*math.cos(angle)+1.3)/2.6*width,
                                 y+(1.3-radius*math.sin(angle))/2.6*height])
            assert len(actual) == len(expected)
            assert max(abs(a-b) for a,b in zip(actual, expected)) < 1e-5
    tables = [g for g in root.findall('.//s:g', ns) if g.attrib.get('id', '').startswith('table_')]
    assert len(tables) == 1
    labels = [''.join(t.itertext()) for t in tables[0].findall('.//s:text', ns)]
    expected = ['Ratio']+[f'{r*100:g}%' for r in RATIOS]
    for method, label in zip(prior.METHODS, ['CaRtGS Top-K', '我们局部']):
        expected += [label]+[f"{next(r['unique_total'] for r in rows if r['method']==method and r['ratio']==ratio)/1000:.2f}"
                               for ratio in RATIOS]
    assert labels == expected


def main():
    source = ROOT.parent / 'data/COMBINED_MEANS.csv'
    original_receipt = ROOT.parent / 'VERIFIED.json'
    protected = [source, original_receipt, *list((ROOT.parent/'figures').glob('*'))]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    rows = list(csv.DictReader(source.open(encoding='utf-8-sig')))
    assert len(rows) == 22 and {r['method'] for r in rows} == set(prior.METHODS)
    for row in rows:
        row['ratio'], row['unique_total'] = float(row['ratio']), float(row['unique_total'])
        assert int(row['scene_mode_groups']) == 32
    base.METHODS = prior.METHODS
    base.LABELS.update(prior.LABELS)
    base.COLORS.update(prior.COLORS)
    base.MARKERS['ours_view_ratio'] = '^'
    base.configure_font()
    figures = ROOT / 'figures'
    figures.mkdir(exist_ok=True)
    (figures/'data-manifest.md').write_text('# 放大图数据清单\n\n'
        '真实数据直接读取原data/COMBINED_MEANS.csv，保持32组等权均值和局部跨视角去重总数U。'
        '主图线性刻度0–42000，局部放大图线性刻度0–6000，仅展示1%、3%、5%、8%、10%。'
        '两个面板的径向刻度不同，已在标题注明；没有改变数值、截断数据或使用非线性变换。\n', encoding='utf-8')
    fig = base.plt.figure(figsize=(15.5, 10.5))
    fig.suptitle('三数据集合并 · Gaussian总数U', fontsize=25, fontweight='bold', y=.975, color='#22313B')
    fig.text(.5, .927, 'Replica 16组 + TUM 10组 + VECtor 6组 · 全部32组等权平均',
             ha='center', fontsize=13, color='#52636F')
    draw(fig.add_axes([.005, .245, .65, .65]), rows, RATIOS, LIMITS['all'], 'all')
    draw(fig.add_axes([.68, .395, .305, .44]), rows, RATIOS[:5], LIMITS['zoom'], 'zoom')
    fig.text(.832, .355, '右图单独放大低Ratio区域\n两图均使用线性刻度；k = 千点',
             ha='center', va='center', fontsize=12, linespacing=1.7, color='#52636F')
    base.RATIOS = RATIOS
    fig.legend(handles=base.legend_handles(), loc='lower center', bbox_to_anchor=(.5, .207),
               ncol=2, frameon=False, fontsize=14, columnspacing=3)
    fig.text(.5, .195, '跨视角去重总数U（千点）', ha='center', fontsize=12, color='#33434F')
    prior.table(fig, rows, 'unique_total', [.045, .080, .91, .095], fontsize=11.5)
    fig.text(.5, .043, '局部：所有评价视角选中母Gaussian ID取并集，同一点只计一次；Top-K：固定保留集大小。',
             ha='center', fontsize=11, color='#52636F')
    output = figures/'COMBINED_GAUSSIAN_COUNT_ENLARGED'
    base.save(fig, output)
    verify_svg(output.with_suffix('.svg'), rows)
    assert before == {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}
    with prior.Image.open(output.with_suffix('.png')) as image:
        assert all(abs(dpi-450) < .1 for dpi in image.info['dpi'])
        dimensions = image.size
    (ROOT/'VERIFIED.json').write_text(json.dumps(dict(complete=True,source_sha256=before[str(source)],
        data_unchanged=True,original_files_unchanged=True,linear_scales=LIMITS,svg_paths_verified=4,
        table_values_verified=22,png_dimensions=dimensions,png_dpi=450),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('ENLARGED_COUNT_VERIFIED',dimensions,'4 paths,22 table values; original data unchanged',flush=True)


if __name__ == '__main__':
    main()

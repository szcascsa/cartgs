"""论文图：横轴为真实ratio，纵轴为2000视角的平均纯渲染加速比。"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = 'OFFICE2_MODE182_RATIO_VS_RENDERING_SPEEDUP'
RATIOS = [.01, .03, .05, .08, .1, .2, .3, .4, .5, .6, .7]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    source, summary = ROOT/'reports/PER_VIEW_SPEEDUP.csv', ROOT/'reports/SUMMARY.csv'
    analysis = ROOT/'reports/ANALYSIS_VALIDATION.json'
    validation = json.loads(analysis.read_text(encoding='utf-8'))
    assert validation['passed'] and sha(source) == validation['data_sha256']
    assert validation['renderer_mode'] == 182 and validation['native_resolution'] == [1200, 680]
    protected = [source, summary, analysis, ROOT/'formal/TIMINGS.csv']
    before = {str(path): sha(path) for path in protected}
    records = read(source)
    previous = {float(row['ratio']): row for row in read(summary)}
    rows = []
    for ratio in RATIOS:
        values = sorted((row for row in records if float(row['ratio']) == ratio), key=lambda row: int(row['frame']))
        assert [int(row['frame']) for row in values] == list(range(2000))
        full = np.asarray([float(row['full_raster_ms']) for row in values])
        current = np.asarray([float(row['ratio_raster_ms']) for row in values])
        speeds = np.asarray([float(row['raster_speedup']) for row in values])
        assert np.array_equal(speeds, full/current)
        result = dict(ratio=ratio, mean_speedup=float(speeds.mean()),
            p05_speedup=float(np.percentile(speeds, 5)), p95_speedup=float(np.percentile(speeds, 95)),
            views=2000, repeats_per_view=3, full_mean_raster_ms=float(full.mean()),
            ratio_mean_raster_ms=float(current.mean()), mean_K=float(np.mean([int(row['K']) for row in values])))
        for key, old_key in [('mean_speedup', 'mean_per_view_raster_speedup'),
                             ('p05_speedup', 'raster_speedup_p05'), ('p95_speedup', 'raster_speedup_p95')]:
            assert result[key] == float(previous[ratio][old_key]), (ratio, key)
        rows.append(result)
    data = HERE/f'{NAME}.csv'
    with data.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = dict(real_data=True, dataset='Replica office2 RGB-D', resolution=[1200, 680],
        renderer='JointReleaseRenderer(mode=182, reuse=True, fused=True, finish=True)',
        x='真实ratio，线性坐标，按视角名义Anchor候选池计算预算',
        y='先按每视角三次计时中位数计算Full/ratio，再对2000视角取算术平均',
        band='视角加速比的第5至95百分位区间；表示视角变化，不是置信区间',
        scope='纯渲染，不含相机准备、Selector、Compensator；不使用先前平滑值',
        sources=before, data=str(data), data_sha256=sha(data),
        script=str(Path(__file__)), script_sha256=sha(Path(__file__)),
        interpolation='只用直线连接真实测量点，无ratio插值或平滑',
        size_inches=[3.5, 2.7], dpi=600)
    (HERE/'data-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (HERE/'data-manifest.md').write_text('# 论文ratio–加速比图数据清单\n\n'
        '|Figure|Data file|Real/mock|Source|Script|Outputs|\n|---|---|---|---|---|---|\n'
        f'|{NAME}|{NAME}.csv|真实测量|reports/PER_VIEW_SPEEDUP.csv、SUMMARY.csv|plot_ratio_speedup.py|PDF、SVG、600dpi PNG|\n\n'
        '横轴是真实ratio，采用线性比例；每个标记对应已测ratio，直线连接测量点，不补造ratio样本。'
        '纵轴为2000个视角的Full/ratio纯渲染加速比算术平均；每视角先分别对三次计时取中位数。'
        '浅色区域是5%–95%视角分位区间，展示不同视角的变化，不能称为均值置信区间。'
        '未使用此前时间轴平滑数据，原始计时和统计结果保持不变。\n', encoding='utf-8')
    family = 'Times New Roman' if any(font.name == 'Times New Roman' for font in font_manager.fontManager.ttflist) else 'DejaVu Serif'
    plt.rcParams.update({'font.family': family, 'font.size': 9, 'axes.labelsize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
        'xtick.major.width': .7, 'ytick.major.width': .7, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 600})
    x = np.asarray([row['ratio'] for row in rows])
    y = np.asarray([row['mean_speedup'] for row in rows])
    low, high = np.asarray([row['p05_speedup'] for row in rows]), np.asarray([row['p95_speedup'] for row in rows])
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    band = ax.fill_between(x, low, high, facecolor='#0072B2', alpha=.13, linewidth=0,
        label='5th–95th percentile', zorder=1)
    line, = ax.plot(x, y, color='#0072B2', lw=1.5, marker='o', ms=3.2,
        markerfacecolor='white', markeredgewidth=.9, label='Ours', zorder=3)
    reference = ax.axhline(1., color='#6B6B6B', ls=(0, (4, 3)), lw=.9, label='Full (1×)', zorder=2)
    assert np.array_equal(line.get_xdata(), x) and np.array_equal(line.get_ydata(), y)
    ax.set(xlabel='Ratio', ylabel='Rendering speedup (×)', xlim=(0, .72), ylim=(.85, 3.75))
    ax.set_title('Rendering Speedup', fontsize=11, pad=9)
    ax.set_xticks(np.arange(0, .71, .1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_yticks(np.arange(1., 3.6, .5))
    ax.tick_params(direction='out', length=3, pad=2)
    ax.grid(axis='y', color='#D8D8D8', lw=.5, alpha=.65, zorder=0)
    ax.legend(handles=[line, band, reference], loc='upper right', frameon=False,
        handlelength=1.8, handletextpad=.5, labelspacing=.35, borderaxespad=.5)
    fig.subplots_adjust(left=.18, right=.98, bottom=.18, top=.88)
    outputs = []
    for ext in ['pdf', 'svg', 'png']:
        path = HERE/f'{NAME}.{ext}'
        fig.savefig(path, dpi=600, facecolor='white')
        outputs.append(path)
    plt.close(fig)
    assert before == {str(path): sha(path) for path in protected}
    (HERE/'caption_zh.md').write_text('Replica office2 RGB-D在1200×680分辨率下，mode182渲染器相对于同一地图Full渲染的加速比。'
        '折线表示2000个视角的平均加速比，阴影表示视角间第5至95百分位区间。'
        '每个视角使用三次单图纯渲染计时的中位数；不包含准备、Selector和Compensator开销。'
        'ratio按当前视角的名义Anchor候选池计算，Full使用全部43197个未补偿Master高斯。\n', encoding='utf-8')
    result = dict(passed=True, points=len(rows), views_per_point=2000, source_files_unchanged=True,
        font=family, exact_source_statistics=True, linear_x_axis=True, smoothing=False,
        output_sha256={path.name: sha(path) for path in outputs})
    (HERE/'FIGURE_VALIDATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

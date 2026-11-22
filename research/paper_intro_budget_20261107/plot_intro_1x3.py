"""1×3横向引言图，标准差以水平界线表示，CV直接标注。"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import analyze_plot as source_plot


METHODS = ['cartgs_full', 'ratio_50', 'fixed_2000', 'fixed_4000']
LABELS = {
    'cartgs_full': 'CaRtGS',
    'ratio_50': r'Ours ($r=0.5$)',
    'fixed_2000': 'Ours (2k points)',
    'fixed_4000': 'Ours (4k points*)',
}


def main():
    source = Path(__file__).resolve().parent / 'results/reports/four_methods_cv'
    output = source.parent / 'intro_1x3'
    output.mkdir(exist_ok=True)
    inputs = [source / 'PLOT_DATA.csv', source / 'SUMMARY.json']
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    rows = source_plot.read(inputs[0])
    summary = json.loads(inputs[1].read_text(encoding='utf-8'))
    by_method = {row['method']: row for row in summary}
    data = {}
    for method in METHODS:
        values = sorted(
            [row for row in rows if row['method'] == method],
            key=lambda row: int(row['frame']),
        )
        assert [int(row['frame']) for row in values] == list(range(792))
        data[method] = {
            key: np.asarray([float(row[key]) for row in values])
            for key in ('contributing_K', 'psnr_smooth', 'total_ms')
        }
        times = data[method]['total_ms']
        assert abs(times.std(ddof=0) - by_method[method]['total_ms_std']) < 1e-12
        assert abs(100 * times.std(ddof=0) / times.mean() - by_method[method]['per_view_latency_cv_percent']) < 1e-12

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 7,
        'axes.titlesize': 7.8,
        'axes.labelsize': 7,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': .65,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        'xtick.major.width': .6,
        'ytick.major.width': .6,
        'pdf.fonttype': 42,
        'svg.fonttype': 'path',
    })
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.65), sharex=True,
    )
    counts, quality, latency = axes
    x = np.arange(792)
    legend_handles = []
    checked_lines = 0
    for method in METHODS:
        color = source_plot.COLORS[method]
        style = source_plot.STYLES[method]
        values = data[method]
        for axis, key, divisor in (
            (counts, 'contributing_K', 1000),
            (quality, 'psnr_smooth', 1),
            (latency, 'total_ms', 1),
        ):
            line = axis.plot(
                x, values[key] / divisor, color=color, ls=style,
                lw=1.0 if axis is not latency else .65,
                zorder=3,
            )[0]
            assert np.array_equal(line.get_ydata(), values[key] / divisor)
            checked_lines += 1
        mean = by_method[method]['total_ms_mean']
        std = by_method[method]['total_ms_std']
        for bound in (mean - std, mean + std):
            line = latency.axhline(bound, color=color, ls=(0, (4, 3)), lw=.65, alpha=.7, zorder=2)
            assert np.array_equal(line.get_ydata(), np.asarray([bound, bound]))
        legend_handles.append(Line2D([], [], color=color, ls=style, lw=1.3, label=LABELS[method]))

    positions = {
        'cartgs_full': (.56, .96),
        'ratio_50': (.70, .64),
        'fixed_4000': (.70, .49),
        'fixed_2000': (.70, .22),
    }
    for method in METHODS:
        mean = by_method[method]['total_ms_mean']
        cv = by_method[method]['per_view_latency_cv_percent']
        latency.annotate(
            f'CV {cv:.2f}%', xy=(775, mean), xycoords='data',
            xytext=positions[method], textcoords='axes fraction',
            color=source_plot.COLORS[method], fontsize=6.4,
            ha='center', va='center',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9, 'pad': 1.2},
            arrowprops={'arrowstyle': '-', 'color': source_plot.COLORS[method], 'lw': .65},
        )

    counts.set(title='(a) Fewer contributing Gaussians', ylabel='Gaussians (k)', ylim=(0, 32))
    counts.set_yticks([0, 10, 20, 30])
    quality.set(title='(b) Rendering quality', ylabel='PSNR (dB)')
    latency.set(title='(c) Total rendering latency', ylabel='Total time (ms)', xlabel='RGB frame index', ylim=(0, 2.0))
    latency.set_yticks([0, .5, 1, 1.5, 2])
    for axis in axes:
        axis.set_xlim(0, 791)
        axis.set_xticks([0, 250, 500, 750])
        axis.set_xlabel('RGB frame index')
        axis.grid(alpha=.18, lw=.45)
    fig.legend(
        handles=legend_handles, loc='upper center', bbox_to_anchor=(.52, .995),
        ncol=4, frameon=False, fontsize=7, handlelength=2.5,
        columnspacing=1.7, labelspacing=.65,
    )
    fig.subplots_adjust(left=.065, right=.99, top=.80, bottom=.23, wspace=.34)
    fig.text(.5, .025, r'Dashed bounds: mean $\pm$ 1 SD. Latency unsmoothed. *4k: one timing repeat excluded; all 792 views retained.',
        fontsize=6, ha='center')

    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'PAPER_INTRO_1X3_EN.{extension}', dpi=600, facecolor='white')
    plt.close(fig)
    assert len(axes) == 3 and checked_lines == 12
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == value for name, value in hashes.items())
    source_plot.save(output / 'FIGURE_PROTOCOL.json', dict(
        language='English', layout='one row and three columns', width_inches=7.2,
        height_inches=2.65, methods=METHODS, frames=792, count='post-render Alpha contributors',
        psnr_smoothing='unchanged Gaussian sigma=35 frames', timing_smoothing=False,
        standard_deviation='two horizontal bounds at mean minus and plus one population standard deviation',
        cv='direct labels, rounded to two decimal places', fixed4000='updated data excluding one repeat at frame 0',
        source_hashes=hashes, verified_curve_arrays=12,
    ))
    (output / 'FIGURE_NOTES.md').write_text(
        '# 1×3英文引言图\n\n'
        '图宽7.2英寸、高2.65英寸，三个横向面板分别为实际Alpha贡献点数、强平滑PSNR和不平滑总耗时。'
        '顶部为四方法统一图例。耗时面板以各方法均值±1个总体标准差的两条水平虚线表示波动范围，CV直接标注。\n\n'
        '固定4000点使用更新后的792视角数据，仅第0帧删除一次异常重复测量。'
        '原始PSNR与点数、其他时间数据、已有PSNR平滑结果均复用已验证输入。'
        'CV计算基于各视角重复测量均值，sigma采用ddof=0。\n',
        encoding='utf-8',
    )
    print('INTRO_1X3_FIGURE_PASS', output)


if __name__ == '__main__':
    main()

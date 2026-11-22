"""从已验收数据生成三方法论文图；只调整展示，不重新测量。"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ['cartgs_full', 'ours_full', 'dynamic_04']
LABELS = ['CaRtGS', 'Ours Full', 'Ours (dynamic ratio)']
COLORS = ['#252525', '#00A6C8', '#D55E00']
STYLES = ['-', '-.', '-']
NAME = 'OFFICE2_DYNAMIC_RATIO_THREE_METHODS'


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    source = ROOT/'reports/PLOT_DATA.csv'
    validation = json.loads((ROOT/'reports/ANALYSIS_VALIDATION.json').read_text())
    assert validation['passed'] and validation['frames'] == 2000
    original = read(source)
    output = ROOT/'figures'
    data_dir = output/'data'
    data_dir.mkdir(exist_ok=True)
    selected = [row for row in original if row['method'] in METHODS]
    assert len(selected) == 6000
    data_path = data_dir/'office2_dynamic_ratio_three_methods.csv'
    write(data_path, selected)
    data, summary = {}, []
    for method, label in zip(METHODS, LABELS):
        rows = sorted((row for row in selected if row['method'] == method), key=lambda row: float(row['frame']))
        assert [int(float(row['frame'])) for row in rows] == list(range(2000))
        values = {key: np.asarray([float(row[key]) for row in rows]) for key in ['frame', 'K', 'contributing_K', 'psnr', 'psnr_smooth', 'raster_ms']}
        assert all(np.isfinite(array).all() for array in values.values())
        data[method] = values
        ms = values['raster_ms']
        summary.append(dict(method=method, label=label, frames=2000, submitted_K_mean=float(values['K'].mean()),
            contributing_K_mean=float(values['contributing_K'].mean()), psnr_mean=float(values['psnr'].mean()),
            render_ms_mean=float(ms.mean()), render_ms_std=float(ms.std(ddof=0)),
            render_ms_cv_percent=float(100*ms.std(ddof=0)/ms.mean())))
    write(ROOT/'reports/THREE_METHODS_SUMMARY.csv', summary)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.titlesize': 11,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'path', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(12.6, 4.4))
    grid = fig.add_gridspec(1, 3, left=.067, right=.985, bottom=.235, top=.79, wspace=.35)
    count = fig.add_subplot(grid[0, 0])
    quality = fig.add_subplot(grid[0, 1], sharex=count)
    inner = grid[0, 2].subgridspec(2, 1, height_ratios=[1, 2.1], hspace=.095)
    upper = fig.add_subplot(inner[0], sharex=count)
    lower = fig.add_subplot(inner[1], sharex=count)
    upper.set_ylim(3.35, 5.4)
    lower.set_ylim(.285, 1.12)
    upper.spines['bottom'].set_visible(False)
    upper.tick_params(axis='x', bottom=False, labelbottom=False)
    for axis, y in [(upper, 0), (lower, 1)]:
        axis.plot([-.015, .015], [y-.025, y+.025], transform=axis.transAxes,
            clip_on=False, color='#252525', lw=.9)
    for method, color, style, stats in zip(METHODS, COLORS, STYLES, summary):
        values = data[method]
        for axis, array in [(count, values['contributing_K']/1000), (quality, values['psnr_smooth']),
                            (upper if method == 'cartgs_full' else lower, values['raster_ms'])]:
            line, = axis.plot(values['frame'], array, color=color, ls=style,
                lw=1.45 if axis in [count, quality] else .75)
            assert np.array_equal(line.get_ydata(), array)
        axis = upper if method == 'cartgs_full' else lower
        for bound in [stats['render_ms_mean']-stats['render_ms_std'], stats['render_ms_mean']+stats['render_ms_std']]:
            axis.axhline(bound, color=color, lw=.8, ls=(0, (4, 3)), alpha=.65)
    # CV以2000帧各自三次计时中位数计算；不将其表述为单次调用波动。
    annotations = [('cartgs_full', upper, (940, 3.55), 1500),
                   ('ours_full', lower, (1010, 1.04), 1740),
                   ('dynamic_04', lower, (1020, .535), 1710)]
    for method, axis, position, frame in annotations:
        i = METHODS.index(method)
        axis.annotate(f'CV {summary[i]["render_ms_cv_percent"]:.2f}%',
            xy=(frame, data[method]['raster_ms'][frame]), xytext=position,
            color=COLORS[i], fontsize=9, arrowprops=dict(arrowstyle='-', color=COLORS[i], lw=.85))
    count.set(title='(a) Contributing Gaussians', ylabel='Gaussians (k)', ylim=(0, 52.5))
    quality.set(title='(b) Rendering quality', ylabel='PSNR (dB)')
    upper.set_title('(c) Rendering latency')
    fig.text(.699, .51, 'Rendering time (ms)', rotation=90, ha='center', va='center', fontsize=10.5)
    upper.set_yticks([3.5, 4.5, 5.3])
    lower.set_yticks([.3, .5, .7, .9, 1.1])
    for axis in [count, quality, upper, lower]:
        axis.set_xlim(0, 1999)
        axis.set_xticks([0, 500, 1000, 1500, 1999])
        axis.grid(alpha=.13, lw=.65)
    for axis in [count, quality, lower]:
        axis.set_xlabel('RGB frame index')
    handles = [Line2D([], [], color=color, ls=style, lw=1.65, label=label)
               for color, style, label in zip(COLORS, STYLES, LABELS)]
    fig.legend(handles=handles, ncol=3, loc='upper center', bbox_to_anchor=(.52, .982),
        frameon=False, handlelength=2.8, columnspacing=2.8)
    fig.text(.525, .088, 'Dashed bounds: mean ± 1 SD. PSNR smoothed (σ = 35 frames); latency unsmoothed.', ha='center', fontsize=8.5)
    fig.text(.525, .043, 'Pure rendering only; median of 3 single-image repeats per view.', ha='center', fontsize=8.5)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(output/f'{NAME}.{extension}', dpi=450, bbox_inches='tight')
    plt.close(fig)
    svg = (output/f'{NAME}.svg').read_text(encoding='utf-8')
    assert all(label in svg for label in LABELS)
    assert not any(text in svg for text in ['0.4 ms', '0.4ms', 'Ours 50%', 'Ours 2k', 'Ours 4k'])
    manifest = dict(real_data=True, dataset='Replica/office2/rgbd', width=2064, height=2208, renderer='Mode40',
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        selected_data=str(data_path), selected_data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), frames=2000, methods=METHODS,
        labels=LABELS, psnr_sigma_frames=35, timing_smoothing=False, count_smoothing=False, dpi=450,
        measurement='pure rendering; per-frame median of 3 independent calls',
        dynamic_budget_provenance='沿用已有离线逐帧冻结预算；本次仅精简图表，不代表实时控制器。')
    (output/f'{NAME}_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/'data-manifest-three-methods.md').write_text('# 三方法图数据清单\n\n'
        '| 图表 | 数据 | 性质 | 来源 | 脚本 | 输出 |\n|---|---|---|---|---|---|\n'
        f'| {NAME} | data/office2_dynamic_ratio_three_methods.csv | 真实测量 | reports/PLOT_DATA.csv | plot_three_methods.py | PNG/SVG/PDF |\n\n'
        '仅保留CaRtGS、Ours Full和动态ratio。数据覆盖完整2000帧；沿用已验收的同轮纯渲染计时。'
        'PSNR曲线沿用σ=35帧平滑；点数、耗时不平滑；虚线为逐帧三次中位数的均值±1个总体标准差。'
        '动态预算来源仍为离线逐帧搜索，不因图例改名而成为实时控制结果。\n', encoding='utf-8')
    result = dict(passed=True, methods=LABELS, frames_per_method=2000, rows=6000,
        exact_source_arrays=True, target_annotation_absent=True, png_svg_pdf=True, dpi=450)
    (ROOT/'reports/THREE_METHODS_FIGURE_VALIDATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(validation=result, summary=summary), ensure_ascii=False))


if __name__ == '__main__':
    main()

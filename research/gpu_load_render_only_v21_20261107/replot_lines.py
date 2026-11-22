"""将既有v21时序绘制成时间分箱均值折线。"""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
RUN = HERE / 'results/runs/formal01'
OUT = RUN / 'reports/line_plots'
BIN_SECONDS = .25
METHODS = ('cartgs_full', 'ours_fixed4000', 'ours_adaptive')
LABELS = ('CaRtGS Full', 'Ours, fixed 4k', 'Ours, adaptive K')
COLORS = ('#36393F', '#D55E00', '#0072B2')
METRICS = ('prepare_ms', 'render_ms', 'total_ms')


def read(path):
    with path.open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global RUN, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN)
    RUN = parser.parse_args().run
    OUT = RUN / 'reports/line_plots'
    inputs = [RUN / name for name in ('TIMINGS.csv', 'QUALITY.csv', 'ENVIRONMENT.json', 'reports/STAGE_SUMMARY.csv')]
    before = {str(path): digest(path) for path in inputs}
    rows = read(RUN / 'TIMINGS.csv')
    quality = {(value['method'], int(value['budget'])): float(value['psnr'])
        for value in read(RUN / 'QUALITY.csv')}
    environment = json.loads((RUN / 'ENVIRONMENT.json').read_text())
    OUT.mkdir(exist_ok=True)
    aggregated = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    edges = np.arange(7) * environment['stage_seconds']
    for repeat in range(3):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
        axes = axes.ravel()
        axes[0].step(edges, [100*value for value in environment['duties']]+[0],
            where='post', color='#6B7280', lw=2)
        for method, color, label in zip(METHODS, COLORS, LABELS):
            selected = [row for row in rows if row['method'] == method and int(row['repeat']) == repeat]
            times = np.array([float(row['elapsed_s']) for row in selected])
            axes[1].step(times, [int(row['K']) for row in selected], where='post',
                color=color, lw=1.2, label=label)
            psnr = [quality['cartgs_full', 35301] if method == 'cartgs_full'
                else quality['ours', int(row['K'])] for row in selected]
            axes[5].step(times, psnr, where='post', color=color, lw=1.1)
            bins = np.floor(times/BIN_SECONDS).astype(int)
            for metric, axis in zip(METRICS, (axes[2], axes[3], axes[4])):
                values = np.array([float(row[metric]) for row in selected])
                centers, means, lower, upper = [], [], [], []
                for index in sorted(set(bins)):
                    group = values[bins == index]
                    mean = float(group.mean())
                    p10, p90 = np.percentile(group, (10, 90))
                    centers.append((index+.5)*BIN_SECONDS)
                    means.append(mean)
                    lower.append(p10)
                    upper.append(p90)
                    aggregated.append(dict(repeat=repeat, method=method, metric=metric,
                        bin_start_s=index*BIN_SECONDS, bin_end_s=(index+1)*BIN_SECONDS,
                        samples=len(group), mean=mean, p10=float(p10), p90=float(p90)))
                axis.fill_between(centers, lower, upper, color=color, alpha=.09, linewidth=0)
                axis.plot(centers, means, color=color, lw=1.8)
        axes[0].set(title='Injected load during rendering', ylabel='Target duty (%)', ylim=(0, 100))
        axes[1].set(title='Actual input Gaussians', ylabel='K (log scale)', yscale='log', ylim=(850, 43000))
        axes[1].yaxis.set_major_locator(FixedLocator([1000, 2000, 4000, 10000, 35301]))
        axes[1].yaxis.set_major_formatter(FixedFormatter(['1k', '2k', '4k', '10k', '35.3k']))
        axes[1].minorticks_off()
        axes[2].set(title='Preparation', ylabel='Mean latency (ms)', ylim=(0, None))
        axes[3].set(title='Rendering', ylabel='Mean latency (ms)', ylim=(0, None))
        axes[4].set(title='Preparation + rendering', ylabel='Mean latency (ms)', ylim=(0, None))
        axes[4].axhline(environment['target_ms'], color='#B91C1C', lw=1.1, ls='--')
        axes[5].set(title='Quality at the active budget', ylabel='PSNR (dB)')
        for axis in axes:
            axis.set(xlabel='Elapsed time (s)', xlim=(0, edges[-1]))
            axis.grid(alpha=.18)
            for boundary in edges[1:-1]:
                axis.axvline(boundary, color='#AAAAAA', lw=.7, ls=':')
        handles = [Line2D([], [], color=color, lw=2, label=label)
            for color, label in zip(COLORS, LABELS)]
        handles.append(Line2D([], [], color='#B91C1C', lw=1.1, ls='--',
            label=f"Target {environment['target_ms']:.3f} ms"))
        fig.suptitle(f'v21 | Render-only injected load | Run {repeat+1}/3', fontsize=12, y=.995)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .954),
            ncol=4, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, .055, 1, .89))
        fig.text(.5, .025,
            'Latency lines: 0.25 s means; bands: P10-P90. K and PSNR: original step sequences.',
            ha='center', fontsize=8, color='#555555')
        fig.text(.5, .004,
            'Methods tested separately on a shared GPU. Gate overhead is excluded from the total-latency panel.',
            ha='center', fontsize=7.5, color='#555555')
        for extension in ('png', 'pdf'):
            fig.savefig(OUT / f'V21_LINE_REPEAT{repeat}.{extension}', dpi=220)
        plt.close(fig)
    with (OUT / 'BINNED_LATENCY.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregated[0]))
        writer.writeheader()
        writer.writerows(aggregated)
    for metric in METRICS:
        assert sum(value['samples'] for value in aggregated if value['metric'] == metric) == len(rows)
    assert before == {str(path): digest(path) for path in inputs}
    (OUT / 'PLOT_METHOD.md').write_text(
        '# v21 折线图显示口径\n\n'
        '准备、渲染和总耗时按0.25秒固定时间窗口计算均值并连线，浅色带为每窗口的P10至P90分位范围。'
        f'全部{len(rows):,}个原始请求均参与计算，点数和PSNR沿用原始阶梯序列。'
        '时间窗口仅作用于新版图表，原始数据及阶段统计哈希核验通过。\n\n'
        '三种方法在共享GPU上分开执行，曲线的时间轴对齐各自的人工负载日程。'
        '总耗时面板展示准备加渲染，负载握手开销位于原始CSV和报告中。\n', encoding='utf-8')
    (OUT / 'VALIDATION.json').write_text(json.dumps(dict(passed=True, original_requests=len(rows),
        bin_seconds=BIN_SECONDS, band_percentiles=[10, 90], aggregated_rows=len(aggregated),
        original_input_sha256=before, original_input_hashes_match=True), indent=2), encoding='utf-8')
    archive = HERE / 'V21_LINE_PLOTS.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as package:
        for path in sorted(OUT.iterdir()):
            if path.is_file():
                package.write(path, path.name)
    print(json.dumps(dict(figures=3, raw_requests=len(rows), archive=str(archive), output=str(OUT))))


if __name__ == '__main__':
    main()

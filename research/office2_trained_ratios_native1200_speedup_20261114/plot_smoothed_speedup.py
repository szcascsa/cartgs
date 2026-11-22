"""只平滑显示曲线；保留原始计时、逐帧数据、统计量和未平滑图。"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from analyze import ROOT, RATIOS, COLORS

NAME = 'OFFICE2_NATIVE1200_TRAINED_RATIOS_SPEEDUP_SMOOTHED'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT/'reports/PER_VIEW_SPEEDUP.csv'
    validation = json.loads((ROOT/'reports/ANALYSIS_VALIDATION.json').read_text())
    assert validation['passed'] and sha(source) == validation['plotted_csv_sha256']
    protected = [source, ROOT/'reports/SUMMARY.csv', ROOT/'formal/TIMINGS.csv',
        ROOT/'reports/ANALYSIS_VALIDATION.json',
        ROOT/'figures/OFFICE2_NATIVE1200_TRAINED_RATIOS_SPEEDUP_OVER_OURS_FULL.png']
    before = {str(path): sha(path) for path in protected}
    with source.open(encoding='utf-8', newline='') as stream:
        records = list(csv.DictReader(stream))
    out = ROOT/'figures'
    (out/'data').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'path'})
    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = []
    for ratio, color in zip(RATIOS, COLORS):
        rows = sorted((row for row in records if float(row['ratio']) == ratio), key=lambda row: int(row['frame']))
        frames = np.asarray([int(row['frame']) for row in rows])
        assert np.array_equal(frames, np.arange(2000))
        raw = np.asarray([float(row['speedup']) for row in rows])
        median = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(raw, 15, mode='reflect'), 31), axis=1)
        smooth = gaussian_filter1d(median, sigma=6, mode='reflect', truncate=3)
        assert np.isfinite(smooth).all() and (smooth > 0).all()
        line, = ax.plot(frames, smooth, color=color, label=f'Ratio {ratio:g}', lw=1.6)
        assert np.array_equal(line.get_ydata(), smooth)
        plotted.extend(dict(frame=int(f), ratio=ratio, raw_speedup=float(x), displayed_speedup=float(y))
            for f, x, y in zip(frames, raw, smooth))
    ax.axhline(1, color='#555555', ls='--', lw=1, label='Ours Full (1×)')
    ax.set(xlabel='RGB frame index', ylabel='Rendering speedup (×)', xlim=(0, 1999), ylim=(0, 5.5))
    ax.set_xticks([0, 500, 1000, 1500, 1999])
    ax.grid(alpha=.15)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.14), ncol=7, fontsize=9)
    fig.suptitle('Replica office2 RGB-D | 1200 × 680 (native) | Mode 40', y=.98, fontsize=13)
    fig.subplots_adjust(left=.085, right=.985, top=.81, bottom=.20)
    fig.text(.53, .055, 'Display smoothing: 31-frame centered median + Gaussian (σ = 6 frames).\n'
        'Pure rendering only; raw timings and reported statistics remain unchanged.', ha='center', va='center', fontsize=9)
    data = out/'data'/f'{NAME}.csv'
    with data.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(plotted[0]))
        writer.writeheader()
        writer.writerows(plotted)
    manifest = dict(real_data=True, source=str(source), source_sha256=sha(source), script=str(Path(__file__)),
        script_sha256=sha(Path(__file__)), data=str(data), data_sha256=sha(data), frames=2000, ratios=RATIOS,
        smoothing=dict(centered_median_window=31, median_padding='reflect', gaussian_sigma_frames=6,
            gaussian_mode='reflect', gaussian_truncate=3, scope='display speedup only'),
        raw_statistics_unchanged=True, y_limits=[0, 5.5])
    (out/f'{NAME}_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (out/f'{NAME}_data-manifest.md').write_text('# 原始分辨率加速比平滑图数据清单\n\n'
        '|Figure|Data file|Real/mock|Source|Script|Outputs|\n|---|---|---|---|---|---|\n'
        f'|{NAME}|data/{NAME}.csv|真实测量|reports/PER_VIEW_SPEEDUP.csv|plot_smoothed_speedup.py|PNG、SVG、PDF|\n\n'
        '显示值采用31帧居中中值滤波，再进行σ=6帧的Gaussian平滑，反射处理边界；不降采样。'
        '纵轴沿用0–5.5×。原始计时、统计结果及未平滑图不变。\n', encoding='utf-8')
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(out/f'{NAME}.{extension}', dpi=450, bbox_inches='tight')
    plt.close(fig)
    assert before == {str(path): sha(path) for path in protected}
    print(json.dumps(dict(passed=True, raw_files_unchanged=len(before), plotted_rows=len(plotted), figure=str(out/f'{NAME}.png'))))


if __name__ == '__main__':
    main()

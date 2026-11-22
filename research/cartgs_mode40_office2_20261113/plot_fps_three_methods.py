"""同轮Mode40数据：CaRtGS、Ours Full、动态ratio的未平滑等效FPS。"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = [('cartgs_mode40', 'CaRtGS (Mode 40)', '#252525'),
           ('ours_full', 'Ours Full', '#00A6C8'),
           ('dynamic_04', 'Ours (dynamic ratio)', '#D55E00')]
NAME = 'OFFICE2_MODE40_FPS_THREE_METHODS_UNSMOOTHED'


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source, timing = ROOT/'reports/PER_FRAME.csv', ROOT/'formal/TIMINGS.csv'
    assert json.loads((ROOT/'reports/ANALYSIS_VALIDATION.json').read_text())['passed']
    manifest = json.loads((ROOT/'figures/DATA_MANIFEST.json').read_text())
    assert sha(timing) == manifest['sources']['formal/TIMINGS.csv']
    records, samples = read(source), {}
    for row in read(timing):
        samples.setdefault((row['method'], int(row['frame'])), []).append(row)
    out = ROOT/'figures'
    (out/'data').mkdir(exist_ok=True)
    plotted, summaries = [], []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'path', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(10, 4.3))
    for method, label, color in METHODS:
        rows = sorted((r for r in records if r['method'] == method), key=lambda r: int(r['frame']))
        assert [int(r['frame']) for r in rows] == list(range(2000))
        ms = np.asarray([float(r['raster_ms']) for r in rows])
        fps = np.asarray([float(r['equivalent_fps']) for r in rows])
        assert np.array_equal(fps, 1000./ms) and np.isfinite(fps).all()
        for i in range(2000):
            repeat = samples[method, i]
            assert sorted(int(r['repeat']) for r in repeat) == [0, 1, 2]
            assert ms[i] == float(np.median([float(r['raster_ms']) for r in repeat]))
        line, = ax.plot(np.arange(2000), fps, color=color, label=label, lw=1.)
        assert np.array_equal(line.get_ydata(), fps)
        plotted.extend(dict(method=method, frame=i, pure_render_ms=float(ms[i]), equivalent_fps=float(fps[i])) for i in range(2000))
        summaries.append(dict(method=method, mean_render_ms=float(ms.mean()), equivalent_fps_from_mean_latency=float(1000/ms.mean())))
    ax.set(xlabel='RGB frame index', ylabel='Equivalent rendering speed (FPS)',
        title='Replica office2 RGB-D | 2064 × 2208 | All use Mode 40', xlim=(0, 1999), ylim=(0, 3500))
    ax.set_xticks([0, 500, 1000, 1500, 1999]); ax.set_yticks(np.arange(0, 3501, 500))
    ax.grid(alpha=.16); ax.legend(frameon=False, loc='upper center', ncol=3, fontsize=10)
    fig.subplots_adjust(left=.095, right=.98, top=.85, bottom=.23)
    fig.text(.535, .06, 'Pure rendering only; FPS = 1000 / latency (ms); no temporal smoothing.', ha='center', fontsize=9)
    data_path = out/'data'/f'{NAME}.csv'
    with data_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(plotted[0])); writer.writeheader(); writer.writerows(plotted)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(out/f'{NAME}.{extension}', dpi=450, bbox_inches='tight')
    plt.close(fig)
    result = dict(passed=True, frames_per_method=2000, methods=[m[0] for m in METHODS],
        smoothing=False, input_timing='每帧三次单图计时的中位数', pure_render_only=True,
        all_Mode40=True, sources={str(p): sha(p) for p in [source, timing, Path(__file__)]},
        data_path=str(data_path), data_sha256=sha(data_path), summary=summaries)
    (ROOT/'reports/FPS_THREE_METHODS_VALIDATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (out/f'{NAME}_data-manifest.md').write_text('# 三方法等效渲染FPS数据清单\n\n'
        f'数据：data/{NAME}.csv；源文件：reports/PER_FRAME.csv、formal/TIMINGS.csv；脚本：plot_fps_three_methods.py。\n\n'
        '三方法均为Mode40，完整2000帧，同轮三次计时取每帧中位数后转换FPS。没有时间轴平滑。'
        '纯渲染时间排除准备、网络和显示；动态预算沿用离线冻结K。导出450 DPI PNG和SVG、PDF。\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, summary=summaries), ensure_ascii=False))


if __name__ == '__main__':
    main()

"""将已验收的逐帧纯渲染耗时直接转换为等效FPS，不做平滑。"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = [('cartgs_full', 'CaRtGS', '#252525'),
           ('dynamic_04', 'Ours (dynamic ratio)', '#D55E00')]
NAME = 'OFFICE2_EQUIVALENT_RENDERING_FPS_UNSMOOTHED'


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source, timing = ROOT/'reports/PLOT_DATA.csv', ROOT/'formal/TIMINGS.csv'
    assert json.loads((ROOT/'reports/ANALYSIS_VALIDATION.json').read_text())['passed']
    manifest = json.loads((ROOT/'figures/DATA_MANIFEST.json').read_text())
    assert sha(timing) == manifest['sources']['formal\\TIMINGS.csv']
    records, measurements = read(source), read(timing)
    samples = {}
    for row in measurements:
        if row['method'] in {method for method, _, _ in METHODS}:
            samples.setdefault((row['method'], int(row['frame'])), []).append(row)
    output = ROOT/'figures'
    (output/'data').mkdir(exist_ok=True)
    data_path = output/'data'/f'{NAME}.csv'
    all_rows, summary = [], []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'svg.fonttype': 'path', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(10, 4.3))
    for method, label, color in METHODS:
        rows = sorted((r for r in records if r['method'] == method), key=lambda r: float(r['frame']))
        assert [int(float(r['frame'])) for r in rows] == list(range(2000))
        milliseconds = np.asarray([float(row['raster_ms']) for row in rows])
        assert np.isfinite(milliseconds).all() and (milliseconds > 0).all()
        for i in range(2000):
            values = samples[method, i]
            assert sorted(int(r['repeat']) for r in values) == [0, 1, 2]
            assert milliseconds[i] == float(np.median([float(r['raster_ms']) for r in values]))
        fps = 1000.0/milliseconds
        assert len(fps) == 2000 and np.allclose(fps*milliseconds, 1000.0, rtol=1e-15)
        line, = ax.plot(np.arange(2000), fps, label=label, color=color, lw=1.1)
        assert np.array_equal(line.get_ydata(), fps)
        all_rows.extend(dict(method=method, frame=i, pure_render_ms=float(milliseconds[i]),
            equivalent_fps=float(fps[i])) for i in range(2000))
        summary.append(dict(method=method, frames=2000, mean_render_ms=float(milliseconds.mean()),
            equivalent_fps_from_mean_latency=float(1000.0/milliseconds.mean()),
            mean_per_frame_equivalent_fps=float(fps.mean()), min_fps=float(fps.min()), max_fps=float(fps.max())))
    ax.set(xlabel='RGB frame index', ylabel='Equivalent rendering speed (FPS)',
        title='Replica office2 RGB-D | 2064 × 2208', xlim=(0, 1999), ylim=(0, 3200))
    ax.set_xticks([0, 500, 1000, 1500, 1999])
    ax.set_yticks(np.arange(0, 3201, 500))
    ax.grid(alpha=.16, lw=.7)
    ax.legend(frameon=False, loc='upper center', ncol=2)
    fig.subplots_adjust(left=.095, right=.98, top=.85, bottom=.23)
    fig.text(.535, .06, 'Pure rendering only; FPS = 1000 / latency (ms); no temporal smoothing.',
        ha='center', fontsize=9)
    with data_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(output/f'{NAME}.{extension}', dpi=450, bbox_inches='tight')
    plt.close(fig)
    result = dict(passed=True, dataset='Replica/office2/rgbd', width=2064, height=2208,
        frames_per_method=2000, formula='FPS = 1000 / per-frame pure render latency in milliseconds',
        input_timing='逐帧三次单图耗时中位数', smoothing=False,
        excludes='准备、网络、预算搜索及显示开销', dynamic_budget='沿用目标0.4ms的离线逐视角冻结预算',
        dpi=450, sources={str(p): sha(p) for p in [source, timing, Path(__file__)]},
        data_csv=str(data_path), data_sha256=sha(data_path), summary=summary)
    (ROOT/'reports/EQUIVALENT_FPS_UNSMOOTHED_VALIDATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/f'{NAME}_data-manifest.md').write_text('# 等效渲染FPS图数据清单\n\n'
        '| 数据 | 性质 | 来源 | 脚本 | 输出 |\n|---|---|---|---|---|\n'
        f'| data/{NAME}.csv | 真实测量转换 | reports/PLOT_DATA.csv及formal/TIMINGS.csv | plot_equivalent_fps.py | PNG/SVG/PDF |\n\n'
        '逐帧由三次计时中位数直接转换FPS，不做时间轴平滑；点数和选择预算没有改变。'
        '展示的是纯渲染的等效FPS，不代表含网络、准备和显示的完整运行帧率。\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, summary=summary), ensure_ascii=False))


if __name__ == '__main__':
    main()

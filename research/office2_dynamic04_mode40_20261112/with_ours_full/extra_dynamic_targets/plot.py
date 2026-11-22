"""Plot three offline dynamic budgets using their independently validated timings."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
METHODS = ['cartgs_full', 'ours_full', 'dynamic_04', 'dynamic_05', 'dynamic_07']
LABELS = ['CaRtGS', 'Ours Full', 'Ours (dynamic ratio, 0.4 ms)',
          'Ours (dynamic ratio, 0.5 ms)', 'Ours (dynamic ratio, 0.7 ms)']
COLORS = ['#252525', '#00A6C8', '#D55E00', '#0077BB', '#7D3C98']
STYLES = ['-', '-.', '-', '-', '-']


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    source = PARENT/'reports/PLOT_DATA.csv'
    rows = read(source)
    data = {}
    for method in METHODS[:3]:
        records = sorted((r for r in rows if r['method'] == method), key=lambda r: float(r['frame']))
        assert [int(float(r['frame'])) for r in records] == list(range(2000))
        data[method] = {key: np.array([float(r[key]) for r in records])
                        for key in ['K', 'contributing_K', 'psnr', 'psnr_smooth', 'raster_ms']}
    sources = [source, PARENT/'reports/ANALYSIS_VALIDATION.json']
    for directory, method in [('target05', 'dynamic_05'), ('target07', 'dynamic_07')]:
        folder = ROOT/directory
        formal = folder/'formal'
        validation = json.loads((formal/'VALIDATION.json').read_text())
        job = json.loads((folder/'logs/formal_job.json').read_text())
        assert validation['passed'] and job['returncode'] == 0
        assert validation['frames'] == 2000 and validation['independent_validation']
        assert hashlib.sha256((formal/'BUDGETS_FROZEN.csv').read_bytes()).hexdigest() == validation['frozen_budget_sha256']
        quality = sorted((r for r in read(formal/'QUALITY_COUNTS.csv') if r['method'] == 'dynamic_04'), key=lambda r: int(r['frame']))
        timings = [r for r in read(formal/'TIMINGS.csv') if r['method'] == 'dynamic_04']
        frozen = sorted(read(formal/'BUDGETS_FROZEN.csv'), key=lambda r: int(r['frame']))
        assert [int(r['frame']) for r in quality] == list(range(2000))
        groups = {}
        for row in timings:
            groups.setdefault(int(row['frame']), []).append(row)
        values = {key: [] for key in ['K', 'contributing_K', 'psnr', 'raster_ms']}
        for i, row in enumerate(quality):
            assert int(frozen[i]['K']) == int(row['K']) and frozen[i]['ids_sha256'] == row['ids_sha256']
            samples = groups[i]
            assert sorted(int(x['repeat']) for x in samples) == [0, 1, 2]
            assert all(int(x['K']) == int(row['K']) for x in samples)
            for key in ['K', 'contributing_K', 'psnr']:
                values[key].append(float(row[key]))
            values['raster_ms'].append(float(np.median([float(x['raster_ms']) for x in samples])))
        data[method] = {key: np.asarray(value) for key, value in values.items()}
        data[method]['psnr_smooth'] = gaussian_filter1d(data[method]['psnr'], 35, mode='reflect')
        attempts = json.loads((formal/'BLOCK_ATTEMPTS.json').read_text())
        accepted = [a for a in attempts if a['accepted']]
        assert all(a['external_gpu_samples'] == a['external_cpu_samples'] == 0 for a in accepted)
        assert len([a for a in accepted if a['phase'] == 'search']) == 20
        assert len([a for a in accepted if a['phase'] == 'validation']) == 30
        sources.extend([formal/'QUALITY_COUNTS.csv', formal/'TIMINGS.csv', formal/'BUDGETS_FROZEN.csv',
                        formal/'VALIDATION.json', formal/'BLOCK_ATTEMPTS.json', folder/'logs/formal_job.json'])
    out = ROOT/'figures'
    out.mkdir(exist_ok=True)
    summary = []
    with (ROOT/'PER_FRAME.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, ['method', 'frame', 'K', 'contributing_K', 'psnr', 'psnr_smooth', 'raster_ms'])
        writer.writeheader()
        for method, label in zip(METHODS, LABELS):
            v = data[method]
            assert all(np.isfinite(x).all() for x in v.values())
            for i in range(2000):
                writer.writerow(dict(method=method, frame=i, **{k: float(a[i]) for k, a in v.items()}))
            time = v['raster_ms']
            summary.append(dict(method=method, label=label, K_mean=float(v['K'].mean()),
                contributing_K_mean=float(v['contributing_K'].mean()), psnr_mean=float(v['psnr'].mean()),
                raster_ms_mean=float(time.mean()), raster_ms_p95=float(np.percentile(time, 95)),
                raster_ms_cv_percent=float(100*time.std()/time.mean())))
    with (ROOT/'SUMMARY.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(summary[0])); writer.writeheader(); writer.writerows(summary)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'path'})
    fig = plt.figure(figsize=(14, 5))
    grid = fig.add_gridspec(1, 3, left=.06, right=.985, bottom=.23, top=.73, wspace=.34)
    count = fig.add_subplot(grid[0, 0]); quality = fig.add_subplot(grid[0, 1], sharex=count)
    sub = grid[0, 2].subgridspec(2, 1, height_ratios=[1, 2.1], hspace=.1)
    upper = fig.add_subplot(sub[0], sharex=count); lower = fig.add_subplot(sub[1], sharex=count)
    upper.set_ylim(3.3, 5.45)
    bottom = np.concatenate([data[m]['raster_ms'] for m in METHODS[1:]])
    lower.set_ylim(max(0, float(bottom.min())-.05), float(bottom.max())+.12)
    assert bottom.max() < 3.3
    upper.spines['bottom'].set_visible(False); upper.tick_params(axis='x', bottom=False, labelbottom=False)
    for axis, y in [(upper, 0), (lower, 1)]:
        axis.plot([-.014, .014], [y-.025, y+.025], transform=axis.transAxes, clip_on=False, color='#252525', lw=.9)
    for method, color, style in zip(METHODS, COLORS, STYLES):
        v = data[method]; x = np.arange(2000)
        for axis, array in [(count, v['contributing_K']/1000), (quality, v['psnr_smooth']),
                            (upper if method == 'cartgs_full' else lower, v['raster_ms'])]:
            line, = axis.plot(x, array, color=color, ls=style, lw=1.2 if axis in [count, quality] else .65)
            assert np.array_equal(line.get_ydata(), array)
    count.set(title='(a) Contributing Gaussians', ylabel='Gaussians (k)', ylim=(0, None))
    quality.set(title='(b) Rendering quality', ylabel='PSNR (dB)')
    upper.set_title('(c) Rendering latency')
    fig.text(.697, .48, 'Rendering time (ms)', rotation=90, ha='center', va='center')
    for ax in [count, quality, upper, lower]:
        ax.set_xlim(0, 1999); ax.set_xticks([0, 500, 1000, 1500, 1999]); ax.grid(alpha=.13)
    for ax in [count, quality, lower]: ax.set_xlabel('RGB frame index')
    handles = [Line2D([], [], color=c, ls=s, lw=1.7, label=l) for c, s, l in zip(COLORS, STYLES, LABELS)]
    fig.legend(handles=handles, ncol=3, loc='upper center', bbox_to_anchor=(.51, .99), frameon=False, fontsize=9.3)
    fig.text(.52, .09, 'PSNR smoothed (σ = 35 frames); latency unsmoothed. Pure rendering only.', ha='center', fontsize=8.5)
    fig.text(.52, .05, 'Dynamic budgets are offline per-view searches; median of 3 single-image repeats per view.', ha='center', fontsize=8.5)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(out/f'OFFICE2_DYNAMIC_RATIO_FIVE_METHODS.{extension}', dpi=450, bbox_inches='tight')
    plt.close(fig)
    manifest = dict(frames=2000, methods=METHODS, labels=LABELS, real_measurements=True,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        same_round=False, timing='per-frame median of three; separate experiment runs',
        dynamic_budget='offline per-view search, not realtime controller', psnr_sigma=35)
    (ROOT/'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

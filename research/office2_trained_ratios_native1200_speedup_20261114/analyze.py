"""核验完整逐帧配对数据，绘制未平滑的训练ratio相对Full加速比。"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RATIOS = [.01, .05, .1, .3, .5, .7]
COLORS = ['#0072B2', '#009E73', '#56B4E9', '#E69F00', '#D55E00', '#CC79A7']
NAME = 'OFFICE2_NATIVE1200_TRAINED_RATIOS_SPEEDUP_OVER_OURS_FULL'


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT/'formal/TIMINGS.csv'
    validation = json.loads((ROOT/'formal/VALIDATION.json').read_text())
    protocol = json.loads((ROOT/'formal/PROTOCOL.json').read_text())
    assert validation['passed'] and validation['frames'] == 2000 and validation['repeats'] == 3
    assert protocol['training_ratios'] == RATIOS and not protocol['smoothing']
    assert (protocol['width'], protocol['height']) == (1200, 680)
    assert protocol['native_resolution'] and validation['native_resolution_verified'] and not protocol['resizing']
    attempts = json.loads((ROOT/'formal/BLOCK_ATTEMPTS.json').read_text())
    accepted = [a for a in attempts if a['accepted']]
    assert len(accepted) == 30
    assert all(a['external_gpu_samples'] == a['external_cpu_samples'] == 0 for a in accepted)
    groups = {}
    for row in read(source):
        groups.setdefault((row['method'], int(row['frame'])), []).append(row)
    assert len(groups) == 14000 and sum(map(len, groups.values())) == 42000

    def samples(method, frame):
        rows = groups[method, frame]
        assert sorted(int(row['repeat']) for row in rows) == [0, 1, 2]
        assert len({(row['M'], row['K'], row['ids_sha256']) for row in rows}) == 1
        times = np.asarray([float(row['raster_ms']) for row in rows])
        assert np.isfinite(times).all() and (times > 0).all()
        return rows, times

    full = np.asarray([np.median(samples('ours_full', f)[1]) for f in range(2000)])
    assert all(int(row['K']) == 43197 for f in range(2000) for row in samples('ours_full', f)[0])
    reports, figures = ROOT/'reports', ROOT/'figures'
    reports.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'path'})
    fig, ax = plt.subplots(figsize=(12, 5.0))
    detail, panels = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
    plotted, summary = [], []
    for ratio, color, sub in zip(RATIOS, COLORS, panels.flat):
        method = f'ratio_{round(ratio*100):02d}'
        latency, k = [], []
        for frame in range(2000):
            rows, times = samples(method, frame)
            item = rows[0]
            assert float(item['ratio']) == ratio
            assert int(item['K']) == min(int(np.floor(ratio*int(item['M']))), int(item['eligible']))
            ms = float(np.median(times))
            latency.append(ms)
            k.append(int(item['K']))
            plotted.append(dict(frame=frame, ratio=ratio, M=int(item['M']), K=int(item['K']),
                full_pure_render_ms=float(full[frame]), ratio_pure_render_ms=ms,
                speedup=float(full[frame]/ms)))
        latency, k = np.asarray(latency), np.asarray(k)
        speedup = full/latency
        label = f'Ratio {ratio:g}'
        line, = ax.plot(np.arange(2000), speedup, color=color, label=label, lw=.8, alpha=.85)
        assert np.array_equal(line.get_ydata(), speedup)
        sub.plot(np.arange(2000), speedup, color=color, lw=.8)
        sub.axhline(1, color='#555555', ls='--', lw=.7)
        sub.set_title(label, fontsize=11)
        sub.grid(alpha=.15)
        summary.append(dict(ratio=ratio, views=2000, repeats=3,
            full_mean_ms=float(full.mean()), ratio_mean_ms=float(latency.mean()),
            mean_per_view_speedup=float(speedup.mean()), median_per_view_speedup=float(np.median(speedup)),
            p05_per_view_speedup=float(np.percentile(speedup, 5)), p95_per_view_speedup=float(np.percentile(speedup, 95)),
            min_speedup=float(speedup.min()), max_speedup=float(speedup.max()),
            ratio_of_mean_latencies=float(full.mean()/latency.mean()), slower_views=int((speedup < 1).sum()),
            mean_K=float(k.mean()), min_K=int(k.min()), max_K=int(k.max())))
    ax.axhline(1, color='#555555', ls='--', lw=1, label='Ours Full (1×)')
    ax.set(xlabel='RGB frame index', ylabel='Rendering speedup (×)',
        xlim=(0, 1999), ylim=(0, 5.5))
    fig.suptitle('Replica office2 RGB-D | 1200 × 680 (native) | Mode 40', y=.98, fontsize=13)
    ax.set_xticks([0, 500, 1000, 1500, 1999])
    ax.grid(alpha=.15)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.14), ncol=7, fontsize=9)
    fig.subplots_adjust(left=.085, right=.985, top=.81, bottom=.20)
    note = 'Full latency / ratio latency per view; median of 3 repeats; no temporal smoothing.\n'
    note += 'Pure rendering only; excludes preparation, Selector and Compensator. Ratios use visible nominal Anchor candidates.'
    fig.text(.53, .055, note, ha='center', va='center', fontsize=9)
    detail.supxlabel('RGB frame index')
    detail.supylabel('Pure rendering speedup over Ours Full (×)')
    detail.suptitle('Replica office2 RGB-D | 1200 × 680 (native) | Mode 40 | No smoothing', fontsize=13)
    detail.tight_layout(rect=(.025, .025, 1, .96))
    for canvas, suffix in [(fig, ''), (detail, '_PANELS')]:
        for extension in ['png', 'svg', 'pdf']:
            canvas.savefig(figures/f'{NAME}{suffix}.{extension}', dpi=450, bbox_inches='tight')
        plt.close(canvas)
    write(reports/'PER_VIEW_SPEEDUP.csv', plotted)
    write(reports/'SUMMARY.csv', summary)
    result = dict(passed=True, frames=2000, ratios=RATIOS, rows=12000, raw_timing_rows=42000,
        width=1200, height=680, native_resolution=True,
        smoothing=False, same_round_Full=True, pure_render_only=True,
        accepted_blocks=len(accepted), rejected_blocks=len(attempts)-len(accepted),
        sources={str(path): sha(path) for path in [source, ROOT/'formal/VALIDATION.json',
            ROOT/'formal/PROTOCOL.json', ROOT/'formal/BLOCK_ATTEMPTS.json', Path(__file__)]},
        plotted_csv_sha256=sha(reports/'PER_VIEW_SPEEDUP.csv'), summary=summary)
    (reports/'ANALYSIS_VALIDATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# 六档训练 ratio 相对 Ours Full 的逐视角纯渲染加速比', '',
        'Replica office2 RGB-D，2000帧，1200×680（原始分辨率），RTX 3090，统一Mode40；未平滑。', '',
        '每视角加速比 = 同轮Full三次计时中位数 / 同轮ratio三次计时中位数。只计单次纯渲染，不含准备、Selector、Compensator。', '',
        '使用已有蒸馏Student、Tensor D及新准备。Full为未选择、未补偿的43197个Master高斯；ratio按当前视角名义Anchor候选池M计算严格预算，非全地图ratio导出。', '',
        '|训练ratio|平均逐视角加速比|5%–95%分位|平均纯渲染ms|平均实际K|慢于Full的视角数|',
        '|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"|{row['ratio']:g}|{row['mean_per_view_speedup']:.3f}×|{row['p05_per_view_speedup']:.3f}–{row['p95_per_view_speedup']:.3f}×|{row['ratio_mean_ms']:.6f}|{row['mean_K']:.1f}|{row['slower_views']}|")
    lines += ['', f'Full平均纯渲染：{full.mean():.6f} ms。', '',
        f"验收{len(accepted)}个200帧整块；弃用{len(attempts)-len(accepted)}个受干扰整块。所有接受窗口无外部活跃GPU/高CPU进程。模型、二进制及保护文件研究前后哈希一致，未修改训练、生产配置或模型，未保存渲染图片。", '',
        '主图为六条原始逐帧曲线；PANELS为相同数据的六个子图。逐帧数据：reports/PER_VIEW_SPEEDUP.csv；原始三次计时：formal/TIMINGS.csv。', '',
        '平均逐视角比值与平均耗时之比不是同一统计量，SUMMARY.csv同时保留两者。减少输入高斯的纯渲染收益不能直接作为完整推理提速。']
    (ROOT/'TRAINED_RATIO_SPEEDUP_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, full_mean_ms=float(full.mean()), summary=summary), ensure_ascii=False))


if __name__ == '__main__':
    main()

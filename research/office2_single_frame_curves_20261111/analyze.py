"""逐帧点数、PSNR、单次渲染速度：绑定真实输入并校验每条曲线。"""
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
METHODS = ['cartgs_full', 'ratio_50', 'fixed_2000', 'fixed_4000']
LABELS = ['CaRtGS', r'Ours ($r=0.5$)', 'Ours (2k points)', 'Ours (4k budget*)']
COLORS = ['#252525', '#12A7C8', '#0077BB', '#D55E00']
STYLES = ['-', '-.', '-', '-']
SIGMA = 35


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure(data, field, output):
    plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'font.size': 9.5,
        'axes.titlesize': 10.5, 'axes.labelsize': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .8, 'svg.fonttype': 'path'})
    baseline = data['cartgs_full'][field]
    ours = np.concatenate([data[m][field] for m in METHODS[1:]])
    low_min, low_max = ours.min(), ours.max()
    high_min, high_max = baseline.min(), baseline.max()
    broken = bool(high_min > low_max*1.15 and high_min-low_max > .15)
    fig = plt.figure(figsize=(12.8, 4.3))
    grid = fig.add_gridspec(1, 3, left=.058, right=.985, top=.76, bottom=.22, wspace=.31)
    counts = fig.add_subplot(grid[0, 0])
    quality = fig.add_subplot(grid[0, 1], sharex=counts)
    if broken:
        subgrid = grid[0, 2].subgridspec(2, 1, height_ratios=[1, 2.1], hspace=.08)
        top = fig.add_subplot(subgrid[0], sharex=counts)
        latency = fig.add_subplot(subgrid[1], sharex=counts)
        time_axes = [top, latency]
        pad_low = max(.02, (low_max-low_min)*.10)
        pad_high = max(.03, (high_max-high_min)*.16)
        latency.set_ylim(max(0, low_min-pad_low), low_max+pad_low)
        top.set_ylim(high_min-pad_high, high_max+pad_high)
        top.spines['bottom'].set_visible(False)
        latency.spines['top'].set_visible(False)
        top.tick_params(axis='x', bottom=False, labelbottom=False)
        for ax, yy in [(top, 0), (latency, 1)]:
            ax.plot([-.012, .012], [yy-.018, yy+.018], color='#252525',
                transform=ax.transAxes, clip_on=False, lw=.8)
        top.set_title('(c) 单帧纯渲染耗时' if field == 'raster_ms' else '(c) 准备＋网络＋渲染耗时', pad=10)
        fig.text(.696, .49, '纯渲染时间 (ms)' if field == 'raster_ms' else '完整流程时间 (ms)',
            rotation=90, ha='center', va='center', fontsize=10)
    else:
        latency = fig.add_subplot(grid[0, 2], sharex=counts)
        time_axes = [latency]
        latency.set_title('(c) 单帧纯渲染耗时' if field == 'raster_ms' else '(c) 准备＋网络＋渲染耗时')
        latency.set_ylabel('纯渲染时间 (ms)' if field == 'raster_ms' else '完整流程时间 (ms)')
    checked = 0
    x = data[METHODS[0]]['frame']
    for method, label, color, style in zip(METHODS, LABELS, COLORS, STYLES):
        d = data[method]
        for ax, values in [(counts, d['contributing_K']/1000), (quality, d['psnr_smooth'])]:
            line, = ax.plot(x, values, color=color, ls=style, lw=1.25)
            assert np.array_equal(line.get_xdata(), x) and np.array_equal(line.get_ydata(), values)
            checked += 1
        for ax in time_axes:
            line, = ax.plot(x, d[field], color=color, ls=style, lw=.65)
            assert np.array_equal(line.get_ydata(), d[field])
            checked += 1
            mean, std = d[field].mean(), d[field].std(ddof=0)
            for bound in [mean-std, mean+std]:
                ax.axhline(bound, color=color, lw=.65, ls=(0, (4, 4)), alpha=.55)
    counts.set(title='(a) 实际参与 Alpha 合成的高斯点数', ylabel='高斯点数（千）', ylim=(0, None))
    quality.set(title='(b) 每视角渲染质量', ylabel='PSNR (dB)')
    for ax in [counts, quality, *time_axes]:
        ax.set_xlim(x.min(), x.max())
        ax.grid(alpha=.16, lw=.5)
        ax.set_xticks(np.linspace(0, x.max()+1, 5, dtype=int)[:-1].tolist()+[int(x.max())])
    for ax in [counts, quality, latency]:
        ax.set_xlabel('RGB 帧编号')
    handles = [Line2D([], [], color=c, ls=s, lw=1.7, label=l) for c, s, l in zip(COLORS, STYLES, LABELS)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .94), ncol=4,
        frameon=False, fontsize=10, handlelength=2.7, columnspacing=2)
    fig.suptitle('Replica office2 RGB-D｜2064×2208｜每帧单次渲染', y=1.005, fontsize=11.5)
    fig.text(.52, .075,
        '点数：实际 Alpha 贡献点；PSNR：高斯平滑（σ＝35帧）；耗时：每帧三次重复的中位数，不平滑。',
        ha='center', fontsize=7.7)
    fig.text(.52, .035,
        '水平虚线：均值 ± 1个标准差。'+('第三栏不含准备和网络推理。' if field == 'raster_ms' else '第三栏包含当前视角准备及 Selector、Compensator 推理。'),
        ha='center', fontsize=7.7)
    fig.text(.52, -.005, '*4k：79帧有效候选不足，实际提交 K＝2088–4000；全部帧均保留。',
        ha='center', fontsize=7.7)
    for extension in ['png', 'svg']:
        fig.savefig(output/f'OFFICE2_SINGLE_FRAME_{field.upper()}.{extension}', dpi=450, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return dict(field=field, broken_axis=broken, checked_curves=checked)


def main():
    reports, output = ROOT/'reports', ROOT/'figures'
    output.mkdir(exist_ok=True)
    validation = json.loads((reports/'VALIDATION.json').read_text())
    job = json.loads((ROOT/'logs/formal_job.json').read_text())
    assert validation['passed'] and job['measurement_valid'], job
    quality, timings = read(reports/'QUALITY_COUNTS.csv'), read(reports/'TIMINGS.csv')
    n = validation['frames']
    assert len(quality) == n*4 and len(timings) == n*4*3
    timed = {}
    for row in timings:
        key = row['method'], int(row['frame'])
        timed.setdefault(key, []).append(row)
    all_rows, summary, data = [], [], {}
    for method in METHODS:
        rows = sorted([r for r in quality if r['method'] == method], key=lambda r: int(r['frame']))
        assert [int(r['frame']) for r in rows] == list(range(n))
        values = {key: [] for key in ['frame', 'K', 'M', 'projected_K', 'contributing_K', 'psnr', 'prepare_ms', 'raster_ms', 'total_ms']}
        for row in rows:
            samples = timed[method, int(row['frame'])]
            assert sorted(int(r['repeat']) for r in samples) == [0, 1, 2]
            assert all(int(t['K']) == int(row['K']) and int(t['M']) == int(row['M']) for t in samples)
            assert 0 <= int(row['contributing_K']) <= int(row['projected_K']) <= int(row['K'])
            for key in values:
                values[key].append(float(row[key]) if key in row else np.median([float(t[key]) for t in samples]))
        values = {key: np.asarray(val) for key, val in values.items()}
        assert all(np.isfinite(a).all() for a in values.values())
        values['psnr_smooth'] = gaussian_filter1d(values['psnr'], sigma=SIGMA, mode='reflect')
        data[method] = values
        for i, row in enumerate(rows):
            all_rows.append(dict(method=method, **{key: float(val[i]) for key, val in values.items()}))
        stats = dict(method=method, frames=n,
            shortfall_frames=sum(int(r['K']) < int(r['requested_K']) for r in rows),
            K_min=int(values['K'].min()), K_max=int(values['K'].max()))
        for key in ['K', 'contributing_K', 'psnr', 'prepare_ms', 'raster_ms', 'total_ms']:
            a = values[key]
            stats.update({key+'_mean': float(a.mean()), key+'_median': float(np.median(a)),
                key+'_p95': float(np.percentile(a, 95)), key+'_std': float(a.std(ddof=0))})
        summary.append(stats)
    write(reports/'PLOT_DATA.csv', all_rows)
    write(reports/'SUMMARY.csv', summary)
    save(reports/'SUMMARY.json', summary)
    rounds = []
    for method in METHODS:
        for repeat in range(3):
            selected = [r for r in timings if r['method'] == method and int(r['repeat']) == repeat]
            assert len(selected) == n
            rounds.append(dict(method=method, repeat=repeat, frames=n,
                **{key+'_mean': float(np.mean([float(r[key]) for r in selected]))
                   for key in ['prepare_ms', 'raster_ms', 'total_ms']}))
    write(reports/'TIMING_ROUNDS.csv', rounds)
    figures = [figure(data, field, output) for field in ['raster_ms', 'total_ms']]
    sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [reports/'QUALITY_COUNTS.csv', reports/'TIMINGS.csv', reports/'PLOT_DATA.csv', reports/'VALIDATION.json', ROOT/'logs/formal_job.json']}
    save(output/'DATA_MANIFEST.json', dict(real_data=True, sources=sources, figures=figures,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), dpi=450,
        psnr_smoothing=dict(method='Gaussian', sigma_frames=SIGMA, mode='reflect'), latency_smoothing=False))
    save(reports/'ANALYSIS_VALIDATION.json', dict(passed=True, frames=n, methods=4,
        quality_rows=len(quality), timing_rows=len(timings), repeats_per_frame=3,
        paired_K_M_equal=True, all_finite=True, all_frames_retained=True,
        timing_samples_excluded=0, checked_curves=sum(f['checked_curves'] for f in figures),
        psnr_smoothing_sigma_frames=SIGMA, latency_smoothing=False))
    (output/'data-manifest.md').write_text('# 图表数据清单\n\n数据均为本轮office2完整轨迹实测。'
        '点数和PSNR来自QUALITY_COUNTS.csv；每帧时间取TIMINGS.csv三次重复的中位数。'
        '主图第三栏为纯渲染，附图为含准备和网络的完整流程。PSNR仅显示时作sigma=35帧的高斯平滑，'
        '点数和时间不平滑，不删除帧或异常时间样本。所有输入SHA256见DATA_MANIFEST.json。\n', encoding='utf-8')
    lines = ['# office2逐帧单次渲染对比', '',
        f'2064×2208，每次请求只渲染一张图像；完整轨迹{n}帧，四种方法各三次计时。硬件RTX3090。', '',
        '| 方法 | 提交K均值 | Alpha贡献点均值 | PSNR均值(dB) | 准备均值(ms) | 纯渲染均值(ms) | 完整均值(ms) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s, label in zip(summary, ['CaRtGS', 'Ours 50%', 'Ours 2k', 'Ours 4k']):
        lines.append('|'+label+'|'+'|'.join(f'{s[key+"_mean"]:.4f}' for key in ['K', 'contributing_K', 'psnr', 'prepare_ms', 'raster_ms', 'total_ms'])+'|')
    lines += ['', '## 图表与口径', '',
        '- 主图三栏：实际Alpha贡献点数、PSNR、纯渲染时间；附图第三栏替换为准备＋网络＋渲染完整墙钟。',
        '- 原始点数/PSNR逐帧保存；PSNR曲线采用sigma=35帧的高斯平滑。耗时为每帧三次计时中位数，不作平滑。',
        '- 50%按当前可见名义Anchor候选池计算预算；2k/4k为提交预算，不代表所有点实际贡献像素。',
        '- 2k全部2000帧达到预算；4k有79帧有效候选不足，实际提交K最少2088、均值3951.163；未补点、未剔除。',
        '- CaRtGS地图84203点，我们43197点；两侧分别使用对应SLAM位姿并按RGB时间戳配对。',
        '- 本方法采用蒸馏Compensator＋Tensor Core＋新准备D＋新G渲染器。收益包含地图、网络和渲染器差异，不能全归Tensor Core。',
        '- PSNR参考图为原RGB按INTER_LINEAR拉伸至2064×2208；不是高分辨率原生GT或Quest3真机。',
        '- 贡献计数使用独立诊断Rasterizer，计数不进入测速。首/中/末帧四方法RGB和radii与实际测速路径逐位一致。',
        '- 纯渲染使用CUDA Event，已排除相机、选点和网络；完整流程独立实测，中位数不能机械相加。',
        '- 无训练、无生产配置修改，不保存渲染图片；源码、配置、模型和二进制保护哈希核验见VALIDATION.json。', '',
        '## 验证', '', f'- 质量/计数记录：{len(quality)}条；计时记录：{len(timings)}条；逐位验收：{validation["parity_checks"]}组。',
        f'- 外部活跃GPU采样：{job["foreign_active_samples"]}；外部CPU活跃采样：{job["foreign_cpu_active_samples"]}。',
        f'- {validation["protected_files"]}个受保护文件前后哈希一致，变化列表为空。',
        '- 首次smoke在加载时因inference_mode不提供Tensor版本号而退出；改为既有no_grad加载方式后验收通过，未修改模型或算法。', '']
    (ROOT/'OFFICE2_SINGLE_FRAME_CURVES_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(passed=True, frames=n, summary=summary, figures=figures), ensure_ascii=False))


if __name__ == '__main__':
    main()

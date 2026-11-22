"""六方法同口径绘图：新增Ours完整Master，保留0.4ms冻结预算。"""
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
METHODS = ['cartgs_full', 'ours_full', 'ratio_50', 'fixed_2000', 'fixed_4000', 'dynamic_04']
LABELS = ['CaRtGS', 'Ours Full', 'Ours 50%', 'Ours 2k', 'Ours 4k*', 'Ours 动态0.4 ms']
COLORS = ['#252525', '#009988', '#12A7C8', '#0077BB', '#D55E00', '#8E44AD']
STYLES = ['-', '--', '-.', '-', '-', '-']


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
    source, reports, output = ROOT/'formal', ROOT/'reports', ROOT/'figures'
    reports.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    validation = json.loads((source/'VALIDATION.json').read_text())
    job = json.loads((ROOT/'logs/formal_job.json').read_text())
    assert validation['passed'] and job['returncode'] == 0
    assert validation['frames'] == 2000 and not validation['changed']
    frozen = ROOT.parent/'formal/BUDGETS_FROZEN.csv'
    assert sha(frozen) == validation['frozen_budget_sha256']
    quality, timings = read(source/'QUALITY_COUNTS.csv'), read(source/'TIMINGS.csv')
    assert len(quality) == 12000 and len(timings) == 36000
    attempts = json.loads((source/'BLOCK_ATTEMPTS.json').read_text())
    accepted = [row for row in attempts if row['accepted']]
    assert all(row['external_gpu_samples'] == row['external_cpu_samples'] == 0 for row in accepted)
    assert sorted((row['repeat'], row['block']) for row in accepted) == [(r, b) for r in range(3) for b in range(10)]
    samples = {}
    for row in timings:
        samples.setdefault((row['method'], int(row['frame'])), []).append(row)
    data, plotted, summary = {}, [], []
    for method in METHODS:
        rows = sorted((row for row in quality if row['method'] == method), key=lambda row: int(row['frame']))
        assert [int(row['frame']) for row in rows] == list(range(2000))
        values = {key: [] for key in ['frame', 'K', 'M', 'ratio', 'projected_K', 'contributing_K', 'psnr', 'raster_ms', 'raster_wall_ms']}
        for row in rows:
            measured = samples[method, int(row['frame'])]
            assert sorted(int(item['repeat']) for item in measured) == [0, 1, 2]
            assert all(int(item['K']) == int(row['K']) and int(item['M']) == int(row['M']) for item in measured)
            assert 0 <= int(row['contributing_K']) <= int(row['projected_K']) <= int(row['K'])
            for key in values:
                values[key].append(float(row[key]) if key in row else float(np.median([float(item[key]) for item in measured])))
        values = {key: np.asarray(array) for key, array in values.items()}
        assert all(np.isfinite(array).all() for array in values.values())
        if method == 'ours_full':
            assert (values['K'] == 43197).all() and (values['ratio'] == 1).all()
        values['psnr_smooth'] = gaussian_filter1d(values['psnr'], sigma=35, mode='reflect')
        data[method] = values
        plotted.extend(dict(method=method, **{key: float(array[i]) for key, array in values.items()}) for i in range(2000))
        current = dict(method=method, frames=2000)
        for key in ['K', 'projected_K', 'contributing_K', 'psnr', 'raster_ms', 'raster_wall_ms']:
            array = values[key]
            current.update({key+'_mean': float(array.mean()), key+'_p95': float(np.percentile(array, 95)),
                key+'_min': float(array.min()), key+'_max': float(array.max())})
        summary.append(current)
    write(reports/'PLOT_DATA.csv', plotted)
    write(reports/'SUMMARY.csv', summary)
    plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'font.size': 9.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'path'})
    fig = plt.figure(figsize=(13.6, 4.9))
    grid = fig.add_gridspec(1, 3, left=.06, right=.986, top=.75, bottom=.23, wspace=.33)
    count = fig.add_subplot(grid[0, 0])
    quality_ax = fig.add_subplot(grid[0, 1], sharex=count)
    reference = data['cartgs_full']['raster_ms']
    ours = np.concatenate([data[method]['raster_ms'] for method in METHODS[1:]])
    broken_axis = bool(ours.max()+.12 < reference.min()-.12)
    if broken_axis:
        inner = grid[0, 2].subgridspec(2, 1, height_ratios=[1, 2.1], hspace=.08)
        top = fig.add_subplot(inner[0], sharex=count)
        latency = fig.add_subplot(inner[1], sharex=count)
        top.set_ylim(reference.min()-.1, reference.max()+.1)
        latency.set_ylim(max(0, ours.min()-.04), ours.max()+.06)
        top.spines['bottom'].set_visible(False)
        top.tick_params(axis='x', bottom=False, labelbottom=False)
        for axis, y in [(top, 0), (latency, 1)]:
            axis.plot([-.012, .012], [y-.018, y+.018], transform=axis.transAxes, clip_on=False, color='#252525', lw=.8)
        time_axes = [top, latency]
    else:
        latency = fig.add_subplot(grid[0, 2], sharex=count)
        latency.set_ylim(0, max(reference.max(), ours.max())*1.05)
        top, time_axes = latency, [latency]
    x = np.arange(2000)
    for method, color, style in zip(METHODS, COLORS, STYLES):
        values = data[method]
        width = 1.4 if method in ['ours_full', 'dynamic_04'] else 1.0
        for axis, array in [(count, values['contributing_K']/1000), (quality_ax, values['psnr_smooth'])]+[(axis, values['raster_ms']) for axis in time_axes]:
            line, = axis.plot(x, array, color=color, ls=style, lw=width if axis in [count, quality_ax] else width*.65)
            assert np.array_equal(line.get_ydata(), array)
    latency.axhspan(.38, .42, color=COLORS[-1], alpha=.08)
    latency.axhline(.4, color='#626262', lw=.8, ls=(0, (4, 3)))
    count.set(title='(a) 实际 Alpha 贡献高斯点数', ylabel='高斯点数（千）', ylim=(0, None))
    quality_ax.set(title='(b) 每视角渲染质量', ylabel='PSNR (dB)')
    top.set_title('(c) 单图纯渲染耗时')
    fig.text(.697, .49, '纯渲染时间 (ms)', rotation=90, ha='center', va='center')
    for axis in [count, quality_ax]+time_axes:
        axis.set_xlim(0, 1999)
        axis.set_xticks([0, 500, 1000, 1500, 1999])
        axis.grid(alpha=.16)
    for axis in [count, quality_ax, latency]:
        axis.set_xlabel('RGB 帧编号')
    handles = [Line2D([], [], color=color, ls=style, lw=1.6, label=label) for color, style, label in zip(COLORS, STYLES, LABELS)]
    fig.legend(handles=handles, ncol=3, loc='upper center', bbox_to_anchor=(.51, .96), frameon=False, fontsize=9.5, handlelength=2.6, columnspacing=3.)
    fig.suptitle('Replica office2 RGB-D｜2064×2208｜Mode 40｜加入 Ours Full', y=1.005, fontsize=12)
    fig.text(.52, .115, 'Ours Full：完整未补偿Master，共43,197点；图(a)显示每帧实际参与Alpha合成的点数。', ha='center', fontsize=8)
    fig.text(.52, .075, '六种方法耗时同轮复测，逐帧三次单图中位数；仅纯渲染，不含准备、网络和预算搜索。', ha='center', fontsize=8)
    fig.text(.52, .035, '仅PSNR采用σ＝35帧平滑。动态0.4 ms沿用冻结离线预算；*4k有79帧候选不足。', ha='center', fontsize=8)
    for extension in ['png', 'svg']:
        fig.savefig(output/f'OFFICE2_DYNAMIC_04_WITH_OURS_FULL.{extension}', dpi=450, bbox_inches='tight')
    plt.close(fig)
    lines = ['# office2三联图补充Ours Full', '',
        'Ours Full使用同一训练模型的完整未补偿Master（43,197点），不调用Selector、Compensator或名义Anchor视锥筛选。'
        '它不是ratio=1的局部LoD，也不是CaRtGS原版地图。', '',
        'Replica office2 RGB-D完整2000帧，RTX3090，2064×2208，Mode40，单次一张图。'
        '新增Full点数/质量实测；原五种方法点数和质量沿用已验收记录，并检查代表帧逐值一致。'
        '六种方法耗时全部同轮重新测量，三轮顺序为正序、倒序和打乱，逐帧取三次中位数。', '',
        '| 方法 | 提交K均值 | 实际Alpha贡献点均值 | PSNR均值/dB | 纯渲染均值/ms | 逐帧耗时p95/ms |',
        '|---|---:|---:|---:|---:|---:|']
    for row, label in zip(summary, LABELS):
        lines.append(f'|{label}|{row["K_mean"]:.2f}|{row["contributing_K_mean"]:.2f}|{row["psnr_mean"]:.6f}|{row["raster_ms_mean"]:.6f}|{row["raster_ms_p95"]:.6f}|')
    lines += ['', '## 验证及范围', '',
        f'- 六方法各2000帧，共12000条点数/质量、36000条同轮计时；验收{len(accepted)}块，重测{len(attempts)-len(accepted)}块。',
        '- 仅按外部CPU/GPU活动判断整块重测，不按耗时筛除；所有接受块的扩展窗口内外部活动为零。',
        '- Full每帧诊断RGB与正常Mode40输出逐位一致，贡献点数由实际Alpha合成标记统计，计数不进入耗时。',
        '- 正式首轮逐帧核对原LoD方案选点SHA，动态K保持原冻结预算，未重新搜索。',
        '- 图上点数是实际Alpha贡献点，不是整图提交K；Full提交43,197点，但实际贡献随视角变化。',
        '- PSNR仅在图中以σ=35帧高斯平滑，表中PSNR为原始逐帧算术均值；点数和时间均不平滑。',
        '- GT沿用INTER_LINEAR放大的原始RGB；Mode40为既有近似渲染器，不代表原生高分辨率真值或Quest3硬件。',
        '- 动态预算属于离线逐帧可达性测试，不是实时控制器，也不是单次调用时延硬保证。',
        f'- {validation["protected_files"]}个受保护文件哈希不变；无训练、编译、生产配置修改、commit/push或渲染图片保存。', '',
        '原图和原始交付保持不变，本次补充独立保存在with_ours_full目录。', '']
    (ROOT/'OURS_FULL_ADDITION_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    sources = [source/'QUALITY_COUNTS.csv', source/'TIMINGS.csv', source/'VALIDATION.json', source/'BLOCK_ATTEMPTS.json',
        ROOT/'logs/formal_job.json', ROOT/'analyze.py', ROOT/'benchmark.py', frozen, ROOT.parent/'formal/QUALITY_COUNTS.csv']
    manifest = dict(real_data=True, sources={str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha(path) for path in sources},
        dpi=450, png_svg=True, frames=2000, methods=METHODS, psnr_sigma_frames=35, timing_smoothing=False, count_smoothing=False)
    (output/'DATA_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/'data-manifest.md').write_text('# 图表数据清单\n\n六方法真实实测；输入SHA见DATA_MANIFEST.json。'
        '点数为实际Alpha贡献点；质量和点数来自formal/QUALITY_COUNTS.csv，时间来自formal/TIMINGS.csv。'
        '脚本为analyze.py；输出PNG/SVG，450 DPI。仅PSNR曲线σ=35帧平滑。\n', encoding='utf-8')
    result = dict(passed=True, frames=2000, quality_rows=12000, timing_rows=36000, six_methods=True,
        full_master_K=43197, no_production_change=True, frozen_budget_sha256=sha(frozen), broken_time_axis=broken_axis)
    (reports/'ANALYSIS_VALIDATION.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(validation=result, summary=summary), ensure_ascii=False))


if __name__ == '__main__':
    main()

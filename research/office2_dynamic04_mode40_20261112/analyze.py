"""逐帧动态K与独立时延验证，Mode40五方法三联图。"""
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
METHODS = ['cartgs_full', 'ratio_50', 'fixed_2000', 'fixed_4000', 'dynamic_04']
LABELS = ['CaRtGS', 'Ours 50%', 'Ours 2k', 'Ours 4k*', 'Ours 动态0.4 ms']
COLORS = ['#252525', '#12A7C8', '#0077BB', '#D55E00', '#8E44AD']
STYLES = ['-', '-.', '-', '-', '-']


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    source, reports, output = ROOT/'formal', ROOT/'reports', ROOT/'figures'
    reports.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    validation = json.loads((source/'VALIDATION.json').read_text())
    job = json.loads((ROOT/'logs/formal_job.json').read_text())
    assert validation['passed'] and job['returncode'] == 0 and validation['independent_validation']
    frozen = read(source/'BUDGETS_FROZEN.csv')
    assert hashlib.sha256((source/'BUDGETS_FROZEN.csv').read_bytes()).hexdigest() == validation['frozen_budget_sha256']
    assert [int(r['frame']) for r in frozen] == list(range(2000))
    quality, timings = read(source/'QUALITY_COUNTS.csv'), read(source/'TIMINGS.csv')
    assert len(quality) == 10000 and len(timings) == 30000
    attempts = json.loads((source/'BLOCK_ATTEMPTS.json').read_text())
    accepted = [a for a in attempts if a['accepted']]
    assert all(a['external_gpu_samples'] == a['external_cpu_samples'] == 0 for a in accepted)
    assert len([a for a in accepted if a['phase'] == 'search']) == 20
    assert sorted((a['repeat'], a['block']) for a in accepted if a['phase'] == 'validation') == [(r, k) for r in range(3) for k in range(10)]
    groups = {}
    for row in timings:
        groups.setdefault((row['method'], int(row['frame'])), []).append(row)
    data, plot, summary = {}, [], []
    for method in METHODS:
        rows = sorted([r for r in quality if r['method'] == method], key=lambda r: int(r['frame']))
        assert [int(r['frame']) for r in rows] == list(range(2000))
        values = {k: [] for k in ['frame', 'K', 'M', 'ratio', 'contributing_K', 'psnr', 'raster_ms', 'raster_wall_ms']}
        for row in rows:
            samples = groups[method, int(row['frame'])]
            assert sorted(int(r['repeat']) for r in samples) == [0, 1, 2]
            assert all(int(t['K']) == int(row['K']) and int(t['M']) == int(row['M']) for t in samples)
            assert int(row['contributing_K']) <= int(row['projected_K']) <= int(row['K'])
            if method == 'dynamic_04':
                f = frozen[int(row['frame'])]
                assert row['ids_sha256'] == f['ids_sha256'] and int(row['K']) == int(f['K'])
            for key in values:
                values[key].append(float(row[key]) if key in row else float(np.median([float(t[key]) for t in samples])))
        values = {k: np.asarray(v) for k, v in values.items()}
        assert all(np.isfinite(v).all() for v in values.values())
        if method == 'dynamic_04':
            assert np.allclose(values['ratio'], values['K']/values['M'], atol=1e-12, rtol=0)
        values['psnr_smooth'] = gaussian_filter1d(values['psnr'], sigma=35, mode='reflect')
        data[method] = values
        for i in range(2000):
            plot.append(dict(method=method, **{k: float(v[i]) for k, v in values.items()}))
        s = dict(method=method, frames=2000)
        for key in ['K', 'ratio', 'contributing_K', 'psnr', 'raster_ms', 'raster_wall_ms']:
            a = values[key]
            s.update({key+'_mean': float(a.mean()), key+'_p50': float(np.median(a)),
                key+'_p95': float(np.percentile(a, 95)), key+'_min': float(a.min()), key+'_max': float(a.max()),
                key+'_std': float(a.std(ddof=0))})
        s['raster_ms_cv_percent'] = 100*s['raster_ms_std']/s['raster_ms_mean']
        summary.append(s)
    write(reports/'PLOT_DATA.csv', plot)
    write(reports/'SUMMARY.csv', summary)
    save(reports/'SUMMARY.json', summary)
    d = data['dynamic_04']
    valid_ms = d['raster_ms']
    calibration = np.asarray([float(r['calibration_ms']) for r in frozen])
    calibration_possible = np.asarray([r['all_tested_below_band'] != 'True' for r in frozen])
    inside = (valid_ms >= .38) & (valid_ms <= .42)
    complete = []
    for i, f in enumerate(frozen):
        samples = sorted(groups['dynamic_04', i], key=lambda r: int(r['repeat']))
        complete.append(dict(frame=i, rgb_index=int(f['rgb_index']), timestamp=f['timestamp'],
            selected_K=int(f['K']), nominal_candidate_M=int(f['M']), eligible_N=int(f['eligible_N']),
            ratio=float(f['ratio']), contributing_K=int(d['contributing_K'][i]), psnr=float(d['psnr'][i]),
            target_ms=.4, calibration_ms=float(f['calibration_ms']),
            validation_repeat0_ms=float(samples[0]['raster_ms']), validation_repeat1_ms=float(samples[1]['raster_ms']),
            validation_repeat2_ms=float(samples[2]['raster_ms']), validation_median_ms=float(valid_ms[i]),
            validation_error_ms=float(valid_ms[i]-.4), within_038_042_ms=bool(inside[i]),
            all_tested_budgets_below_038_ms=f['all_tested_below_band'],
            max_pool_calibration_ms=float(f['max_pool_ms']), selected_pool_exhausted=f['selected_pool_exhausted'],
            tested_budgets=int(f['tested_budgets']), ids_sha256=f['ids_sha256']))
    write(reports/'PER_FRAME_DYNAMIC_BUDGET.csv', complete)
    result = dict(target_ms=.4, band_ms=[.38, .42], frames=2000,
        validation_mean_ms=float(valid_ms.mean()), validation_p50_ms=float(np.median(valid_ms)),
        validation_p95_ms=float(np.percentile(valid_ms, 95)), validation_max_ms=float(valid_ms.max()),
        within_band_frames=int(inside.sum()), within_band_percent=float(100*inside.mean()),
        below_band_frames=int((valid_ms < .38).sum()), above_band_frames=int((valid_ms > .42).sum()),
        within_band_among_search_possible_frames=int(inside[calibration_possible].sum()),
        search_possible_frames=int(calibration_possible.sum()),
        within_band_among_search_possible_percent=float(100*inside[calibration_possible].mean()) if calibration_possible.any() else None,
        all_tested_budgets_below_band_frames=int((~calibration_possible).sum()),
        calibration_mean_ms=float(calibration.mean()), calibration_validation_delta_mean_ms=float((valid_ms-calibration).mean()),
        K_min=int(d['K'].min()), K_max=int(d['K'].max()), K_mean=float(d['K'].mean()),
        ratio_min=float(d['ratio'].min()), ratio_max=float(d['ratio'].max()), ratio_mean=float(d['ratio'].mean()),
        objective_uses_psnr=False, offline_oracle=True, independent_validation=True)
    result.update(mean_absolute_target_error_ms=float(np.abs(valid_ms-.4).mean()),
        p95_absolute_target_error_ms=float(np.percentile(np.abs(valid_ms-.4), 95)),
        latency_cv_percent=float(100*valid_ms.std(ddof=0)/valid_ms.mean()))
    raw_dynamic = np.asarray([float(r['raster_ms']) for r in timings if r['method'] == 'dynamic_04'])
    result.update(single_render_samples=len(raw_dynamic), single_render_p95_ms=float(np.percentile(raw_dynamic, 95)),
        single_render_max_ms=float(raw_dynamic.max()), single_render_above_042_samples=int((raw_dynamic > .42).sum()))
    save(reports/'TARGET_ATTAINMENT.json', result)
    search_cost = dict(
        search_wall_s=float(sum(a['end_unix']-a['start_unix'] for a in attempts if a['phase'] == 'search')),
        tested_frame_K_candidates=sum(a['samples'] for a in attempts if a['phase'] == 'search'),
        process_elapsed_s=job['elapsed_s'],
        includes_CPU_and_render_search=True, includes_wait_and_quality_in_process_elapsed=True,
        excluded_from_pure_render_target=True)
    save(reports/'SEARCH_COSTS.json', search_cost)
    plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'font.size': 9.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'path'})
    fig = plt.figure(figsize=(13.3, 4.6))
    grid = fig.add_gridspec(1, 3, left=.059, right=.986, top=.76, bottom=.23, wspace=.33)
    count = fig.add_subplot(grid[0, 0])
    quality_ax = fig.add_subplot(grid[0, 1], sharex=count)
    sub = grid[0, 2].subgridspec(2, 1, height_ratios=[1, 2.1], hspace=.08)
    top = fig.add_subplot(sub[0], sharex=count)
    latency = fig.add_subplot(sub[1], sharex=count)
    reference = data['cartgs_full']['raster_ms']
    ours = np.concatenate([data[m]['raster_ms'] for m in METHODS[1:]])
    top.set_ylim(reference.min()-.1, reference.max()+.1)
    latency.set_ylim(max(0, ours.min()-.04), ours.max()+.04)
    top.spines['bottom'].set_visible(False)
    top.tick_params(axis='x', bottom=False, labelbottom=False)
    for ax, yy in [(top, 0), (latency, 1)]:
        ax.plot([-.012, .012], [yy-.018, yy+.018], transform=ax.transAxes, clip_on=False, color='#252525', lw=.8)
    x = np.arange(2000)
    for method, color, ls in zip(METHODS, COLORS, STYLES):
        value = data[method]
        width = 1.35 if method == 'dynamic_04' else 1.0
        for ax, array in [(count, value['contributing_K']/1000), (quality_ax, value['psnr_smooth']),
                          (top, value['raster_ms']), (latency, value['raster_ms'])]:
            line, = ax.plot(x, array, color=color, ls=ls, lw=width if ax in [count, quality_ax] else width*.65)
            assert np.array_equal(line.get_ydata(), array)
    latency.axhspan(.38, .42, color=COLORS[-1], alpha=.08)
    latency.axhline(.4, color='#626262', lw=.8, ls=(0, (4, 3)))
    count.set(title='(a) 实际 Alpha 贡献高斯点数', ylabel='高斯点数（千）', ylim=(0, None))
    quality_ax.set(title='(b) 每视角渲染质量', ylabel='PSNR (dB)')
    top.set_title('(c) 单图纯渲染耗时')
    fig.text(.695, .49, '纯渲染时间 (ms)', rotation=90, ha='center', va='center')
    for ax in [count, quality_ax, top, latency]:
        ax.set_xlim(0, 1999)
        ax.set_xticks([0, 500, 1000, 1500, 1999])
        ax.grid(alpha=.16)
    for ax in [count, quality_ax, latency]:
        ax.set_xlabel('RGB 帧编号')
    handles = [Line2D([], [], color=c, ls=ls, lw=1.6, label=label) for c, ls, label in zip(COLORS, STYLES, LABELS)]
    fig.legend(handles=handles, ncol=5, loc='upper center', bbox_to_anchor=(.51, .94), frameon=False, fontsize=9.2, handlelength=2.6)
    fig.suptitle('Replica office2 RGB-D｜2064×2208｜Mode 40｜动态预算目标0.4 ms', y=1.005, fontsize=12)
    fig.text(.52, .09, '动态K为逐帧离线搜索结果，固定后独立复测；阴影为0.38–0.42 ms，不表示硬时延保证。', ha='center', fontsize=8)
    fig.text(.52, .05, 'PSNR采用σ＝35帧高斯平滑；点数和耗时不平滑。耗时为三次单图中位数，不含准备、网络及预算搜索。', ha='center', fontsize=8)
    fig.text(.52, .01, '*4k有79帧候选不足；候选受限、低于目标的动态帧照实保留。', ha='center', fontsize=8)
    for ext in ['png', 'svg']:
        fig.savefig(output/f'OFFICE2_DYNAMIC_04_COMPARISON.{ext}', dpi=450, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.4), sharex=True)
    axes[0].plot(x, d['K'], color=COLORS[-1], lw=1, label='动态实际提交 K')
    axes[0].plot(x, [int(r['eligible_N']) for r in frozen], color='#8B8B8B', lw=.8, alpha=.7, label='有效候选上限')
    axes[0].set(ylabel='提交高斯点数', title='每帧应该提交多少个点：冻结预算表（未经平滑）')
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(x, d['ratio'], color=COLORS[-1], lw=1)
    axes[1].set(xlabel='RGB 帧编号', ylabel='实际 ratio＝K / M', ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.18)
        ax.set_xlim(0, 1999)
    fig.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(output/f'PER_FRAME_K_RATIO.{ext}', dpi=450, bbox_inches='tight')
    plt.close(fig)
    lines = ['# Mode40动态ratio：office2单图0.4ms目标实验', '',
        '## 结论与独立验证', '',
        f'完整2000帧离线搜索K，冻结预算后另做三轮单图渲染。目标0.40±0.02ms。实际平均{result["validation_mean_ms"]:.6f}ms，'
        f'p95 {result["validation_p95_ms"]:.6f}ms，{result["within_band_frames"]}/2000帧落入目标区间；'
        f'{result["above_band_frames"]}帧高于0.42ms，{result["below_band_frames"]}帧低于0.38ms。', '',
        f'K范围{result["K_min"]}–{result["K_max"]}，均值{result["K_mean"]:.2f}；ratio范围'
        f'{result["ratio_min"]:.6f}–{result["ratio_max"]:.6f}。逐帧完整结果见reports/PER_FRAME_DYNAMIC_BUDGET.csv。', '',
        f'搜索中{result["all_tested_budgets_below_band_frames"]}帧的所有已测预算（含完整有效候选）均低于0.38ms；'
        '这些帧不通过加点到候选外、复制点或延时凑目标。该搜索不是任意ratio的穷举，不能证明未测预算绝无其他局部峰值。', '',
        f'以上主结果为逐帧三次中位数。6000次动态单图原始计时p95为{result["single_render_p95_ms"]:.6f}ms，'
        f'最大{result["single_render_max_ms"]:.6f}ms，{result["single_render_above_042_samples"]}次高于0.42ms；不能将中位数上限理解为每次调用的硬保证。', '',
        '## 五方法对照', '', '| 方法 | 提交K均值 | Alpha贡献均值 | PSNR均值 | 纯渲染均值ms |', '|---|---:|---:|---:|---:|']
    for s, label in zip(summary, LABELS):
        lines.append('|'+label+'|'+'|'.join(f'{s[k+"_mean"]:.5f}' for k in ['K', 'contributing_K', 'psnr', 'raster_ms'])+'|')
    lines += ['', '## 搜索方法与边界', '',
        '- 唯一优化目标为最小化实测纯渲染中位数与0.4ms的绝对差，PSNR不进入搜索。候选仅来自当前名义Anchor视锥有效池。',
        '- 沿用同一输入的历史1/5/10/20/30/40/50/60/70/80/90/100%网格作为初始K提案依据；'
        '本轮重新实测历史最接近点、插值提案及有效候选上限，再最多8次局部细化。历史时间不进入最终结果。',
        '- 每候选一次暖机、三次单图CUDA Event测量；再最多8次局部实测细化。使用跨目标的区间或最佳邻域，不假设K与耗时严格线性或全局单调。',
        '- 每次候选测量前先渲染CaRtGS、重新准备候选以匹配调用条件；仅候选本身进入CUDA Event。'
        '最终图表仅用新冻结K的独立验证数据；目标区间始终为0.38–0.42ms。',
        '- 动态固定预算请求的网络ratio为K/M；M是名义候选总点数，有效候选数量另记。',
        '- 搜索与验证相互独立；预算表SHA在最终画质和三轮验证前冻结，不根据验证结果偷偷修改。',
        '- 离线逐帧查表属于可达性研究，不能称为在线因果自适应。搜索本身多次试渲染的时间另计，未塞进0.4ms渲染指标。',
        f'- 搜索区间累计墙钟{search_cost["search_wall_s"]:.2f}s，测试{search_cost["tested_frame_K_candidates"]}个帧×K候选；每候选另有暖机和三次单图试渲染。'
        '该成本不会凭查表结果消失，不适合直接当作每帧在线控制开销。',
        '- 各方法最终一次请求只渲染一张图；准备、Selector、Compensator、诊断计数、GT解码和显示均不在纯渲染计时内。',
        '- 所有实际Alpha贡献点数由Mode40既有diagnose统计，诊断RGB与实际RGB逐位一致；CaRtGS沿用已核验不变的原版贡献记录。',
        '- 原基线选点SHA和PSNR与前轮逐值匹配，所有数据来自相同冻结地图和网络；新增ratio只改变预算及其条件输入。',
        '- PSNR相对INTER_LINEAR放大的原RGB，非原生2064×2208真值。Mode40本身为近似渲染路径，不宣称RGB逐位等价于F2。',
        '- RTX3090、office2 RGB-D；不代表Quest3真机。两侧地图点数CaRtGS84203、本方法43197，分别使用自身SLAM位姿按RGB时间戳对齐。',
        f'- {validation["protected_files"]}个受保护文件前后SHA不变，无训练、无编译、无生产配置/共享环境修改，无commit/push，未保存渲染图片。', '',
        '## 运行环境', '', f'- 全程监控{job["observations"]}次，外部GPU活动{job["foreign_active_samples"]}次，外部CPU活动{job["foreign_cpu_active_samples"]}次。',
        '- 所有接受的搜索/验证块前后扩展窗口内外部活动均为0。GPU未锁频，监控包含等待间隙；结果不构成未来任意负载下的硬时延保证。',
        '- 搜索100帧块、验证200帧块，前后各2秒检测外部活动；受干扰整块重跑，所有现场保留，不按速度筛选。', '']
    (ROOT/'OFFICE2_DYNAMIC_04_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    sources = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [source/'BUDGETS_FROZEN.csv', source/'QUALITY_COUNTS.csv', source/'TIMINGS.csv', source/'VALIDATION.json',
         source/'BLOCK_ATTEMPTS.json', ROOT/'logs/formal_job.json', reports/'PER_FRAME_DYNAMIC_BUDGET.csv']}
    save(output/'DATA_MANIFEST.json', dict(real_data=True, sources=sources, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dpi=450, png_svg=True, psnr_sigma_frames=35, timing_smoothing=False, point_count_smoothing=False))
    (output/'data-manifest.md').write_text('# 图表数据清单\n\n全部来自本轮冻结K、实际Mode40贡献计数和独立计时。'
        '三联图含五方法，点数为实际Alpha贡献点；附图为提交K及K/M。'
        'PSNR仅展示时σ=35帧高斯平滑；点数、K、ratio和时间不平滑。原始输入SHA见DATA_MANIFEST.json。\n', encoding='utf-8')
    save(reports/'ANALYSIS_VALIDATION.json', dict(passed=True, frames=2000, timing_rows=30000, quality_rows=10000,
        all_frames_retained=True, frozen_K_ids_verified=True, independent_validation=True, offline_oracle=True,
        target_ms=.4, target_band_ms=[.38, .42], diagnostic_RGB_bitwise=True))
    print(json.dumps(dict(passed=True, target=result, summary=summary), ensure_ascii=False))


if __name__ == '__main__':
    main()

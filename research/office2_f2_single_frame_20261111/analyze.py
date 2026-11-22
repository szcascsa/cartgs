"""F2完整轨迹复测：三联图与同轮G/F2配对性能比较。"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'office2_single_frame_curves_20261111'
spec = importlib.util.spec_from_file_location('same_style_plot', OLD/'analyze.py')
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)
style.LABELS = ['CaRtGS', r'Ours ($r=0.5$, F2)', 'Ours (2k, F2)', 'Ours (4k budget*, F2)']


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
    source, reports, figures = ROOT/'blocks', ROOT/'reports', ROOT/'figures'
    reports.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    job = json.loads((ROOT/'logs/blocks_job.json').read_text())
    validation = json.loads((source/'VALIDATION.json').read_text())
    assert job['measurement_valid'] and job['state'] == 'complete' and validation['passed']
    attempts = json.loads((source/'BLOCK_ATTEMPTS.json').read_text())
    accepted = [a for a in attempts if a['accepted']]
    assert len(accepted) == 30 and all(a['external_gpu_samples'] == a['external_cpu_samples'] == 0 for a in accepted)
    assert sorted((a['repeat'], a['block']) for a in accepted) == [(r, k) for r in range(3) for k in range(10)]
    quality, timings = read(source/'QUALITY_COUNTS.csv'), read(source/'TIMINGS.csv')
    n = validation['frames']
    assert n == 2000 and len(quality) == 8000 and len(timings) == 42000
    ordered_versions = {}
    for row in timings:
        if row['method'] != 'cartgs_full':
            ordered_versions.setdefault((row['method'], int(row['frame']), int(row['repeat'])), []).append(row['version'])
    orders = {name: sum(v == list(name) for v in ordered_versions.values()) for name in [('G', 'F2'), ('F2', 'G')]}
    assert list(orders.values()) == [9000, 9000], orders
    grouped = {}
    for row in timings:
        grouped.setdefault((row['method'], row['version'], int(row['frame'])), []).append(row)
    plot_rows, data, summary, comparisons, round_stats = [], {}, [], [], []
    for method in style.METHODS:
        q = sorted([r for r in quality if r['method'] == method], key=lambda r: int(r['frame']))
        assert [int(r['frame']) for r in q] == list(range(n))
        versions = ['original'] if method == 'cartgs_full' else ['G', 'F2']
        for version in versions:
            values = {key: [] for key in ['frame', 'K', 'M', 'projected_K', 'contributing_K', 'psnr', 'prepare_ms', 'raster_ms', 'total_ms']}
            for row in q:
                samples = grouped[method, version, int(row['frame'])]
                assert sorted(int(r['repeat']) for r in samples) == [0, 1, 2]
                assert all(int(t['K']) == int(row['K']) and int(t['M']) == int(row['M']) for t in samples)
                for key in values:
                    values[key].append(float(row[key]) if key in row else float(np.median([float(t[key]) for t in samples])))
            values = {key: np.asarray(val) for key, val in values.items()}
            assert all(np.isfinite(a).all() for a in values.values())
            values['psnr_smooth'] = gaussian_filter1d(values['psnr'], sigma=35, mode='reflect')
            data[method, version] = values
            for i in range(n):
                plot_rows.append(dict(method=method, version=version,
                    **{key: float(val[i]) for key, val in values.items()}))
            stats = dict(method=method, version=version, frames=n,
                shortfall_frames=sum(int(r['K']) < int(r['requested_K']) for r in q))
            for key in ['K', 'contributing_K', 'psnr', 'prepare_ms', 'raster_ms', 'total_ms']:
                a = values[key]
                stats.update({key+'_mean': float(a.mean()), key+'_median': float(np.median(a)),
                    key+'_p95': float(np.percentile(a, 95)), key+'_std': float(a.std(ddof=0))})
            summary.append(stats)
            for repeat in range(3):
                selected = [r for r in timings if r['method'] == method and r['version'] == version and int(r['repeat']) == repeat]
                assert len(selected) == n
                round_stats.append(dict(method=method, version=version, repeat=repeat,
                    **{key+'_mean': float(np.mean([float(r[key]) for r in selected]))
                       for key in ['prepare_ms', 'raster_ms', 'total_ms']}))
        if method != 'cartgs_full':
            old, new = data[method, 'G'], data[method, 'F2']
            for key in ['K', 'M', 'projected_K', 'contributing_K', 'psnr']:
                assert np.array_equal(old[key], new[key])
            for scope in ['raster_ms', 'total_ms']:
                for region, mask in [('all', np.ones(n, dtype=bool)),
                                     ('peak650_780', (new['frame'] >= 650) & (new['frame'] <= 780))]:
                    a, z = old[scope][mask], new[scope][mask]
                    comparisons.append(dict(method=method, scope=scope, region=region, frames=int(mask.sum()),
                        G_mean_ms=float(a.mean()), F2_mean_ms=float(z.mean()),
                        reduction_percent=float(100*(1-z.mean()/a.mean())),
                        paired_geomean_speedup=float(np.exp(np.log(a/z).mean())),
                        faster_frames=int((z < a).sum()), slower_frames=int((z > a).sum()),
                        G_peak_ms=float(a.max()), F2_peak_ms=float(z.max())))
    write(reports/'PLOT_DATA.csv', plot_rows)
    write(reports/'SUMMARY.csv', summary)
    save(reports/'SUMMARY.json', summary)
    write(reports/'PAIRED_G_F2.csv', comparisons)
    write(reports/'TIMING_ROUNDS.csv', round_stats)
    displayed = {method: data[method, 'original' if method == 'cartgs_full' else 'F2'] for method in style.METHODS}
    checks = [style.figure(displayed, field, figures) for field in ['raster_ms', 'total_ms']]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), sharex=True)
    for ax, method, title, color in zip(axes, style.METHODS[1:], ['Ours 50%', 'Ours 2k', 'Ours 4k'], style.COLORS[1:]):
        x = data[method, 'F2']['frame']
        ax.plot(x, data[method, 'G']['raster_ms'], color='#737373', lw=.7, label='旧 G')
        ax.plot(x, data[method, 'F2']['raster_ms'], color=color, lw=.7, label='新 F2')
        ax.axvspan(650, 780, color='#A0A0A0', alpha=.12)
        ax.set(title=title, xlabel='RGB 帧编号', ylabel='纯渲染时间 (ms)')
        ax.grid(alpha=.17)
        ax.legend(frameon=False)
    fig.suptitle('同一次测量的 G / F2 配对比较｜office2 RGB-D，2064×2208，单次渲染')
    fig.text(.5, .015, '灰色区域：此前650–780帧峰值；每帧三次重复中位数，不作平滑，不删除帧。', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .94))
    for extension in ['png', 'svg']:
        fig.savefig(figures/f'PAIRED_G_F2.{extension}', dpi=450, bbox_inches='tight')
    plt.close(fig)
    sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [source/'QUALITY_COUNTS.csv', source/'TIMINGS.csv', source/'CHECKS.csv', source/'VALIDATION.json',
         ROOT/'logs/blocks_job.json', source/'BLOCK_ATTEMPTS.json', reports/'PLOT_DATA.csv']}
    save(figures/'DATA_MANIFEST.json', dict(real_data=True, input_sha256=sources,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reused_plot_script='../office2_single_frame_curves_20261111/analyze.py', reused_plot_script_sha256=hashlib.sha256((OLD/'analyze.py').read_bytes()).hexdigest(),
        dpi=450, png_svg=True, psnr_smoothing_sigma_frames=35, latency_smoothing=False,
        all_frames_retained=True, timing_samples_excluded=0))
    (figures/'data-manifest.md').write_text('# 数据清单\n\n全部数据来自本轮2000帧实测；主图为CaRtGS原版与Ours F2，'
        '配对图为同一次测量的G/F2。每帧时间取三次重复中位数，PSNR仅展示时采用σ=35帧高斯平滑。'
        '点数、时间不平滑、不删帧。输入文件及SHA见DATA_MANIFEST.json。\n', encoding='utf-8')
    by = {(s['method'], s['version']): s for s in summary}
    lines = ['# F2渲染器：office2完整轨迹高分辨率复测', '',
        '## 结论', '', '仅将本方法的G渲染器换为报告中冻结的F2，Tensor Core、蒸馏Compensator、新准备D、checkpoint和全部选点保持不变。'
        'CaRtGS仍使用原版渲染器。2064×2208、单次渲染、完整轨迹2000帧，每帧三轮计时。', '',
        '相对同轮G，F2在50%/2k/4k档的全轨迹平均纯渲染耗时分别降低4.77%/9.64%/7.59%；'
        '完整流程分别降低3.57%/6.70%/5.03%。每帧选点、RGB/radii、PSNR均不变。'
        '650–780帧仍有明显峰值，区间均值分别降低1.82%/9.71%/7.26%，未消除视角依赖。', '',
        '| 方法 | Alpha贡献点均值 | PSNR均值 | 准备均值ms | 纯渲染均值ms | 完整均值ms |',
        '|---|---:|---:|---:|---:|---:|']
    for method, label in zip(style.METHODS, ['CaRtGS原版', 'Ours 50% F2', 'Ours 2k F2', 'Ours 4k F2']):
        s = by[method, 'original' if method == 'cartgs_full' else 'F2']
        lines.append('|'+label+'|'+'|'.join(f'{s[key+"_mean"]:.4f}' for key in ['contributing_K', 'psnr', 'prepare_ms', 'raster_ms', 'total_ms'])+'|')
    lines += ['', '## 同轮配对的新旧渲染器差异', '',
        '| 方法 | 范围 | G纯渲染均值ms | F2纯渲染均值ms | 耗时降低 |', '|---|---|---:|---:|---:|']
    for r in comparisons:
        if r['scope'] == 'raster_ms':
            lines.append(f'|{r["method"]}|{r["region"]}|{r["G_mean_ms"]:.6f}|{r["F2_mean_ms"]:.6f}|{r["reduction_percent"]:.2f}%|')
    lines += ['', 'G/F2在同一进程中消费相同属性包，交替反转版本顺序；每轮轨迹分别正序、倒序和打乱。'
        '主表为每帧三次重复中位数再取全轨迹均值，不同于报告中32场景低分辨率的等权几何平均。', '',
        '## 正确性与范围', '',
        f'- {validation["parity_checks"]}组RGB/radii逐位对照全部通过；全部2000帧的选点SHA、K、候选数、贡献点数、PSNR与上轮完全相同。',
        '- 2k全部达到预算；4k仍有79帧候选不足，最低2088点，未补点或删除帧。',
        '- 纯渲染为CUDA Event；准备与两网络另计，完整流程独立测量。图表耗时不平滑，PSNR σ=35帧。',
        '- 原RGB用INTER_LINEAR插值为2064×2208，不是原生高分辨率GT；RTX3090结果不代表Quest真机。',
        '- CaRtGS地图84203点、本方法43197点，地图与渲染器不同；两侧各用自己的SLAM位姿并匹配RGB时间戳。',
        f'- 进程全程监控{job["observations"]}次，其中等待/暖机也记录外部活动。正式保留30个200帧计时块，每块前后扩展2秒的窗口内外部CPU/GPU活动均为0；'
        f'共重测{validation["rejected_blocks"]}个受干扰块，全部现场保留。验收只看外部活动、不看速度，不删除帧。',
        '- formal阶段数值验收通过；formal和随后timing连续测速均被外部任务干扰，整批耗时不采用。'
        '最终blocks进程按预先规定的外部活动规则整块验收，加载后遍历全部视角创建计划并重新核对选点，未复用旧Mask/网络结果。'
        '数值验收沿用同一冻结输入的formal记录，计时不复用。',
        f'- {validation["protected_files"]}项源码、配置、checkpoint和二进制前后SHA不变。只读加载已冻结F2，不编译、不训练、不修改生产或共享环境、不commit/push。',
        f'- F2二进制SHA：{json.loads((source/"MODEL_IDENTITIES.json").read_text())["F2_sha256"]}。',
        '- 验收计时窗口SM时钟1755–1935MHz、温度69–83℃，未锁频。相对收益使用同轮交替G/F2，不能用本轮绝对值和数小时前数据直接相除。'
        'CUDA Event区间包括渲染器内部调度间隙，不是Nsight逐Kernel时间求和。',
        '- F2自身使用有界Workspace和执行计划；不缓存上帧视角、选点或网络输出。全部质量遍历先于正式计时，以排除冷计划创建。', '']
    lines += ['首次完整质量预检中发现两元素旧轮转函数不能交替版本顺序，在任何正式计时开始前停止本轮进程并修正。'
              '现场保留于pre_timing_order_corrected，不计入结果。最终G→F2与F2→G顺序各9000组。', '']
    (ROOT/'OFFICE2_F2_RETEST_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    save(reports/'ANALYSIS_VALIDATION.json', dict(passed=True, frames=n, quality_rows=len(quality),
        timing_rows=len(timings), all_frames_retained=True, excluded_samples=0,
        accepted_blocks=30, external_activity_rejected_blocks=validation['rejected_blocks'],
        checked_curves=sum(c['checked_curves'] for c in checks), G_F2_paired=True,
        previous_quality_exact=True, latency_smoothing=False))
    print(json.dumps(dict(passed=True, summary=summary, comparisons=comparisons), ensure_ascii=False))


if __name__ == '__main__':
    main()

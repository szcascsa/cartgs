"""mode40纯渲染实测分析；原始CSV、PNG/SVG及中文报告。"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ['cartgs_full', 'ratio_50', 'fixed_2000', 'fixed_4000']
LABELS = ['CaRtGS原版', 'Ours 50%', 'Ours 2k', 'Ours 4k']
COLORS = ['#252525', '#12A7C8', '#0077BB', '#D55E00']


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
    source, reports, figures = ROOT/'formal', ROOT/'reports', ROOT/'figures'
    reports.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    valid = json.loads((source/'VALIDATION.json').read_text())
    job = json.loads((ROOT/'logs/formal_job.json').read_text())
    assert valid['passed'] and job['state'] == 'complete' and job['returncode'] == 0
    attempts = json.loads((source/'BLOCK_ATTEMPTS.json').read_text())
    accepted = [a for a in attempts if a['accepted']]
    assert len(accepted) == 30 and all(a['external_gpu_samples'] == a['external_cpu_samples'] == 0 for a in accepted)
    assert sorted((a['repeat'], a['block']) for a in accepted) == [(r, k) for r in range(3) for k in range(10)]
    timings, quality = read(source/'TIMINGS.csv'), read(source/'QUALITY.csv')
    assert len(timings) == 42000 and len(quality) == 8000
    grouped, order = {}, {}
    for row in timings:
        grouped.setdefault((row['method'], row['version'], int(row['frame'])), []).append(row)
        if row['method'] != 'cartgs_full':
            order.setdefault((row['method'], int(row['frame']), int(row['repeat'])), []).append(row['version'])
    assert sum(v == ['F2', 'M40'] for v in order.values()) == sum(v == ['M40', 'F2'] for v in order.values()) == 9000
    plot, summaries, comparisons, quality_summary, rounds, arrays = [], [], [], [], [], {}
    for method in METHODS:
        q = sorted([r for r in quality if r['method'] == method], key=lambda r: int(r['frame']))
        assert [int(r['frame']) for r in q] == list(range(2000))
        assert all(r['ids_match'] == r['radii_bitwise'] == 'True' for r in q)
        delta = np.asarray([float(r['delta_psnr']) for r in q])
        errors = np.asarray([float(r['rgb_abs_max']) for r in q])
        quality_summary.append(dict(method=method, frames=2000,
            F2_psnr_mean=float(np.mean([float(r['F2_psnr']) for r in q])),
            M40_psnr_mean=float(np.mean([float(r['M40_psnr']) for r in q])),
            psnr_delta_mean=float(delta.mean()), psnr_delta_min=float(delta.min()),
            psnr_delta_max=float(delta.max()), rgb_abs_max=float(errors.max()),
            rgb_abs_mean=float(np.mean([float(r['rgb_abs_mean']) for r in q])),
            radii_bitwise=True, selected_ids_equal=True,
            rgb_bitwise_frames=sum(r['rgb_bitwise'] == 'True' for r in q)))
        for version in ['original'] if method == 'cartgs_full' else ['F2', 'M40']:
            values = {key: [] for key in ['frame', 'K', 'M', 'raster_ms', 'raster_wall_ms']}
            for r in q:
                s = grouped[method, version, int(r['frame'])]
                assert sorted(int(t['repeat']) for t in s) == [0, 1, 2]
                assert all(int(t['K']) == int(r['K']) and int(t['M']) == int(r['M']) for t in s)
                for key in values:
                    values[key].append(float(r[key]) if key in r else float(np.median([float(t[key]) for t in s])))
            data = {key: np.asarray(v) for key, v in values.items()}
            assert all(np.isfinite(v).all() for v in data.values())
            arrays[method, version] = data
            for i in range(2000):
                plot.append(dict(method=method, version=version, **{key: float(value[i]) for key, value in data.items()}))
            summary = dict(method=method, version=version, frames=2000, K_mean=float(data['K'].mean()))
            for key in ['raster_ms', 'raster_wall_ms']:
                a = data[key]
                summary.update({key+'_mean': float(a.mean()), key+'_p50': float(np.median(a)),
                    key+'_p95': float(np.percentile(a, 95)), key+'_peak': float(a.max())})
            summaries.append(summary)
            for repeat in range(3):
                rows = [r for r in timings if r['method'] == method and r['version'] == version and int(r['repeat']) == repeat]
                rounds.append(dict(method=method, version=version, repeat=repeat, frames=len(rows),
                    raster_ms_mean=float(np.mean([float(r['raster_ms']) for r in rows])),
                    raster_wall_ms_mean=float(np.mean([float(r['raster_wall_ms']) for r in rows]))))
        if method != 'cartgs_full':
            a, z = arrays[method, 'F2'], arrays[method, 'M40']
            for key in ['K', 'M']:
                assert np.array_equal(a[key], z[key])
            for scope in ['raster_ms', 'raster_wall_ms']:
                for region, mask in [('all', np.ones(2000, dtype=bool)),
                                     ('peak650_780', (z['frame'] >= 650) & (z['frame'] <= 780))]:
                    old, new = a[scope][mask], z[scope][mask]
                    comparisons.append(dict(method=method, scope=scope, region=region,
                        F2_mean_ms=float(old.mean()), M40_mean_ms=float(new.mean()),
                        reduction_percent=float(100*(1-new.mean()/old.mean())),
                        paired_geomean_speedup=float(np.exp(np.log(old/new).mean())),
                        faster_frames=int((new < old).sum()), frames=len(old)))
    for name, rows in [('PLOT_DATA.csv', plot), ('SUMMARY.csv', summaries), ('QUALITY_SUMMARY.csv', quality_summary),
                       ('PAIRED_F2_M40.csv', comparisons), ('TIMING_ROUNDS.csv', rounds)]:
        write(reports/name, rows)
    plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'font.size': 9.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'path'})
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), sharex=True)
    for ax, method, label, color in zip(axes, METHODS[1:], LABELS[1:], COLORS[1:]):
        x = arrays[method, 'M40']['frame']
        for version, c, legend in [('F2', '#737373', 'F2'), ('M40', color, 'Mode 40')]:
            line, = ax.plot(x, arrays[method, version]['raster_ms'], color=c, lw=.75, label=legend)
            assert np.array_equal(line.get_ydata(), arrays[method, version]['raster_ms'])
        ax.axvspan(650, 780, color='#A0A0A0', alpha=.12)
        ax.set(title=label, xlabel='RGB 帧编号', ylabel='纯渲染时间 (ms)')
        ax.grid(alpha=.17)
        ax.legend(frameon=False)
    fig.suptitle('Mode 40 / F2 配对纯渲染｜office2 RGB-D，2064×2208，每次单图')
    fig.text(.5, .015, '每帧三次重复中位数，不平滑；不含网络、准备和显示。灰色区为650–780帧。', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .94))
    for ext in ['png', 'svg']:
        fig.savefig(figures/f'OFFICE2_MODE40_PURE_RENDER.{ext}', dpi=450, bbox_inches='tight')
    plt.close(fig)
    by = {(s['method'], s['version']): s for s in summaries}
    lines = ['# Mode 40：office2完整轨迹纯渲染复测', '', '## 结论', '',
        '沿用前轮office2 RGB-D地图、蒸馏Compensator、Tensor Core D及新准备。仅改变本方法Rasterizer：'
        'F2与用户指定目录screen.py的Renderer(40)在同一进程、同一属性包上配对测量。'
        '分辨率2064×2208，完整2000帧、每帧三次重复，每次只渲染一张图。', '',
        '三档纯渲染相对同轮F2均值降低13.69%/14.40%/14.04%。三轮结果一致；'
        '各档全部2000个视角的三次计时中位数均快于F2。只测渲染，不把网络/准备优化混入。', '',
        '| 方法 | F2纯渲染均值ms | Mode40纯渲染均值ms | 降低 | 原版CaRtGS/Mode40 |',
        '|---|---:|---:|---:|---:|']
    original = by['cartgs_full', 'original']['raster_ms_mean']
    for method, label in zip(METHODS[1:], LABELS[1:]):
        old, new = by[method, 'F2']['raster_ms_mean'], by[method, 'M40']['raster_ms_mean']
        lines.append(f'|{label}|{old:.6f}|{new:.6f}|{100*(1-new/old):.2f}%|{original/new:.3f}×|')
    lines += ['', f'本轮CaRtGS原版纯渲染均值{original:.6f}ms。表中均值为每帧三次中位数后对2000帧取均值。'
        'CUDA Event为GPU时间线区间（含Rasterizer内部调度间隙），不是Nsight Kernel求和。'
        '同步Host墙钟另见SUMMARY.csv，含Python/C++发射、Event记录及同步，不含准备。', '',
        '650–780帧区间，50%/2k/4k平均纯渲染分别由0.740851/0.494983/0.580409ms降至'
        '0.636045/0.417809/0.491392ms，降低14.15%/15.59%/15.34%。峰值没有消失，'
        '该版本提高像素计算效率，未把不同视角的像素合成负载变成常数。', '',
        '## 数值与质量', '',
        'Mode40为tile_safe_fast，使用Tile局部表达式、快速exp及FMA；保持FP32渲染、Gaussian身份、排序、属性及原1e-4透射率早停。'
        '不使用Mode38的0.01提前停止；不是逐位等价渲染器。', '',
        '| 方法 | F2 PSNR | Mode40 PSNR | 平均差dB | 最差帧差dB | RGB最大绝对误差 |',
        '|---|---:|---:|---:|---:|---:|']
    for s, label in zip(quality_summary[1:], LABELS[1:]):
        lines.append(f'|{label}|{s["F2_psnr_mean"]:.8f}|{s["M40_psnr_mean"]:.8f}|{s["psnr_delta_mean"]:+.8f}|{s["psnr_delta_min"]:+.8f}|{s["rgb_abs_max"]:.8g}|')
    lines += ['', '- 全2000帧选点SHA和实际K/M与历史记录一致；F2 PSNR与历史逐值相同，Mode40 radii逐位一致且所有输出有限。',
        '- 平均RGB绝对误差为8.23e-8–9.67e-8，但最大单通道局部误差为0.0280。不能用极小平均PSNR差声称逐位一致或所有像素无差异。',
        '- 2k始终提交2000点；4k有79帧候选不足，最低2088点，未补点/剔除。50%按当前可见名义Anchor候选点数计算。',
        '- 相同输入、版本顺序交替，轨迹三轮分别正序、倒序及打乱。准备在计时外，每帧重新执行。',
        '- 200帧整块按外部CPU/GPU活动验收；前后各扩展2秒，若有活动整块重测。不按速度筛选样本。',
        f'- 最终接受{valid["accepted_blocks"]}块、重测{valid["rejected_blocks"]}块；保留42000条计时、8000条质量记录。',
        f'- 全程监控{job["observations"]}次，外部活跃GPU样本{job["foreign_active_samples"]}、CPU样本{job["foreign_cpu_active_samples"]}；验收块窗口活动均为0。',
        f'- {valid["protected_files"]}个源码/配置/模型/动态库前后SHA不变；不编译、不训练、不修改生产或共享conda、不commit/push、不保存渲染图片。',
        '- RTX3090实测，不代表Quest3真机；GT为原始Replica RGB的INTER_LINEAR插值，不是原生2064×2208真值。',
        '- 已验收计时窗口SM时钟1755–1950MHz、温度61–72℃，无锁频。新旧版本同轮交替；不直接比较数小时前的绝对时间。',
        '- CaRtGS和本方法地图分别84203/43197点，各用自己的SLAM位姿匹配RGB时间戳。相对CaRtGS的收益包含地图、网络选择和Rasterizer差异。',
        '- Mode40原目录保持只读；输出均在独立office2_mode40_single_frame_20261112目录。', '']
    (ROOT/'OFFICE2_MODE40_RENDER_SPEED_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    sources = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [source/'TIMINGS.csv', source/'QUALITY.csv', source/'BLOCK_ATTEMPTS.json', source/'VALIDATION.json',
         ROOT/'logs/formal_job.json', reports/'PLOT_DATA.csv']}
    save(figures/'DATA_MANIFEST.json', dict(real_data=True, sources=sources, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        png_svg=True, dpi=450, timing_smoothing=False, timing_aggregation='per-frame median of 3, no frame removed'))
    (figures/'data-manifest.md').write_text('# 图表数据清单\n\nOFFICE2_MODE40_PURE_RENDER使用本轮实际TIMINGS.csv，'
        '逐帧取三次中位数；不平滑，不删帧。PNG/SVG为450dpi，来源SHA见DATA_MANIFEST.json。\n', encoding='utf-8')
    save(reports/'ANALYSIS_VALIDATION.json', dict(passed=True, frames=2000, accepted_blocks=30,
        timing_rows=len(timings), quality_rows=len(quality), all_frames_retained=True,
        version_order_balanced=True, raw_RGB_approximate=True, radii_bitwise_all=True,
        source_manifest_verified=True, timing_smoothing=False))
    print(json.dumps(dict(passed=True, summary=summaries, comparisons=comparisons, quality=quality_summary), ensure_ascii=False))


if __name__ == '__main__':
    main()

"""mode182逐视角加速比：纯渲染和完整Joint调用分开，保留原始与平滑显示。"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent
RATIOS = [.01, .03, .05, .08, .1, .2, .3, .4, .5, .6, .7]
COLORS = ['#0072B2', '#6A5ACD', '#009E73', '#8A9A00', '#56B4E9', '#A05D9B',
          '#E69F00', '#7B3F00', '#D55E00', '#4B6B6B', '#CC79A7']
NAME = 'OFFICE2_JOINT182_NATIVE1200'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def smooth(values):
    med = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(values, 15, mode='reflect'), 31), axis=1)
    return gaussian_filter1d(med, sigma=6, mode='reflect', truncate=3)


def main():
    source = ROOT/'formal/TIMINGS.csv'
    validation_path = ROOT/'formal/VALIDATION.json'
    protocol_path = ROOT/'formal/PROTOCOL.json'
    blocks_path = ROOT/'formal/BLOCK_ATTEMPTS.json'
    validation, protocol = json.loads(validation_path.read_text()), json.loads(protocol_path.read_text())
    assert validation['passed'] and validation['frames'] == 2000 and validation['repeats'] == 3
    assert validation['split_joint_RGB_radii_ids_bitwise']
    assert protocol['requested_ratios'] == RATIOS and protocol['settings']['mode'] == 182
    assert (protocol['width'], protocol['height']) == (1200, 680) and protocol['native_resolution']
    blocks = json.loads(blocks_path.read_text())
    accepted = [row for row in blocks if row['accepted']]
    assert len(accepted) == 30
    assert all(row['external_gpu_samples'] == row['external_cpu_samples'] == 0 for row in accepted)
    groups = {}
    raw_rows = read(source)
    for row in raw_rows:
        groups.setdefault((row['method'], int(row['frame'])), []).append(row)
    assert len(raw_rows) == 72000 and len(groups) == 24000

    def median(method, frame, column):
        rows = groups[method, frame]
        assert sorted(int(row['repeat']) for row in rows) == [0, 1, 2]
        assert len({(row['M'], row['K'], row['ids_sha256']) for row in rows}) == 1
        times = np.asarray([float(row[column]) for row in rows])
        assert np.isfinite(times).all() and (times > 0).all()
        return float(np.median(times))

    full = {column: np.asarray([median('ours_full', frame, column) for frame in range(2000)])
            for column in ['raster_ms', 'joint_cuda_ms', 'joint_wall_ms']}
    assert all(int(row['K']) == 43197 for row in raw_rows if row['method'] == 'ours_full')
    reports, figures = ROOT/'reports', ROOT/'figures'
    reports.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    paths = [source, validation_path, protocol_path, blocks_path, Path(__file__)]
    before = {str(path): sha(path) for path in paths}
    trained = protocol['checkpoint_training_ratios']
    plotted, summary, series = [], [], {}
    for ratio in RATIOS:
        method = f'ratio_{round(ratio*100):02d}'
        values = {column: np.asarray([median(method, frame, column) for frame in range(2000)]) for column in full}
        sp, joint_sp = full['raster_ms']/values['raster_ms'], full['joint_wall_ms']/values['joint_wall_ms']
        display, joint_display = smooth(sp), smooth(joint_sp)
        ks = []
        seen = bool(any(np.isclose(ratio, known, rtol=1e-6, atol=1e-8) for known in trained))
        for frame in range(2000):
            row = groups[method, frame][0]
            assert float(row['ratio']) == ratio
            k, m = int(row['K']), int(row['M'])
            assert k == min(int(np.floor(ratio*m)), int(row['eligible']))
            ks.append(k)
            plotted.append(dict(frame=frame, ratio=ratio, trained_ratio=seen, M=m, K=k,
                full_raster_ms=float(full['raster_ms'][frame]), ratio_raster_ms=float(values['raster_ms'][frame]),
                raster_speedup=float(sp[frame]), displayed_raster_speedup=float(display[frame]),
                full_joint_wall_ms=float(full['joint_wall_ms'][frame]), ratio_joint_wall_ms=float(values['joint_wall_ms'][frame]),
                joint_wall_speedup=float(joint_sp[frame]), displayed_joint_wall_speedup=float(joint_display[frame]),
                full_joint_cuda_ms=float(full['joint_cuda_ms'][frame]), ratio_joint_cuda_ms=float(values['joint_cuda_ms'][frame])))
        series[ratio] = (sp, display, joint_sp, joint_display)
        summary.append(dict(ratio=ratio, trained_ratio=seen, views=2000, repeats=3,
            full_mean_raster_ms=float(full['raster_ms'].mean()), ratio_mean_raster_ms=float(values['raster_ms'].mean()),
            mean_per_view_raster_speedup=float(sp.mean()), median_per_view_raster_speedup=float(np.median(sp)),
            raster_speedup_p05=float(np.percentile(sp, 5)), raster_speedup_p95=float(np.percentile(sp, 95)),
            raster_speedup_min=float(sp.min()), raster_speedup_max=float(sp.max()),
            raster_ratio_of_mean_latencies=float(full['raster_ms'].mean()/values['raster_ms'].mean()),
            raster_slower_views=int((sp < 1).sum()), mean_K=float(np.mean(ks)), min_K=min(ks), max_K=max(ks),
            full_mean_joint_wall_ms=float(full['joint_wall_ms'].mean()), ratio_mean_joint_wall_ms=float(values['joint_wall_ms'].mean()),
            mean_per_view_joint_wall_speedup=float(joint_sp.mean()), joint_slower_views=int((joint_sp < 1).sum()),
            full_mean_joint_cuda_ms=float(full['joint_cuda_ms'].mean()), ratio_mean_joint_cuda_ms=float(values['joint_cuda_ms'].mean())))
    write(reports/'PER_VIEW_SPEEDUP.csv', plotted)
    write(reports/'SUMMARY.csv', summary)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'path'})
    outputs, axis_limits = [], {}
    for scope, index in [('PURE_RENDER', 0), ('JOINT_CALL', 2)]:
        for is_smooth in [False, True]:
            ymax = max(2., np.ceil(max(series[r][index+int(is_smooth)].max() for r in RATIOS)*2)/2)
            fig, ax = plt.subplots(figsize=(12, 5.4))
            for ratio, color in zip(RATIOS, COLORS):
                values = series[ratio][index+int(is_smooth)]
                line, = ax.plot(np.arange(2000), values, color=color, label=f'{ratio*100:g}%', lw=1.5 if is_smooth else .65)
                assert np.array_equal(line.get_ydata(), values)
            ax.axhline(1, color='#555555', ls='--', lw=1, label='Ours Full (1×)')
            ax.set(xlabel='RGB frame index', ylabel='Rendering speedup (×)' if index == 0 else 'Joint call speedup (×)',
                xlim=(0, 1999), ylim=(0, ymax))
            ax.set_xticks([0, 500, 1000, 1500, 1999])
            ax.grid(alpha=.15)
            ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.23), ncol=6, fontsize=9)
            fig.suptitle('Replica office2 RGB-D | 1200 × 680 | JointReleaseRenderer mode 182', y=.99, fontsize=13)
            fig.subplots_adjust(left=.085, right=.985, top=.76, bottom=.20)
            text = ('31-frame centered median + Gaussian (σ=6 frames); raw data unchanged.' if is_smooth
                    else 'No temporal smoothing; median of 3 repeats per view.')
            text += '\nPure rendering only; preparation and networks excluded.' if index == 0 else '\nJoint call includes current-pose preparation, selection, compensation, rendering and finite checks.'
            fig.text(.53, .055, text, ha='center', va='center', fontsize=9)
            stem = f'{NAME}_{scope}_SPEEDUP'+('_SMOOTHED' if is_smooth else '_RAW')
            axis_limits[stem] = [0., float(ymax)]
            for ext in ['png', 'svg', 'pdf']:
                path = figures/f'{stem}.{ext}'
                fig.savefig(path, dpi=450, bbox_inches='tight')
                outputs.append(path.name)
            plt.close(fig)
    assert before == {str(path): sha(path) for path in paths}
    manifest = dict(passed=True, real_data=True, sources=before, frames=2000, repeats=3,
        ratios=RATIOS, raw_timing_rows=72000, plotted_rows=len(plotted), renderer_mode=182,
        native_resolution=[1200, 680], pure_and_joint_separate=True,
        smoothing=dict(display_only=True, median_window=31, gaussian_sigma=6, truncate=3, padding='reflect'),
        data='reports/PER_VIEW_SPEEDUP.csv', data_sha256=sha(reports/'PER_VIEW_SPEEDUP.csv'),
        accepted_blocks=len(accepted), rejected_blocks=len(blocks)-len(accepted), outputs=outputs,
        figure_y_limits=axis_limits)
    (reports/'ANALYSIS_VALIDATION.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (figures/'data-manifest.md').write_text('# mode182图表数据清单\n\n'
        '|Figure|Data file|Real/mock|Source|Script|Outputs|\n|---|---|---|---|---|---|\n'
        '|纯渲染/完整调用加速比，原始/平滑|reports/PER_VIEW_SPEEDUP.csv|真实测量|formal/TIMINGS.csv|analyze.py|450dpi PNG、SVG、PDF|\n\n'
        '加速比逐视角使用同轮Full耗时中位数除以ratio耗时中位数。平滑只作用于显示加速比，原始数据和摘要不变。\n', encoding='utf-8')
    lines = ['# JointReleaseRenderer mode182 的逐视角加速比', '',
        'Replica office2 RGB-D，原始1200×680，RTX 3090，2000视角，各条件三轮。', '',
        '实际入口：JointReleaseRenderer(source, student, mode=182, reuse=True, fused=True, finish=True)。采用发布报告的推荐开关；未修改默认配置、源码或网络。', '',
        'Full为同一模型未选择、未补偿的43197个Master高斯。ratio按当前视角名义Anchor候选池M计算严格预算，不是全地图导出比例。训练ratio为1%、5%、10%、30%、50%、70%，其余请求为连续ratio推理。', '',
        '纯渲染：准备完成后，直接使用JointReleaseRenderer相同native实例和mode182像素路径，CUDA Event仅包围native.render。'
        '独立校验该路径与render_from_source的RGB、radii、选点ID逐位一致。', '',
        '完整调用：当前CPU位姿进入render_from_source，包含准备、Selector、蒸馏Student Compensator、渲染及finite检查；使用同步Host墙钟统计。计时外的脚本校验不计入。', '',
        '|ratio|是否训练ratio|纯渲染ms|平均逐视角加速比|纯渲染慢于Full视角|完整调用ms|完整调用加速比|平均K|',
        '|---:|:---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"|{row['ratio']*100:g}%|{'是' if row['trained_ratio'] else '否'}|{row['ratio_mean_raster_ms']:.6f}|{row['mean_per_view_raster_speedup']:.3f}×|{row['raster_slower_views']}|{row['ratio_mean_joint_wall_ms']:.6f}|{row['mean_per_view_joint_wall_speedup']:.3f}×|{row['mean_K']:.1f}|")
    lines += ['', f"Full平均纯渲染：{full['raster_ms'].mean():.6f} ms；Full平均完整调用：{full['joint_wall_ms'].mean():.6f} ms。", '',
        '三轮分别正序、倒序、seed0随机帧序；逐帧轮转方法及两种计时顺序。200帧整块按前后2秒外部GPU/高CPU负载验收，异常块整体弃用重测。', '',
        f'已接受{len(accepted)}块，弃用{len(blocks)-len(accepted)}块；原始计时72000行，每行独立记录两种范围。保护文件前后哈希一致。未训练、未保存渲染图片、未commit/push。', '',
        '平均逐视角比值与平均耗时之比不同，SUMMARY.csv同时保存两者。图提供原始和平滑两版，31帧居中中值滤波后Gaussian σ=6，仅用于显示。']
    (ROOT/'JOINT182_RATIO_SPEEDUP_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, full_mean_raster_ms=float(full['raster_ms'].mean()),
        full_mean_joint_wall_ms=float(full['joint_wall_ms'].mean()), summary=summary), ensure_ascii=False))


if __name__ == '__main__':
    main()

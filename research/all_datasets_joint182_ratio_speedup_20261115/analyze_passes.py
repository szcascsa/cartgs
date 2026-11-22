"""32组全轨迹配对结果：每场景/模式一条ratio曲线，分别报告纯渲染和完整调用。"""
from pathlib import Path
import csv
import argparse
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
RATIOS = [.01, .03, .05, .08, .1, .2, .3, .4, .5, .6, .7]
METHODS = ['ours_full']+[f'ratio_{round(r*100):02d}' for r in RATIOS]
KINDS = ['Replica', 'TUM_RGBD', 'VECtor']
TITLES = {'Replica': 'Replica', 'TUM_RGBD': 'TUM', 'VECtor': 'VECtor'}
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#7B3F00', '#6A5ACD']
STYLE = {'mono': ('--', 'o'), 'rgbd': ('-', 's'), 'stereo': ('-', 's')}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label(scene):
    name = scene['scene'].replace('freiburg', 'fr').replace('_long_office_household', '_office')
    mode = {'mono': 'Mono', 'rgbd': 'RGB-D', 'stereo': 'Stereo'}[scene['mode']]
    return f"{name} / {mode} (N={scene['gaussians']:,})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-pass', action='store_true')
    args = parser.parse_args()
    repeat_ids = [0] if args.first_pass else [0, 1, 2]
    repeats = len(repeat_ids)
    destination = ROOT/'first_pass' if args.first_pass else ROOT
    destination.mkdir(exist_ok=True)
    queue = json.loads((ROOT/'QUEUE.json').read_text())
    assert all(row['formal'] == row['smoke'] == 'complete' for row in queue['scenes'])
    assert args.first_pass or queue['state'] == 'complete'
    scenes = json.loads((ROOT/'SCENES_FROZEN.json').read_text())
    assert len(scenes) == 32
    scenes.sort(key=lambda row: (KINDS.index(row['dataset_kind']), row['scene'], row['mode']))
    reports, figures = destination/'reports', destination/'figures'
    reports.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    summary, coverage, provenance = [], [], {}
    per_view = reports/'PER_VIEW_SPEEDUP.csv'
    per_stream = per_view.open('w', encoding='utf-8', newline='')
    per_writer = None
    total_rows, plotted_rows, checked_packets, protected_files, accepted_blocks = 0, 0, 0, 0, 0
    rejected_blocks = 0
    for scene in scenes:
        run = ROOT/'runs'/scene['slug']/'formal'
        run_dirs = [run]
        validation = json.loads((run/'VALIDATION.json').read_text())
        if not args.first_pass and validation['repeats'] == 1:
            run_dirs.append(run.parent/'remaining')
        protocol = json.loads((run/'PROTOCOL.json').read_text())
        identity = json.loads((run/'MODEL_IDENTITIES.json').read_text())
        n = scene['coverage']['evaluated_views']
        assert validation['passed'] and validation['warm_plan_misses'] == 0
        assert validation['frames'] == n and validation['repeats'] in [1, 3]
        assert validation['split_joint_RGB_radii_ids_bitwise'] and validation['ids_and_K_stable']
        assert protocol['dataset'] == scene['id'] and protocol['requested_ratios'] == RATIOS
        assert protocol['settings'] == dict(mode=182, reuse=True, fused=True, finish=True)
        assert identity['fingerprint'] and identity['student_sha256'] == scene['student_sha256']
        assert identity['ours_N'] == scene['gaussians'] and identity['student_parameters'] == 3534
        before, after = [json.loads((run/name).read_text()) for name in ['PROTECTED_BEFORE.json', 'PROTECTED_AFTER.json']]
        assert before == after
        blocks = json.loads((run/'BLOCK_ATTEMPTS.json').read_text())
        good = [row for row in blocks if row['accepted']]
        assert len(good) == validation['repeats']*math.ceil(n/500)
        assert all(row['external_gpu_samples'] == row['external_cpu_samples'] == 0 for row in good)
        counts = np.zeros((3, len(METHODS), n), dtype=np.int8)
        quantities = {key: np.empty((3, len(METHODS), n), dtype=np.float64) for key in
            ['raster_ms', 'joint_wall_ms', 'joint_cuda_ms', 'K', 'M', 'eligible']}
        hashes = np.empty((3, len(METHODS), n), dtype=object)
        combined_rows = []
        for current_run in run_dirs:
            if current_run != run:
                extra_v = json.loads((current_run/'VALIDATION.json').read_text())
                assert extra_v['passed'] and extra_v['repeats'] == 2 and extra_v['warm_plan_misses'] == 0
                assert json.loads((current_run/'PROTECTED_BEFORE.json').read_text()) == json.loads((current_run/'PROTECTED_AFTER.json').read_text())
                extra_blocks = json.loads((current_run/'BLOCK_ATTEMPTS.json').read_text())
                extra_good = [b for b in extra_blocks if b['accepted']]
                assert len(extra_good) == 2*math.ceil(n/500)
                assert all(b['external_gpu_samples'] == b['external_cpu_samples'] == 0 for b in extra_good)
                blocks.extend(extra_blocks)
                good.extend(extra_good)
                for filename in ['TIMINGS.csv', 'VALIDATION.json', 'PROTOCOL.json', 'MODEL_IDENTITIES.json', 'BLOCK_ATTEMPTS.json']:
                    provenance[str(current_run/filename)] = sha(current_run/filename)
            combined_rows.extend(row for row in read(current_run/'TIMINGS.csv') if int(row['repeat']) in repeat_ids)
        for row in combined_rows:
            repeat, method, frame = int(row['repeat']), METHODS.index(row['method']), int(row['frame'])
            assert 0 <= repeat < 3 and 0 <= frame < n
            counts[repeat, method, frame] += 1
            for key in quantities:
                quantities[key][repeat, method, frame] = float(row[key])
            hashes[repeat, method, frame] = row['ids_sha256']
        assert (counts[repeat_ids] == 1).all()
        quantities = {key: values[repeat_ids] for key, values in quantities.items()}
        hashes = hashes[repeat_ids]
        for key, values in quantities.items():
            assert np.isfinite(values).all(), (scene['id'], key)
            if key.endswith('_ms'):
                assert (values > 0).all()
            else:
                assert all(np.array_equal(values[0], other) for other in values[1:])
        assert all(np.array_equal(hashes[0], other) for other in hashes[1:])
        assert (quantities['K'][:, 0] == scene['gaussians']).all()
        medians = {key: np.median(values, axis=0) for key, values in quantities.items() if key.endswith('_ms')}
        for mi, ratio in enumerate(RATIOS, 1):
            k, m, eligible = [quantities[key][0, mi] for key in ['K', 'M', 'eligible']]
            assert np.array_equal(k, np.minimum(np.floor(ratio*m), eligible))
            speedup = medians['raster_ms'][0]/medians['raster_ms'][mi]
            joint = medians['joint_wall_ms'][0]/medians['joint_wall_ms'][mi]
            for frame in range(n):
                record = dict(scene=scene['id'], dataset=scene['dataset_kind'], mode=scene['mode'], frame=frame,
                    ratio=ratio, N=scene['gaussians'], M=int(m[frame]), K=int(k[frame]),
                    full_raster_ms=float(medians['raster_ms'][0, frame]), ratio_raster_ms=float(medians['raster_ms'][mi, frame]),
                    raster_speedup=float(speedup[frame]), full_joint_wall_ms=float(medians['joint_wall_ms'][0, frame]),
                    ratio_joint_wall_ms=float(medians['joint_wall_ms'][mi, frame]), joint_wall_speedup=float(joint[frame]))
                if per_writer is None:
                    per_writer = csv.DictWriter(per_stream, list(record))
                    per_writer.writeheader()
                per_writer.writerow(record)
                plotted_rows += 1
            summary.append(dict(scene=scene['id'], dataset=scene['dataset_kind'], scene_name=scene['scene'], mode=scene['mode'],
                ratio=ratio, N=scene['gaussians'], views=n, repeats=repeats, width=protocol['width'], height=protocol['height'],
                mean_K=float(k.mean()), min_K=int(k.min()), max_K=int(k.max()),
                full_mean_raster_ms=float(medians['raster_ms'][0].mean()), ratio_mean_raster_ms=float(medians['raster_ms'][mi].mean()),
                mean_raster_speedup=float(speedup.mean()), median_raster_speedup=float(np.median(speedup)),
                raster_speedup_p05=float(np.percentile(speedup, 5)), raster_speedup_p95=float(np.percentile(speedup, 95)),
                ratio_of_mean_raster_latencies=float(medians['raster_ms'][0].mean()/medians['raster_ms'][mi].mean()),
                raster_slower_views=int((speedup < 1).sum()), full_mean_joint_wall_ms=float(medians['joint_wall_ms'][0].mean()),
                ratio_mean_joint_wall_ms=float(medians['joint_wall_ms'][mi].mean()), mean_joint_wall_speedup=float(joint.mean()),
                joint_slower_views=int((joint < 1).sum()), full_mean_joint_cuda_ms=float(medians['joint_cuda_ms'][0].mean()),
                ratio_mean_joint_cuda_ms=float(medians['joint_cuda_ms'][mi].mean())))
        sample_count = int(counts.sum())
        coverage.append(dict(scene=scene['id'], dataset=scene['dataset_kind'], mode=scene['mode'], N=scene['gaussians'],
            width=protocol['width'], height=protocol['height'], **scene['coverage'], timing_rows=sample_count,
            accepted_blocks=len(good), rejected_blocks=len(blocks)-len(good), workspace_bytes=validation['renderer_workspace_bytes'],
            split_joint_bitwise=True, plan_misses=0, input_integrity=True))
        for path in [run/'TIMINGS.csv', run/'VALIDATION.json', run/'PROTOCOL.json', run/'MODEL_IDENTITIES.json', run/'BLOCK_ATTEMPTS.json']:
            provenance[str(path)] = sha(path)
        total_rows += sample_count
        checked_packets += validation['checked_packets']
        protected_files += len(before)
        accepted_blocks += len(good)
        rejected_blocks += len(blocks)-len(good)
        print('AGGREGATED', scene['id'], n, flush=True)
    per_stream.close()
    assert len(summary) == 352 and total_rows == sum(row['coverage']['evaluated_views'] for row in scenes)*12*repeats
    assert plotted_rows == sum(row['coverage']['evaluated_views'] for row in scenes)*11
    write(reports/'SUMMARY.csv', summary)
    write(reports/'COVERAGE.csv', coverage)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    outputs = []
    for scope, column in [('PURE_RENDER', 'mean_raster_speedup'), ('JOINT_CALL', 'mean_joint_wall_speedup')]:
        ymax = max(2., math.ceil(max(row[column] for row in summary)*2)/2+.15)

        def draw(ax, kind, legend):
            selected = [scene for scene in scenes if scene['dataset_kind'] == kind]
            names = sorted({scene['scene'] for scene in selected})
            color_by_scene = {name: COLORS[i] for i, name in enumerate(names)}
            for scene in selected:
                rows = sorted((row for row in summary if row['scene'] == scene['id']), key=lambda row: row['ratio'])
                assert [row['ratio'] for row in rows] == RATIOS
                x = np.asarray(RATIOS)
                y = np.asarray([row[column] for row in rows])
                linestyle, marker = STYLE[scene['mode']]
                line, = ax.plot(x, y, color=color_by_scene[scene['scene']], ls=linestyle, lw=1.3,
                    marker=marker, ms=3, markerfacecolor='white', markeredgewidth=.75, label=label(scene))
                assert np.array_equal(line.get_xdata(), x) and np.array_equal(line.get_ydata(), y)
            ax.axhline(1, color='#666666', ls=':', lw=1)
            ax.set(xlabel='Ratio', xlim=(0, .72), ylim=(0, ymax))
            ax.set_xticks(np.arange(0, .71, .1))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.grid(axis='y', alpha=.2, lw=.5)
            if legend:
                ax.legend(loc='upper left', bbox_to_anchor=(1.015, 1.02), frameon=False, fontsize=7.5,
                    handlelength=2., handletextpad=.5, labelspacing=.6)

        fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.2), sharey=True)
        for ax, kind in zip(axes, KINDS):
            draw(ax, kind, False)
            ax.set_title(TITLES[kind], fontsize=12, pad=8)
            ax.legend(loc='upper center', bbox_to_anchor=(.5, -.20), frameon=False, ncol=2, fontsize=6.6,
                handlelength=1.9, labelspacing=.35, columnspacing=.8)
        axes[0].set_ylabel('Rendering speedup (×)' if scope == 'PURE_RENDER' else 'Joint call speedup (×)')
        title = 'Rendering Speedup' if scope == 'PURE_RENDER' else 'Joint Call Speedup'
        if args.first_pass:
            title += ' — First Pass'
        fig.suptitle(title, fontsize=14, y=.99)
        fig.subplots_adjust(left=.055, right=.99, top=.89, bottom=.44, wspace=.10)
        for ext in ['png', 'pdf', 'svg']:
            path = figures/f'ALL_DATASETS_MODE182_{scope}_RATIO_SPEEDUP.{ext}'
            fig.savefig(path, dpi=600, bbox_inches='tight')
            outputs.append(str(path.relative_to(destination)))
        plt.close(fig)
        for kind in KINDS:
            fig, ax = plt.subplots(figsize=(7.5, 3.7))
            draw(ax, kind, True)
            ax.set_ylabel('Rendering speedup (×)' if scope == 'PURE_RENDER' else 'Joint call speedup (×)')
            ax.set_title(f"{TITLES[kind]} — "+('Rendering Speedup' if scope == 'PURE_RENDER' else 'Joint Call Speedup')+
                (' (1 repeat)' if args.first_pass else ' (3 repeats)'), fontsize=12, pad=9)
            fig.subplots_adjust(left=.095, right=.60, bottom=.17, top=.88)
            for ext in ['png', 'pdf', 'svg']:
                path = figures/f'{TITLES[kind].upper()}_MODE182_{scope}_RATIO_SPEEDUP.{ext}'
                fig.savefig(path, dpi=600, bbox_inches='tight')
                outputs.append(str(path.relative_to(destination)))
            plt.close(fig)
    sources = dict(provenance)
    for path in [ROOT/'SCENES_FROZEN.json', Path(__file__)]:
        sources[str(path)] = sha(path)
    result = dict(passed=True, groups=32, datasets=dict(Replica=16, TUM=10, VECtor=6),
        ratios=RATIOS, repeats=repeats, first_pass=args.first_pass, total_views=sum(row['coverage']['evaluated_views'] for row in scenes),
        raw_timing_rows=total_rows, per_view_rows=plotted_rows, summary_rows=len(summary),
        checked_packets=checked_packets, protected_file_checks=protected_files,
        accepted_blocks=accepted_blocks, rejected_blocks=rejected_blocks,
        all_accepted_windows_clean=True, all_split_joint_RGB_radii_ID_bitwise=True,
        no_warm_plan_misses=True, renderer_mode=182, all_students=3534,
        sources=sources, summary_sha256=sha(reports/'SUMMARY.csv'), per_view_sha256=sha(per_view),
        outputs=outputs, no_training=True, no_render_images=True, production_changed=False)
    (reports/'ANALYSIS_VALIDATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (figures/'data-manifest.md').write_text('# 三数据集mode182加速比曲线数据清单\n\n'
        '|Figure|Data file|Real/mock|Source|Script|Outputs|\n|---|---|---|---|---|---|\n'
        '|各数据集及合图，纯渲染/完整调用|reports/SUMMARY.csv|真实测量|runs/*/formal/TIMINGS.csv|analyze.py|600dpi PNG、PDF、SVG|\n\n'
        '每个场景每种相机模式一条曲线；同场景颜色相同，Mono虚线圆点，RGB-D/Stereo实线方点。'
        '每个ratio点先计算每视角本版各轮耗时中位数的Full/ratio，再对有效全轨迹等权取均值。'
        '不平滑、不插值；纯渲染和完整调用分开。N表示对应模型的Master总数，实际K见CSV。\n', encoding='utf-8')
    lines = ['# 三数据集全部场景/相机模式 mode182 加速比报告', '',
        '32组：Replica 8场景×Mono/RGB-D，TUM 5场景×Mono/RGB-D，VECtor 3场景×Mono/Stereo。'
        f"共{result['total_views']:,}个有效全轨迹视角，Full加11档ratio，各{repeats}轮；本版计时{total_rows:,}行。", '',
        '入口：JointReleaseRenderer(mode=182, reuse=True, fused=True, finish=True)，每组使用自己的冻结3534参数Student及匹配Master。'
        '这轮只测速度，不把蒸馏质量或GT质量门限当作通过；历史部署检查状态保留在SCENES_FROZEN.json。', '',
        '每个视角仅渲染一张图；VECtor Stereo表示双目训练checkpoint的单张左图输出，不是左右两图总耗时。'
        '分辨率分别为Replica 1200×680、TUM 640×480、VECtor 1224×1024，首尾原始RGB尺寸已核验。', '',
        'Full是该场景/模式全部未补偿Master高斯。局部ratio按名义Anchor候选池M计算严格K=min(floor(r×M),eligible)。'
        '加速比为各视角Full中位数耗时除以ratio中位数耗时，再对视角取算术平均，未混用跨场景Full。', '',
        '纯渲染只计JointReleaseRenderer同一native实例的native.render CUDA Event；准备和网络在计时外。'
        '完整调用使用当前CPU位姿的render_from_source同步Host墙钟，包含准备、选点、补偿、渲染与finite检查。', '',
        '每组先独立smoke和12视角×12条件的RGB/radii/选点ID逐位核验，再全轨迹计时；第一版仅正序一轮，最终版含正序、倒序、seed0随机顺序；方法和范围顺序轮转。'
        '预热有界容量计划，正式计时无计划未命中。每500帧以内整块按前后2秒外部负载验收。', '',
        '## 各组纯渲染平均逐视角加速比', '',
        '|场景/模式|N|视角|1%|3%|5%|8%|10%|20%|30%|40%|50%|60%|70%|',
        '|---|---:|---:|'+'---:|'*11]
    for scene in scenes:
        rows = [row for row in summary if row['scene'] == scene['id']]
        lines.append(f"|{scene['id']}|{scene['gaussians']}|{scene['coverage']['evaluated_views']}|"+'|'.join(f"{row['mean_raster_speedup']:.3f}×" for row in rows)+'|')
    lines += ['', '## 完整性与限制', '',
        f"输入/代码保护文件前后哈希比较{protected_files}项全部通过；接受{accepted_blocks}个负载干净的计时块，弃用{rejected_blocks}块。未训练、未改生产项目或共享环境、未保存渲染图片、未commit/push。", '',
        'TUM fr1_room Mono移除356条完全相同的时间戳/位姿重复行，仅评价978个独立视角；未更改原轨迹文件。各组覆盖与重复记录见reports/COVERAGE.csv。', '',
        '曲线展示平均值，不保证每个视角都更快；慢于Full的视角数量和5%/95%分位见SUMMARY.csv。'
        '不同场景地图大小、图像分辨率、Gaussian footprint和遮挡均不同，跨场景曲线差异不能全部归因于网络。'
        '两种耗时范围独立记录，不能用纯渲染倍数代替端到端收益。']
    (destination/'ALL_DATASETS_MODE182_RATIO_SPEEDUP_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['passed', 'groups', 'total_views', 'raw_timing_rows', 'per_view_rows', 'accepted_blocks', 'rejected_blocks']}), flush=True)


if __name__ == '__main__':
    main()

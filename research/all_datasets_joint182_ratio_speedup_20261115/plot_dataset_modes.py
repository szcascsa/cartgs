"""一张论文图、六条曲线：每数据集/模式等权平均场景加速比。"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
RATIOS = [.01, .03, .05, .08, .1, .2, .3, .4, .5, .6, .7]
GROUPS = [('Replica', 'mono', 'Replica Mono', '#0072B2', '--', 'o', 8),
          ('Replica', 'rgbd', 'Replica RGB-D', '#0072B2', '-', 's', 8),
          ('TUM_RGBD', 'mono', 'TUM Mono', '#D55E00', '--', 'o', 5),
          ('TUM_RGBD', 'rgbd', 'TUM RGB-D', '#D55E00', '-', 's', 5),
          ('VECtor', 'mono', 'VECtor Mono', '#009E73', '--', 'o', 3),
          ('VECtor', 'stereo', 'VECtor Stereo', '#009E73', '-', 's', 3)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-pass', action='store_true')
    args = parser.parse_args()
    root = ROOT/'first_pass' if args.first_pass else ROOT
    source, analysis = root/'reports/SUMMARY.csv', root/'reports/ANALYSIS_VALIDATION.json'
    validation = json.loads(analysis.read_text(encoding='utf-8'))
    assert validation['passed'] and validation['repeats'] == (1 if args.first_pass else 3)
    assert sha(source) == validation['summary_sha256']
    before = {str(path): sha(path) for path in [source, analysis]}
    with source.open(encoding='utf-8', newline='') as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 352 and len({row['scene'] for row in records}) == 32
    output = root/'figures/six_dataset_mode_curves'
    output.mkdir(parents=True, exist_ok=True)
    family = 'Times New Roman' if any(font.name == 'Times New Roman' for font in font_manager.fontManager.ttflist) else 'DejaVu Serif'
    plt.rcParams.update({'font.family': family, 'font.size': 9, 'axes.labelsize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    data, outputs, axis_limits = [], [], {}
    for scope, column in [('PURE_RENDER', 'mean_raster_speedup'), ('JOINT_CALL', 'mean_joint_wall_speedup')]:
        fig, ax = plt.subplots(figsize=(5.3, 3.6))
        peak = 1.
        for dataset, mode, label, color, linestyle, marker, expected_scenes in GROUPS:
            values = []
            for ratio in RATIOS:
                rows = [row for row in records if row['dataset'] == dataset and row['mode'] == mode and float(row['ratio']) == ratio]
                assert len(rows) == len({row['scene'] for row in rows}) == expected_scenes
                mean = float(np.mean([float(row[column]) for row in rows]))
                values.append(mean)
                data.append(dict(scope=scope, dataset=dataset, mode=mode, ratio=ratio,
                    scene_equal_weight_mean_speedup=mean, scene_count=expected_scenes,
                    total_views=sum(int(row['views']) for row in rows), repeats=validation['repeats']))
            line, = ax.plot(RATIOS, values, color=color, ls=linestyle, marker=marker,
                lw=1.6, ms=3.6, markerfacecolor='white', markeredgewidth=.85, label=label)
            assert np.array_equal(line.get_ydata(), np.asarray(values))
            peak = max(peak, max(values))
        ax.axhline(1, color='#6B6B6B', ls=':', lw=.9, zorder=0)
        ax.annotate('Our Complete Map (1×)', xy=(.705, 1.), xytext=(-3, -7),
            textcoords='offset points', ha='right', va='top', fontsize=8, color='#606060')
        ylim = [0., float(np.ceil(peak*2)/2+.1)]
        ax.set(xlabel='Ratio', ylabel='Rendering speedup (×)' if scope == 'PURE_RENDER' else 'Joint call speedup (×)',
            xlim=(0, .72), ylim=ylim)
        ax.set_title('Rendering Speedup' if scope == 'PURE_RENDER' else 'Joint Call Speedup', fontsize=12, pad=10)
        ax.set_xticks(np.arange(0, .71, .1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.tick_params(direction='out', length=3, width=.7)
        ax.grid(axis='y', alpha=.23, lw=.5)
        ax.legend(loc='upper right', frameon=False, ncol=1, handlelength=2.4,
            handletextpad=.5, labelspacing=.5, borderaxespad=.6)
        fig.subplots_adjust(left=.13, right=.98, top=.86, bottom=.17)
        stem = f'ALL_DATASETS_MODE182_SIX_CURVES_{scope}'
        axis_limits[scope] = ylim
        for extension in ['png', 'pdf', 'svg']:
            path = output/f'{stem}.{extension}'
            fig.savefig(path, dpi=600, facecolor='white')
            outputs.append(path)
        plt.close(fig)
    data_path = output/'SIX_CURVES_DATA.csv'
    with data_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    assert before == {str(path): sha(path) for path in [source, analysis]}
    manifest = dict(passed=True, curves=6, ratio_points_per_curve=11, repeats=validation['repeats'],
        first_pass=args.first_pass, source_sha256=before, data_sha256=sha(data_path),
        aggregation='先计算每场景全轨迹平均逐视角Full/ratio，再对同数据集同相机模式的场景等权平均',
        ratio='当前视角名义Anchor候选池上的预算比例', baseline_label='Our Complete Map (1×)',
        smoothing=False, linear_x=True,
        extra_rendering=False, extra_training=False, axis_limits=axis_limits,
        script_sha256=sha(Path(__file__)), outputs={path.name: sha(path) for path in outputs})
    (output/'FIGURE_VALIDATION.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/'data-manifest.md').write_text('# 六条数据集/模式加速比曲线\n\n'
        '|Figure|Data file|Real/mock|Source|Script|Outputs|\n|---|---|---|---|---|---|\n'
        '|纯渲染/完整调用|SIX_CURVES_DATA.csv|真实测量|reports/SUMMARY.csv|plot_dataset_modes.py|600dpi PNG、PDF、SVG|\n\n'
        f"本版每视角{validation['repeats']}轮。Replica每模式8个场景、TUM每模式5个场景、VECtor每模式3个场景。"
        '每个场景先以自己的Full为基准得到全轨迹平均逐视角加速比，再在数据集/模式内对场景等权平均。'
        '不按轨迹长度加权，不合并不同场景的耗时分母。六条线共享一个坐标系，真实ratio线性横轴，未平滑。'
        '纯渲染不含准备或网络；完整调用单独出图。VECtor Stereo为双目训练模型的单张左图推理。\n', encoding='utf-8')
    (output/'caption_zh.md').write_text('三数据集各相机模式下，局部预算渲染相对于Our Complete Map的纯渲染加速比。'
        '同一数据集使用相同颜色，单目为虚线圆点，RGB-D或双目为实线方点。'
        f"每视角测量{validation['repeats']}轮；先在每个场景内计算Our Complete Map与局部ratio的配对耗时比值均值，"
        '再对同数据集同模式的场景等权平均。Replica、TUM、VECtor分别采用1200×680、640×480、1224×1024原始分辨率。'
        'VECtor双目模式统计双目训练模型的单张左图渲染；准备和网络推理不计入。1×虚线表示Our Complete Map基准。\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, repeats=validation['repeats'], curves=6, data_points=66,
        pure_render_png=str(output/'ALL_DATASETS_MODE182_SIX_CURVES_PURE_RENDER.png')), ensure_ascii=False))


if __name__ == '__main__':
    main()

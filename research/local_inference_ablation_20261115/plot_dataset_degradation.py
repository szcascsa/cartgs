"""三个数据集的局部推理消融退化雷达图，读取已核验的真实实验结果。"""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Wedge
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'dataset_degradation'
SOURCE = ROOT/'reports/LOCAL_ABLATION_COMPARISON.csv'
SOURCE_SHA = '3bc9799f89f26c41cf0d11bf3a3fa3d966e506d1ba45b0cb7daea8c14c4f3825'
RATIOS = [.01, .03, .05, .08, .1, .2, .3, .4, .5, .6, .7]
VARIANTS = ['selector_no_comp', 'random_with_comp']
LABELS = {'selector_no_comp': '关闭 Compensator（保留 Selector）',
          'random_with_comp': '随机选择＋保留 Compensator'}
SHORT_LABELS = {'selector_no_comp': '无补偿', 'random_with_comp': '随机＋补偿'}
COLORS = {'selector_no_comp': '#6A99C1', 'random_with_comp': '#C99B2E'}
MARKERS = {'selector_no_comp': 's', 'random_with_comp': 'D'}
DATASETS = {
    'TUM': dict(source='TUM_RGBD', title='TUM RGB-D', modes=['mono', 'rgbd'],
        mode_text='单目与RGB-D', scenes=['freiburg1_desk', 'freiburg1_room', 'freiburg1_xyz',
                                    'freiburg2_xyz', 'freiburg3_long_office_household']),
    'REPLICA': dict(source='Replica', title='Replica', modes=['mono', 'rgbd'],
        mode_text='单目与RGB-D', scenes=['office0', 'office1', 'office2', 'office3',
                                    'office4', 'room0', 'room1', 'room2']),
    'VECTOR': dict(source='VECtor', title='VECtor', modes=['mono', 'stereo'],
        mode_text='单目与双目', scenes=['corner-slow', 'corridors-dolly', 'robot-normal']),
}
METRICS = {
    'psnr': dict(title='PSNR下降', unit='dB', maximum=18., ticks=[3., 6., 9., 12., 15., 18.], decimals=2),
    'ssim': dict(title='SSIM下降', unit='', maximum=.7, ticks=[.1, .2, .3, .4, .5, .6, .7], decimals=3),
    'lpips': dict(title='LPIPS上升', unit='', maximum=.25, ticks=[.05, .10, .15, .20, .25], decimals=3),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return math.fsum(values)/len(values)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate():
    assert digest(SOURCE) == SOURCE_SHA
    verified = json.loads((ROOT/'reports/VERIFIED.json').read_text(encoding='utf-8'))
    assert verified['complete'] and verified['groups'] == 32 and verified['new_evaluations'] == 704
    with SOURCE.open(encoding='utf-8-sig') as stream:
        raw = list(csv.DictReader(stream))
    assert len(raw) == 352
    lookup = {(row['group'], float(row['ratio'])): row for row in raw}
    assert len(lookup) == 352
    per_group, per_scene, dataset_means = [], [], []
    for tag, config in DATASETS.items():
        for scene in config['scenes']:
            for ratio in RATIOS:
                for variant in VARIANTS:
                    for metric in METRICS:
                        values = []
                        for mode in config['modes']:
                            group = f"{config['source']}/{scene}/{mode}"
                            row = lookup[group, ratio]
                            baseline = float(row['baseline_'+metric])
                            ablation = float(row[variant+'_'+metric])
                            degradation = ablation-baseline if metric == 'lpips' else baseline-ablation
                            assert math.isfinite(degradation)
                            sign = 1. if metric == 'lpips' else -1.
                            assert abs(degradation-sign*float(row[variant+'_delta_'+metric])) < 1e-12
                            point = dict(dataset=tag, scene=scene, mode=mode, group=group, ratio=ratio,
                                variant=variant, metric=metric, baseline=baseline, ablation=ablation,
                                degradation=degradation, N=int(row['N']), K_mean=float(row['K_mean']),
                                frames=int(row['frames']), baseline_source=row['baseline_source'])
                            per_group.append(point)
                            values.append(degradation)
                        per_scene.append(dict(dataset=tag, scene=scene, ratio=ratio, variant=variant,
                            metric=metric, degradation=mean(values), camera_modes=2))
        for ratio in RATIOS:
            for variant in VARIANTS:
                for metric in METRICS:
                    points = [row for row in per_scene if row['dataset'] == tag and row['ratio'] == ratio
                              and row['variant'] == variant and row['metric'] == metric]
                    value = mean([row['degradation'] for row in points])
                    assert len(points) == len(config['scenes'])
                    assert 0 <= value <= METRICS[metric]['maximum']
                    dataset_means.append(dict(dataset=tag, ratio=ratio, variant=variant, metric=metric,
                        degradation=value, scenes=len(points), scene_mode_groups=2*len(points),
                        aggregation='各场景内两种相机模式等权平均，再对场景等权平均'))
    assert len(per_group) == 2112 and len(per_scene) == 1056 and len(dataset_means) == 198
    data = OUTPUT/'data'
    write_csv(data/'PER_GROUP_DEGRADATION.csv', per_group)
    write_csv(data/'PER_SCENE_DEGRADATION.csv', per_scene)
    write_csv(data/'DATASET_DEGRADATION.csv', dataset_means)
    shutil.copyfile(SOURCE, data/SOURCE.name)
    return dataset_means


def configure_font():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})


def draw_radar(ax, rows, metric):
    config = METRICS[metric]
    angles = np.pi/2-np.arange(len(RATIOS))*2*np.pi/len(RATIOS)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.30, 1.30)
    ax.set_ylim(-1.30, 1.30)
    ax.add_patch(Circle((0, 0), 1, facecolor='#FBFCFD', edgecolor='none', zorder=0))
    for point in unit:
        ax.plot([0, point[0]], [0, point[1]], color='#DCE1E5', linewidth=.75, zorder=1)
    for tick in config['ticks']:
        radius = tick/config['maximum']
        ax.add_patch(Polygon(unit*radius, closed=True, fill=False, edgecolor='#CCD3D9',
                             linewidth=.85, linestyle=(0, (3, 3)), zorder=1))
        ax.text(-radius, .025, f'{tick:g}', fontsize=9, color='#63737F', va='bottom', ha='center',
                zorder=7, bbox=dict(facecolor='white', edgecolor='none', pad=.5, alpha=.9))
    ax.text(0, -.025, '0', ha='center', va='top', color='#63737F', fontsize=8, zorder=7)
    start, end = np.array(colors.to_rgb('#A5B6C1')), np.array(colors.to_rgb('#C4CED4'))
    half_width = 180/len(RATIOS)
    for index, (angle, ratio) in enumerate(zip(angles, RATIOS)):
        degree = math.degrees(angle)
        blend = index/(len(RATIOS)-1)
        ax.add_patch(Wedge((0, 0), 1.245, degree-half_width+.9, degree+half_width-.9,
            width=.122, facecolor=start*(1-blend)+end*blend, edgecolor='none', zorder=2))
        rotation = (degree-90+90) % 180-90
        ax.text(1.184*math.cos(angle), 1.184*math.sin(angle), f'{ratio*100:g}%', rotation=rotation,
                ha='center', va='center', fontsize=11, fontweight='bold', color='#26343F', zorder=3)
    for variant in VARIANTS:
        points = [next(row for row in rows if row['metric'] == metric and row['variant'] == variant
                       and row['ratio'] == ratio) for ratio in RATIOS]
        values = np.array([point['degradation'] for point in points])
        xy = unit*(values/config['maximum'])[:, None]
        ax.fill(xy[:, 0], xy[:, 1], color=COLORS[variant], alpha=.075, zorder=3)
        line, = ax.plot(np.r_[xy[:, 0], xy[0, 0]], np.r_[xy[:, 1], xy[0, 1]],
            color=COLORS[variant], linewidth=2.0, marker=MARKERS[variant], markersize=4.6,
            markerfacecolor='white', markeredgewidth=1.3, zorder=5)
        line.set_gid('degradation_'+metric+'_'+variant)
        # 首端的数值帮助定位幅度，其余Ratio的精确数值由下表完整给出。
        top_value = values[0]
        x = .13 if variant == 'random_with_comp' else -.13
        horizontal = 'left' if variant == 'random_with_comp' else 'right'
        text = ax.text(x, xy[0, 1]+.035, f'{top_value:.{config["decimals"]}f}',
                      ha=horizontal, va='bottom', fontsize=10, fontweight='bold', color=COLORS[variant],
                      zorder=8, bbox=dict(facecolor='white', edgecolor='none', pad=.7, alpha=.9))
        text.set_gid('first_ratio_value_'+metric+'_'+variant)
    title = config['title']+(f"（{config['unit']}）" if config['unit'] else '')
    ax.set_title(title, fontsize=18, fontweight='bold', color='#22313B', pad=12)


def draw_table(fig, rows, metric, bounds):
    ax = fig.add_axes(bounds)
    ax.axis('off')
    decimals = METRICS[metric]['decimals']
    cell_text = []
    for variant in VARIANTS:
        values = [next(row['degradation'] for row in rows if row['metric'] == metric
                      and row['variant'] == variant and row['ratio'] == ratio) for ratio in RATIOS]
        cell_text.append([SHORT_LABELS[variant]]+[f'{value:.{decimals}f}' for value in values])
    table = ax.table(cellText=cell_text, colLabels=['Ratio']+[f'{100*r:g}%' for r in RATIOS],
        colWidths=[.17]+[.83/11]*11, cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#DFE5E9')
        cell.set_linewidth(.5)
        cell.set_facecolor('#F0F4F6' if row == 0 else ('white' if row == 1 else '#F8FAFB'))
        if row == 0:
            cell.get_text().set_color('#33434F')
        elif column == 0:
            cell.get_text().set_color(COLORS[VARIANTS[row-1]])
            cell.get_text().set_fontweight('bold')


def plot_dataset(tag, config, rows):
    selected = [row for row in rows if row['dataset'] == tag]
    assert len(selected) == 66
    fig = plt.figure(figsize=(18.0, 8.9))
    for column, metric in enumerate(METRICS):
        ax = fig.add_axes([.012+column/3, .335, .309, .505])
        draw_radar(ax, selected, metric)
        draw_table(fig, selected, metric, [.012+column/3, .095, .309, .115])
        text = 'PSNR下降（dB）' if metric == 'psnr' else ('SSIM下降' if metric == 'ssim' else 'LPIPS上升')
        fig.text(1/6+column/3, .228, text, ha='center', va='center', fontsize=10, color='#33434F')
    count = len(config['scenes'])
    fig.suptitle(f"{config['title']} · {count}场景均值 · 局部Ratio消融退化", fontsize=23,
                 fontweight='bold', color='#22313B', y=.982)
    fig.text(.5, .927, f"每场景先平均{config['mode_text']}，再对{count}个场景等权平均",
             fontsize=12, ha='center', color='#536672')
    fig.text(.5, .890, '相对同Ratio完整局部方案的退化量，中心为0，越外表示质量下降越大',
             fontsize=11, ha='center', color='#536672')
    handles = [Line2D([], [], color=COLORS[variant], marker=MARKERS[variant], markerfacecolor='white',
                     linewidth=2., markersize=6, label=LABELS[variant]) for variant in VARIANTS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .273), ncol=2,
               frameon=False, fontsize=12, handlelength=2.2, columnspacing=3.)
    fig.text(.5, .050, 'PSNR / SSIM退化 = 完整方案 − 消融方案       LPIPS退化 = 消融方案 − 完整方案',
             ha='center', fontsize=10.5, color='#536672')
    fig.text(.5, .020, '同一指标在三个数据集使用相同线性径向刻度，11个Ratio等角度排列。随机选择使用seed=0。',
             ha='center', fontsize=9.5, color='#657782')
    destination = OUTPUT/'figures'/f'{tag}_LOCAL_ABLATION_DEGRADATION'
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(destination.with_suffix('.'+extension), dpi=450)
    plt.close(fig)
    return [destination.with_suffix('.'+extension) for extension in ['png','svg','pdf']]


def main():
    configure_font()
    rows = aggregate()
    figures = OUTPUT/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    manifest = ['# 局部消融退化图数据清单', '',
        '| 图 | 数据文件 | 数据性质 | 来源 | 脚本 | 输出 |', '|---|---|---|---|---|---|']
    for tag in DATASETS:
        manifest.append(f'| {tag}三联雷达图 | data/DATASET_DEGRADATION.csv | 真实实测 | LOCAL_ABLATION_COMPARISON.csv | plot_dataset_degradation.py | PNG／SVG／PDF |')
    manifest += ['', 'PSNR与SSIM的退化量取完整方案减消融，LPIPS取消融减完整方案。'
                 '先对各场景的两种相机模式等权平均，再对场景等权平均。']
    (figures/'data-manifest.md').write_text('\n'.join(manifest)+'\n', encoding='utf-8')
    files = [path for tag,config in DATASETS.items() for path in plot_dataset(tag,config,rows)]
    assert digest(SOURCE) == SOURCE_SHA
    metadata = dict(complete=True, datasets=3, figures=3, png=3, svg=3, pdf=3,
        source=str(SOURCE), source_sha256=SOURCE_SHA, real_data=True, source_unchanged=True,
        source_groups=32, source_comparison_rows=352, mean_rows=198,
        aggregation='各场景两种相机模式等权平均，然后场景等权平均',
        degradation={'psnr':'baseline-ablation','ssim':'baseline-ablation','lpips':'ablation-baseline'},
        radial_scales={key:[0,config['maximum']] for key,config in METRICS.items()},
        ratios=RATIOS, datasets_config=DATASETS,
        files={str(path.relative_to(OUTPUT)).replace('\\','/'):digest(path) for path in files})
    (OUTPUT/'SOURCE_MANIFEST.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (OUTPUT/'README_结果说明.md').write_text('# 三数据集局部Ratio消融退化图\n\n'
        '每个数据集一张三联雷达图，展示关闭Compensator与随机选择保留Compensator的质量退化。'
        '比较基准为同一Ratio的完整局部方案。PSNR／SSIM下降和LPIPS上升均为正值，向外表示退化增加。\n\n'
        'TUM为5场景，Replica为8场景，VECtor为3场景。每场景先平均两种相机模式，再对场景等权平均。'
        '每个模式的输入指标已是完整轨迹均值。模式和场景的权重均等。\n\n'
        '径向刻度在三个数据集中保持一致，PSNR为0–18 dB，SSIM为0–0.7，LPIPS为0–0.25。'
        'Ratio按参考图等角度排列，图下表保留全部11个Ratio的数值。\n\n'
        '无补偿使用LoD Fine终点Master属性。随机选择使用固定seed0优先级。数据为冻结checkpoint的推理消融，'
        '各视角的候选池和实际K已与完整方案匹配。完整精度与来源保存在data目录。\n', encoding='utf-8')
    with zipfile.ZipFile(OUTPUT/'LOCAL_ABLATION_DATASET_DEGRADATION.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for folder in ['figures','data']:
            for path in (OUTPUT/folder).rglob('*'):
                if path.is_file():
                    archive.write(path,path.relative_to(OUTPUT))
        for name in ['SOURCE_MANIFEST.json','README_结果说明.md']:
            archive.write(OUTPUT/name,name)
        archive.write(Path(__file__),'plot_dataset_degradation.py')
    print('DATASET_DEGRADATION_COMPLETE', 3, 'FIGURES', flush=True)


if __name__ == '__main__':
    main()

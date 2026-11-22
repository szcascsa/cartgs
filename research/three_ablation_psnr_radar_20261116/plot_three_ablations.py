"""从实测指标生成三个数据集的局部Ratio四项消融PSNR下降图。"""
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
RESEARCH = ROOT.parent
INFERENCE = RESEARCH/'local_inference_ablation_20261115'
NO_IMPORTANCE = RESEARCH/'no_importance_all_scenes_20261115/outputs/01a1de13-ebef-7091-b328-52d3c81cfc40'
RERUN = RESEARCH/'dual_view_li_vector_stereo_rerun_20261111/reports'
RATIOS = [.01, .03, .05, .08, .1, .2, .3, .4, .5, .6, .7]
VARIANTS = ['selector_no_comp', 'random_with_comp', 'no_importance', 'master_detach']
LABELS = ['关闭 Compensator', '随机选择＋保留 Compensator', '关闭 GI＋LI 重要性监督', 'Detach Master Gaussian']
SHORT_LABELS = ['无补偿', '随机＋补偿', '无重要性', 'Detach Master']
COLORS = ['#6A99C1', '#C99B2E', '#C54551', '#4C9B82']
MARKERS = ['s', 'D', 'o', '^']
RADIAL_MIN, RADIAL_MAX = -3., 18.
DATASETS = {
    'Replica': dict(title='Replica', groups=15, total=16, scenes=8, modes='单目＋RGB-D'),
    'TUM_RGBD': dict(title='TUM RGB-D', groups=7, total=10, scenes=4, modes='单目＋RGB-D'),
    'VECtor': dict(title='VECtor', groups=5, total=6, scenes=3, modes='单目＋双目'),
}
SOURCES = {
    'inference_comparison': INFERENCE/'reports/LOCAL_ABLATION_COMPARISON.csv',
    'inference_verification': INFERENCE/'reports/VERIFIED.json',
    'no_importance_metrics': NO_IMPORTANCE/'逐场景逐ratio完整指标.csv',
    'no_importance_status': NO_IMPORTANCE/'场景状态与失败.csv',
    'no_importance_verification': NO_IMPORTANCE/'交付核验.json',
    'rerun_comparison': RERUN/'RERUN_COMPARISON.csv',
    'rerun_verification': RERUN/'RERUN_VERIFIED.json',
    'master_detach_metrics': ROOT/'data/source_master_detach_lod_metrics.csv',
    'master_detach_verification': ROOT/'data/source_master_detach_final_audit.json',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (ROOT/'data'/name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return math.fsum(values)/len(values)


def aggregate():
    assert digest(SOURCES['inference_comparison']) == '3bc9799f89f26c41cf0d11bf3a3fa3d966e506d1ba45b0cb7daea8c14c4f3825'
    assert digest(SOURCES['no_importance_metrics']) == '74c7710e7301f3c5938a53a054e0cff66762cb87c9e64204c8c1c53ef8a4b0df'
    assert digest(SOURCES['master_detach_metrics']) == 'cf11e00e18911f418902376f259d4c6f8f5f6bacf76eb23a03b6c63d0759c85e'
    assert digest(SOURCES['master_detach_verification']) == '2d1a9803f7713b0287eb5d25e430e489a926dfc2544cc1bbbdc085f3d0121c00'
    inference = json.loads(SOURCES['inference_verification'].read_text(encoding='utf-8'))
    importance = json.loads(SOURCES['no_importance_verification'].read_text(encoding='utf-8'))
    rerun = json.loads(SOURCES['rerun_verification'].read_text(encoding='utf-8'))
    detach_audit = json.loads(SOURCES['master_detach_verification'].read_text(encoding='utf-8'))
    assert inference['complete'] and inference['new_evaluations'] == 704
    assert importance['完成数'] == 27 and importance['连续零选点失败数'] == 5
    assert rerun['complete'] and len(rerun['jobs']) == 2
    assert detach_audit['audit_passed'] and detach_audit['all_32_groups_succeeded']
    assert detach_audit['completed_evaluations'] == detach_audit['csv_rows'] == 736
    raw = read_csv(SOURCES['inference_comparison'])
    lookup = {(row['group'], float(row['ratio'])): row for row in raw}
    assert len(lookup) == len(raw) == 352
    local = [row for row in read_csv(SOURCES['no_importance_metrics']) if row['scope'] == 'view_ratio']
    assert len(local) == 352
    statuses = read_csv(SOURCES['no_importance_status'])
    successful = {row['id'] for row in statuses if row['status'] == '完成'}
    assert len(successful) == 27
    rerun_lookup = {(row['id'], float(row['ratio'])): row
                    for row in read_csv(SOURCES['rerun_comparison']) if row['scope'] == 'view_ratio'}
    detach_rows = read_csv(SOURCES['master_detach_metrics'])
    detach_lookup = {(row['id'], float(row['ratio'])): row for row in detach_rows if row['scope'] == 'view_ratio'}
    assert len(detach_rows) == 736 and len(detach_lookup) == 352
    coverage = [dict(group=row['id'], dataset=row['dataset'], scene=row['scene'], mode=row['mode'],
        included=row['id'] in successful, status=row['status'], failure=row['failure_error']) for row in statuses]
    points, audit, detach_all = [], [], []
    for (group, ratio), previous in lookup.items():
        detached = detach_lookup[group, ratio]
        assert int(previous['N']) == int(detached['gaussian_count'])
        assert int(previous['frames']) == int(detached['evaluated_frames'])
        assert float(previous['K_mean']) == float(detached['actual_K_mean'])
        detach_all.append(dict(dataset=detached['dataset'], scene=detached['scene'], mode=detached['mode'],
            group=group, ratio=ratio, baseline_psnr=float(previous['baseline_psnr']),
            detach_psnr=float(detached['psnr']),
            psnr_decrease=float(previous['baseline_psnr'])-float(detached['psnr']),
            frames=int(detached['evaluated_frames']), N=int(detached['gaussian_count']),
            K_mean=float(detached['actual_K_mean']), included_in_common_27=group in successful,
            baseline_source=previous['baseline_source'], detach_source=detached['output']))
    for row in local:
        group, ratio = row['id'], float(row['ratio'])
        if group not in successful:
            assert row['psnr_D'] == ''
            continue
        previous = lookup[group, ratio]
        detached = detach_lookup[group, ratio]
        baseline = float(previous['baseline_psnr'])
        original_baseline = float(row['psnr_A'])
        assert int(previous['N']) == int(row['gaussian_N'])
        assert int(previous['frames']) == int(row['frames'])
        assert float(previous['K_mean']) == float(row['actual_K_mean_D'])
        assert float(row['actual_K_mean_A']) == float(row['actual_K_mean_D'])
        assert int(detached['gaussian_count']) == int(row['gaussian_N'])
        assert int(detached['evaluated_frames']) == int(row['frames'])
        assert float(detached['actual_K_mean']) == float(previous['K_mean'])
        if (group, ratio) in rerun_lookup:
            repeated = rerun_lookup[group, ratio]
            assert abs(baseline-float(repeated['new_psnr'])) < 1e-12
            assert abs(original_baseline-float(repeated['old_psnr'])) < 1e-12
            assert 'dual_view_li_vector_stereo_rerun_20261111' in previous['baseline_source']
        else:
            assert abs(baseline-original_baseline) < 1e-12
        assert abs(float(row['psnr_D'])-original_baseline-float(row['delta_psnr'])) < 1e-12
        audit.append(dict(group=group, ratio=ratio, old_baseline_psnr=original_baseline,
            adopted_baseline_psnr=baseline, baseline_shift=baseline-original_baseline,
            old_no_importance_decrease=-float(row['delta_psnr']),
            adopted_no_importance_decrease=baseline-float(row['psnr_D']),
            baseline_source=previous['baseline_source'], old_baseline_model=row['model_A']))
        for variant in VARIANTS:
            if variant == 'no_importance':
                value, intervention, source = float(row['psnr_D']), '重新训练', row['evaluation_D']
            elif variant == 'master_detach':
                value, intervention, source = float(detached['psnr']), '重新训练', detached['output']
            else:
                value, intervention, source = float(previous[variant+'_psnr']), '推理干预', str(SOURCES['inference_comparison'])
            assert math.isfinite(value)
            points.append(dict(dataset=row['dataset'], scene=row['scene'], mode=row['mode'], group=group,
                ratio=ratio, variant=variant, intervention=intervention,
                baseline_psnr=baseline, ablation_psnr=value, psnr_decrease=baseline-value,
                frames=int(row['frames']), N=int(row['gaussian_N']), K_mean=float(previous['K_mean']),
                baseline_source=previous['baseline_source'],
                ablation_source=source))
    assert len(points) == 1188 and len(audit) == 297 and len(detach_all) == 352
    grouped = defaultdict(list)
    for point in points:
        grouped[(point['dataset'], point['scene'], point['ratio'], point['variant'])].append(point)
    scenes = [dict(dataset=dataset, scene=scene, ratio=ratio, variant=variant,
        camera_modes=','.join(sorted(point['mode'] for point in values)), mode_count=len(values),
        baseline_psnr=mean([point['baseline_psnr'] for point in values]),
        ablation_psnr=mean([point['ablation_psnr'] for point in values]),
        psnr_decrease=mean([point['psnr_decrease'] for point in values]))
        for (dataset, scene, ratio, variant), values in grouped.items()]
    averages = []
    for dataset, config in DATASETS.items():
        groups = {point['group'] for point in points if point['dataset'] == dataset}
        assert len(groups) == config['groups']
        for ratio in RATIOS:
            for variant in VARIANTS:
                values = [row for row in scenes if row['dataset'] == dataset and row['ratio'] == ratio
                          and row['variant'] == variant]
                value = mean([row['psnr_decrease'] for row in values])
                assert len(values) == config['scenes'] and RADIAL_MIN <= value <= RADIAL_MAX
                averages.append(dict(dataset=dataset, ratio=ratio, variant=variant,
                    baseline_psnr=mean([row['baseline_psnr'] for row in values]),
                    ablation_psnr=mean([row['ablation_psnr'] for row in values]), psnr_decrease=value,
                    scene_count=len(values), scene_mode_groups=len(groups)))
    assert len(scenes) == 660 and len(averages) == 132
    write_csv('GROUP_COVERAGE.csv', coverage)
    write_csv('BASELINE_REFERENCE_AUDIT.csv', audit)
    write_csv('MASTER_DETACH_ALL32_PSNR_DECREASE.csv', detach_all)
    write_csv('PER_GROUP_PSNR_DECREASE.csv', points)
    write_csv('PER_SCENE_PSNR_DECREASE.csv', scenes)
    write_csv('DATASET_PSNR_DECREASE.csv', averages)
    return averages


def draw_radar(ax, rows):
    angles = np.pi/2-np.arange(len(RATIOS))*2*np.pi/len(RATIOS)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.30, 1.30)
    ax.set_ylim(-1.30, 1.30)
    ax.add_patch(Circle((0, 0), 1, facecolor='#FBFCFD', edgecolor='none', zorder=0))
    zero_radius = (0-RADIAL_MIN)/(RADIAL_MAX-RADIAL_MIN)
    ax.add_patch(Circle((0, 0), zero_radius, facecolor='#EFF7F2', edgecolor='none', zorder=.5))
    for point in unit:
        ax.plot([0, point[0]], [0, point[1]], color='#DCE1E5', linewidth=.75, zorder=1)
    for tick in [0, 3, 6, 9, 12, 15, 18]:
        radius = (tick-RADIAL_MIN)/(RADIAL_MAX-RADIAL_MIN)
        ax.add_patch(Polygon(unit*radius, closed=True, fill=False, edgecolor='#CCD3D9',
                             linewidth=.85, linestyle=(0, (3, 3)), zorder=1))
        ax.text(-radius, .025, str(tick), fontsize=9, color='#63737F', va='bottom', ha='center',
                zorder=7, bbox=dict(facecolor='white', edgecolor='none', pad=.5, alpha=.9))
    ax.text(0, -.025, '−3', ha='center', va='top', color='#63737F', fontsize=8, zorder=7)
    start, end = np.array(colors.to_rgb('#A5B6C1')), np.array(colors.to_rgb('#C4CED4'))
    half_width = 180/len(RATIOS)
    for index, (angle, ratio) in enumerate(zip(angles, RATIOS)):
        degree = math.degrees(angle)
        blend = index/(len(RATIOS)-1)
        ax.add_patch(Wedge((0, 0), 1.245, degree-half_width+.9, degree+half_width-.9,
            width=.122, facecolor=start*(1-blend)+end*blend, edgecolor='none', zorder=2))
        ax.text(1.184*math.cos(angle), 1.184*math.sin(angle), f'{ratio*100:g}%',
                rotation=degree % 180-90, ha='center', va='center', fontsize=11,
                fontweight='bold', color='#26343F', zorder=3)
    for index, variant in enumerate(VARIANTS):
        values = np.array([next(row['psnr_decrease'] for row in rows if row['variant'] == variant
                                and row['ratio'] == ratio) for ratio in RATIOS])
        xy = unit*((values-RADIAL_MIN)/(RADIAL_MAX-RADIAL_MIN))[:, None]
        ax.fill(xy[:, 0], xy[:, 1], color=COLORS[index], alpha=.055, zorder=3)
        line, = ax.plot(np.r_[xy[:, 0], xy[0, 0]], np.r_[xy[:, 1], xy[0, 1]],
            color=COLORS[index], linewidth=2., marker=MARKERS[index], markersize=4.7,
            markerfacecolor='white', markeredgewidth=1.3, zorder=5)
        line.set_gid('psnr_decrease_'+variant)
        offsets = [-.20, .20, .07, -.07]
        ax.text(offsets[index], xy[0, 1]+.025, f'{values[0]:.2f}',
                ha='right' if offsets[index] < 0 else 'left', va='bottom', fontsize=9.4,
                fontweight='bold', color=COLORS[index], zorder=8,
                bbox=dict(facecolor='white', edgecolor='none', pad=.7, alpha=.9))


def draw_table(fig, rows, bounds):
    ax = fig.add_axes(bounds)
    ax.axis('off')
    values = [[SHORT_LABELS[index]]+[f'{next(row["psnr_decrease"] for row in rows if row["variant"] == variant and row["ratio"] == ratio):.2f}'
               for ratio in RATIOS] for index, variant in enumerate(VARIANTS)]
    table = ax.table(cellText=values, colLabels=['Ratio']+[f'{100*r:g}%' for r in RATIOS],
        colWidths=[.17]+[.83/11]*11, cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(7.3)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#DFE5E9')
        cell.set_linewidth(.5)
        cell.set_facecolor('#F0F4F6' if row == 0 else ('white' if row % 2 else '#F8FAFB'))
        if row == 0:
            cell.get_text().set_color('#33434F')
        elif column == 0:
            cell.get_text().set_color(COLORS[row-1])
            cell.get_text().set_fontweight('bold')


def plot(rows, dataset=None):
    combined = dataset is None
    selected = list(DATASETS) if combined else [dataset]
    fig = plt.figure(figsize=(18, 10.0) if combined else (8, 10.0))
    for column, name in enumerate(selected):
        config = DATASETS[name]
        points = [row for row in rows if row['dataset'] == name]
        ax = fig.add_axes([.012+column/3, .325, .309, .50] if combined else [.06, .355, .88, .54])
        draw_radar(ax, points)
        if combined:
            fig.text(1/6+column/3, .868, f'{config["title"]} · {config["scenes"]}场景均值',
                ha='center', fontsize=18, fontweight='bold', color='#22313B')
            fig.text(1/6+column/3, .838, f'{config["modes"]} · 有效{config["groups"]}／{config["total"]}组',
                ha='center', fontsize=10.5, color='#536672')
        draw_table(fig, points, [.012+column/3, .082, .309, .150] if combined else [.035, .080, .93, .155])
    handles = [Line2D([], [], color=COLORS[index], marker=MARKERS[index], markerfacecolor='white',
        linewidth=2., markersize=6, label=LABELS[index]) for index in range(len(VARIANTS))]
    title = '局部 Ratio 下四种消融的 PSNR 下降' if combined else f'{DATASETS[dataset]["title"]} · 四种局部Ratio消融'
    fig.suptitle(title, fontsize=23 if combined else 20, fontweight='bold', color='#22313B', y=.982)
    fig.text(.5, .935, 'PSNR下降 = 同Ratio完整方案 − 消融方案（dB）；越外表示下降越大',
        ha='center', fontsize=12 if combined else 10.5, color='#536672')
    if not combined:
        config = DATASETS[dataset]
        fig.text(.5, .910, f'{config["scenes"]}场景均值 · {config["modes"]} · 有效{config["groups"]}／{config["total"]}组',
            ha='center', fontsize=10, color='#536672')
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .260 if combined else .245),
        ncol=4 if combined else 2, frameon=False, fontsize=11 if combined else 9.5,
        handlelength=2.0, columnspacing=1.8)
    fig.text(.5, .052 if combined else .047,
        '四条线统一使用27个共同有效场景／模式；先平均同场景有效模式，再对场景等权平均。',
        ha='center', fontsize=10 if combined else 7.4, color='#536672')
    fig.text(.5, .023 if combined else .023,
        '三图统一−3–18 dB刻度；负值表示消融PSNR高于完整方案。关闭重要性与Detach Master为重新训练消融。',
        ha='center', fontsize=9.5 if combined else 7.2, color='#657782')
    name = 'THREE_DATASET_LOCAL_ABLATION_PSNR_DECREASE' if combined else f'{dataset.upper()}_FOUR_ABLATION_PSNR_DECREASE'
    paths = []
    for extension in ['png', 'svg', 'pdf']:
        path = ROOT/'figures'/f'{name}.{extension}'
        fig.savefig(path, dpi=450)
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei', 'axes.unicode_minus':False,
        'svg.fonttype':'none', 'pdf.fonttype':42, 'savefig.facecolor':'white'})
    for folder in ['data', 'figures']:
        (ROOT/folder).mkdir(exist_ok=True)
    hashes = {name:dict(path=str(path), sha256=digest(path)) for name, path in SOURCES.items()}
    averages = aggregate()
    for name in ['inference_comparison', 'no_importance_metrics', 'no_importance_status']:
        shutil.copyfile(SOURCES[name], ROOT/'data'/f'source_{name}.csv')
    (ROOT/'figures/data-manifest.md').write_text(
        '# 三数据集局部Ratio四项消融图数据清单\n\n'
        '| 图 | 数据文件 | 数据性质 | 来源 | 脚本 | 输出 |\n|---|---|---|---|---|---|\n'
        '| 三联图及三个独立图 | data/DATASET_PSNR_DECREASE.csv | 真实实测 | 推理消融、无GI／LI训练消融、Detach Master训练消融 | plot_three_ablations.py | 450 DPI PNG／SVG／PDF |\n\n'
        '下降量为最新完整方案减消融。四条线统一使用原图27组共同有效集合，模式内平均后场景等权。'
        'Detach Master全部32组另见data/MASTER_DETACH_ALL32_PSNR_DECREASE.csv。\n', encoding='utf-8')
    figures = plot(averages)
    for dataset in DATASETS:
        figures.extend(plot(averages, dataset))
    assert all(digest(SOURCES[name]) == record['sha256'] for name, record in hashes.items())
    manifest = dict(complete=True, real_data=True, original_sources_unchanged=True, sources=hashes,
        source_groups=32, included_groups=27, excluded_zero_selection_groups=5,
        datasets=DATASETS, ratios=RATIOS, variants=dict(zip(VARIANTS, LABELS)),
        definition='psnr_decrease=latest_complete_local_psnr-ablation_local_psnr',
        aggregation='各场景内有效相机模式等权平均，再对有效场景等权平均',
        baseline='VECtor corner-slow/robot-normal双目使用20261111复跑，其余沿用当前完整方案',
        intervention={'selector_no_comp':'固定模型推理干预', 'random_with_comp':'固定模型推理干预',
                      'no_importance':'GI=0、LI=0，重新训练Fine',
                      'master_detach':'Full、全局LoD、局部LoD三路均detach Master，重新训练Fine'},
        master_detach_all_groups=32, master_detach_evaluations=736,
        dataset_mean_rows=len(averages), radial_limits=[RADIAL_MIN,RADIAL_MAX],
        outputs={str(path.relative_to(ROOT)).replace('\\','/'):digest(path) for path in figures})
    (ROOT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (ROOT/'README_结果说明.md').write_text(
        '# 三数据集局部Ratio四种消融的PSNR下降\n\n'
        '已生成一张三联图，以及Replica、TUM RGB-D、VECtor各自的独立图。每图四条线分别为关闭补偿、随机选择保留补偿、关闭GI与LI重要性监督、Detach Master Gaussian。\n\n'
        'PSNR下降量统一为当前完整局部方案减消融，单位dB，正值表示消融质量降低，负值表示消融PSNR更高。11档Ratio等角度排列，三数据集统一使用−3–18dB线性径向刻度。表中保留两位小数，CSV保留完整精度，未平滑或裁剪下降量。\n\n'
        '## 有效数据\n\n'
        '关闭重要性有5组连续零选点失败。为维持四条线严格同组比较，图中共同剔除这些场景／模式，使用相同的27组，不以零质量填充。Replica为8场景15组，office0仅有单目；TUM为4场景7组，freiburg2_xyz仅有RGB-D，freiburg3_long_office_household两个模式均剔除；VECtor为3场景5组，corridors-dolly仅有单目。Detach Master自身已完成全部32组和736项评价，其全量逐组下降另存于data/MASTER_DETACH_ALL32_PSNR_DECREASE.csv。\n\n'
        '每组先使用全轨迹逐帧PSNR均值。同场景的有效模式等权平均，随后场景等权平均。因此，只有一个有效模式的场景仍占一个场景权重。逐组覆盖见data/GROUP_COVERAGE.csv。\n\n'
        '## 统一完整方案基准\n\n'
        'corner-slow与robot-normal双目的关闭重要性源表仍引用11月7日完整模型，另两项推理消融已使用11月11日复跑模型。本次按用户要求统一使用复跑模型PSNR，重新计算关闭重要性的下降量。每Ratio的旧基准、新基准和修正量保存在data/BASELINE_REFERENCE_AUDIT.csv，原始报告保持原值。两组复跑的Full起点、配置、初始化、关键帧序列和Ratio序列已核验一致。\n\n'
        '全部27组的N、评价帧数和局部实际K均核对一致。四种消融每Ratio使用同一最新完整方案PSNR；Detach Master的全部32组也逐项核对了N、帧数与实际K。\n\n'
        '## 实验解释范围\n\n'
        '关闭Compensator和随机选择属于冻结模型的推理干预，保留共同训练的Master。关闭重要性与Detach Master属于从原Full起点重新执行Fine：前者关闭GI／LI但允许Master更新，后者保留GI／LI并固定Master，只训练网络和Anchor Feature。四条曲线包含不同类型的干预，不能直接用幅度给模块因果作用排序。无重要性与Detach实验的健康停止规则均存在各自历史差异；单seed结果不包含重复试验置信区间。\n', encoding='utf-8')
    with zipfile.ZipFile(ROOT/'FOUR_ABLATION_PSNR_FIGURES.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in figures:
            archive.write(path, path.relative_to(ROOT))
        for path in (ROOT/'data').rglob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(ROOT))
        archive.write(ROOT/'figures/data-manifest.md', Path('figures/data-manifest.md'))
        for path in [Path(__file__), ROOT/'SOURCE_MANIFEST.json', ROOT/'README_结果说明.md']:
            archive.write(path, path.name)
    print(json.dumps({'complete':True, 'figures':4, 'included_groups':27, 'dataset_mean_rows':132,
        'output':str(ROOT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

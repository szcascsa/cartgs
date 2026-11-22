"""最终冻结证据绘图：实际K、归一化缩放、全部场景和速度—质量；不平滑原始测量。"""
from pathlib import Path
import hashlib
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
REPORTS = HERE / 'reports'
OUT = HERE / 'figures' / 'release_20261115'
PRIMARY = [f'local{x:g}' for x in [.1, .2, .3, .4, .5]]
COLORS = ['#595959', '#4E79A7', '#E15759', '#59A14F', '#B07AA1']
USED = set()
ENTRIES = []


def read(name):
    USED.add(name)
    return pd.read_csv(REPORTS / name)


def save(fig, name, inputs):
    fig.tight_layout()
    fig.savefig(OUT / (name + '.png'), dpi=450, bbox_inches='tight')
    fig.savefig(OUT / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)
    ENTRIES.append(dict(figure=name, inputs=inputs, type='真实测量', script=Path(__file__).name))


def ratio_rows(frame):
    return frame[frame.condition.str.startswith('local') | frame.condition.eq('full')]


def curve(ax, frame, variants, normalized=False):
    for color, (variant, label) in zip(COLORS, variants.items()):
        data = frame[frame.variant.eq(variant)]
        if data.empty:
            continue
        x, y = ('normalized_K', 'normalized_T') if normalized else ('K', 'T')
        data = data.groupby('condition').agg(x=(x, 'median'), y=(y, 'median'),
            low=(y, lambda a: a.quantile(.1)), high=(y, lambda a: a.quantile(.9))).sort_values('x')
        scale = 1. if normalized else 1000.
        ax.plot(data.x, data.y * scale, '.-', color=color, label=label, lw=1.05, markersize=3)
        if normalized:
            ax.fill_between(data.x, data.low, data.high, color=color, alpha=.07)
    if normalized:
        ax.plot([0, 1], [0, 1], '--', lw=.8, color='#888888', label='理想 T 比 = K 比')
        ax.set_xlabel('K / K_Full')
        ax.set_ylabel('T / T_Full')
        ax.set_xlim(0, 1.02)
    else:
        ax.set_xlabel('实际输入 Gaussian 数 K')
        ax.set_ylabel('同步墙钟中位延迟（μs）')
        ax.set_xscale('symlog', linthresh=100)
    ax.set_ylim(bottom=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for path in [Path('C:/Windows/Fonts/msyh.ttc'), Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.unicode_minus': False, 'legend.frameon': False, 'axes.grid': True, 'grid.alpha': .2,
        'svg.fonttype': 'none'})
    selected = json.loads((REPORTS / 'SELECTED_RASTER_20261115.json').read_text())
    variant = selected['variant']
    prefix = 'TAIL' if selected['mode'] == 203 else 'STRONG'
    pure = ratio_rows(read('RELEASE_THIN_PER_FRAME.csv.gz'))
    variants = {'F2': '冻结 F2', 'thin182': '起点最快 Thin182', 'new_thin': '最终 ' + variant}
    for scene in sorted(pure.scene.unique()):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
        for ax, workload, title in zip(axes, ['nested', 'student_lod'], ['A：固定属性嵌套删点', 'B：真实 Student LoD']):
            frame = pure[pure.scene.eq(scene) & pure.workload.eq(workload)]
            if frame.empty:
                ax.text(.5, .5, '缺少匹配 Student\n仅报告固定属性实验', ha='center', va='center', transform=ax.transAxes)
            curve(ax, frame, variants)
            ax.set_title(title)
        axes[0].legend(fontsize=6)
        fig.suptitle(scene)
        save(fig, 'raster_' + scene.replace('/', '__'), ['RELEASE_THIN_PER_FRAME.csv.gz'])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ax, workload, title in zip(axes, ['nested', 'student_lod'], ['固定属性：32 组', '真实 Student LoD：18 组']):
        curve(ax, pure[pure.workload.eq(workload)], variants, normalized=True)
        ax.set_title(title)
    axes[0].legend(fontsize=6)
    save(fig, 'raster_normalized', ['RELEASE_THIN_PER_FRAME.csv.gz'])
    stats = read('RELEASE_THIN_SCENE_RATIO.csv')
    frame = stats[stats.variant.eq('new_thin') & stats.condition.isin(PRIMARY)]
    for workload in ['nested', 'student_lod']:
        points = frame[frame.workload.eq(workload)].groupby(['scene', 'condition']).thin182_speedup.median().unstack()
        fig, ax = plt.subplots(figsize=(7.3, max(3, len(points) * .19 + 1.1)))
        image = ax.imshow(points, cmap='RdYlGn', aspect='auto', vmin=.85, vmax=1.25)
        ax.grid(False)
        ax.set_xticks(range(len(points.columns)), [x[5:] for x in points.columns])
        ax.set_yticks(range(len(points)), points.index, fontsize=6)
        for y in range(len(points)):
            for x in range(len(points.columns)):
                ax.text(x, y, f'{points.iloc[y, x]:.3f}×', ha='center', va='center', fontsize=6)
        ax.set_xlabel('Ratio')
        ax.set_title(f'{workload}：最终方案相对起点最快 Thin182（双轮中位加速比）')
        fig.colorbar(image, ax=ax, label='加速比；低于 1 表示变慢')
        save(fig, workload + '_incremental_heatmap', ['RELEASE_THIN_SCENE_RATIO.csv'])
    overview = pd.concat([read('TAIL_OVERVIEW.csv'), read('STRONG_OVERVIEW.csv')])
    overview = overview.drop_duplicates(['workload', 'variant'], keep='last')
    overview = overview[overview.variant.isin(['F2', 'previous182', 'alpha2', 'alpha3', 'alpha4'])]
    fast = read('RELEASE_THIN_OVERVIEW.csv')
    extra = []
    for workload in ['nested', 'student_lod']:
        q = overview[(overview.workload == workload) & (overview.variant == variant)].iloc[0]
        speed = fast[(fast.workload == workload) & (fast.variant == 'new_thin')].iloc[0]
        extra.append(dict(workload=workload, variant='最终轻量入口', psnr_loss=q.psnr_loss,
            F2_speedup=speed.F2_speedup, quality_pass=q.quality_pass))
    overview = pd.concat([overview, pd.DataFrame(extra)], ignore_index=True)
    global_pass = overview.groupby('variant').quality_pass.all().to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, workload in zip(axes, ['nested', 'student_lod']):
        for index, (_, row) in enumerate(overview[overview.workload.eq(workload)].iterrows()):
            good = bool(global_pass[row.variant])
            ax.scatter(row.psnr_loss, row.F2_speedup, color=COLORS[index % len(COLORS)], marker='o' if good else 'x', s=28)
            ax.annotate(row.variant, (row.psnr_loss, row.F2_speedup), xytext=(4, 4), textcoords='offset points', fontsize=6)
        ax.axvline(.2, ls='--', color='#E15759', lw=.8)
        ax.axhline(1, color='#888888', lw=.7)
        ax.set_xlabel('等权平均 PSNR 损失（dB，相对 F2）')
        ax.set_ylabel('主要 Ratio 相对 F2 加速比')
        ax.set_title(workload + '；× 为跨输入质量门限不通过')
    fig.suptitle('自然场景速度—质量；不代表合成极端重叠输入也合格', fontsize=9)
    save(fig, 'speed_quality_pareto', ['TAIL_OVERVIEW.csv', 'STRONG_OVERVIEW.csv', 'RELEASE_THIN_OVERVIEW.csv'])
    joint_file = REPORTS / 'RELEASE_JOINT_PER_FRAME.csv.gz'
    if joint_file.exists():
        joint = ratio_rows(read(joint_file.name))
        network_variants = {'A_fp32_F2': 'A FP32 + F2', 'B_fp32_new': 'B FP32 + 新渲染',
            'C_native_F2': 'C 已有 TC + F2', 'D_final': 'D 最终联合', 'I_previous': '起点已有联合'}
        for scene in sorted(joint.scene.unique()):
            fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
            frame = joint[joint.scene.eq(scene)]
            curve(axes[0], frame, network_variants)
            curve(axes[1], frame, network_variants, normalized=True)
            axes[0].set_title('当前 CPU 相机 → 准备 → 网络 → 渲染')
            axes[1].set_title('相对各自 Full 的归一化延迟')
            axes[0].legend(fontsize=6)
            fig.suptitle(scene)
            save(fig, 'joint_' + scene.replace('/', '__'), [joint_file.name])
    manifest = {name: hashlib.sha256((REPORTS / name).read_bytes()).hexdigest() for name in sorted(USED)}
    (OUT / 'DATA_SHA256.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    pd.DataFrame(ENTRIES).to_csv(OUT / 'DATA_MANIFEST.csv', index=False, encoding='utf-8-sig')
    (OUT / 'README.md').write_text('# 最终实测性能图\n\n所有数据均为真实测量。无模拟数据、无原始计时平滑。曲线为条件内中位数，阴影为帧和场景的10%—90%范围。\n\n固定属性与真实Student补偿分别绘制。Full提速会改变归一化分母，必须同时查看绝对延迟。\n\n数据文件、脚本及哈希见 DATA_MANIFEST.csv 和 DATA_SHA256.json。\n', encoding='utf-8')
    print('RELEASE_FIGURES_COMPLETE', len(ENTRIES))


if __name__ == '__main__':
    main()

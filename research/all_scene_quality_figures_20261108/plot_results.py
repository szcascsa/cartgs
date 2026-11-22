"""逐场景绘制真实全轨迹结果，PNG/SVG及完整数值CSV。"""
from collections import Counter
from pathlib import Path
import argparse
import csv
import hashlib
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT/'figures'
RATIOS = [.01, .03, .05, .08, .1, .2, .3, .4, .5, .6, .7]
TRAIN = {.01, .05, .1, .3, .5, .7}
COLORS = dict(global_lod='#0077BB', local_lod='#EE7733', ours_full='#009988', vanilla='#555555')
FIELDS = ['dataset', 'scene', 'mode', 'method', 'scope', 'ratio', 'ratio_kind',
          'psnr', 'ssim', 'lpips', 'gaussian_N', 'actual_K_mean', 'actual_K_min',
          'actual_K_max', 'evaluated_frames', 'evaluation_protocol', 'evaluation_path',
          'checkpoint', 'vanilla_repeat', 'fine_steps', 'fine_attempt']


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def records(group):
    common = {key: group[key] for key in ['dataset', 'scene', 'mode']}
    vanilla = group['vanilla']
    result = [dict(common, method='CaRtGS Vanilla', scope='full', ratio=1., ratio_kind='full',
                   **{key: vanilla[key] for key in ['psnr', 'ssim', 'lpips']},
                   gaussian_N=vanilla['gaussians'], actual_K_mean=vanilla['gaussians'],
                   actual_K_min=vanilla['gaussians'], actual_K_max=vanilla['gaussians'],
                   evaluated_frames=vanilla['frames'], evaluation_protocol='cartgs_original_rgb_full_trajectory',
                   evaluation_path=vanilla['evaluation'], checkpoint=vanilla['checkpoint'],
                   vanilla_repeat=vanilla['repeat'], fine_steps='', fine_attempt='')]
    for item in group['ours']:
        result.append(dict(common, method='双网络真实视角+LI', scope=item['scope'],
                           ratio=item['ratio'], ratio_kind=item['ratio_kind'],
                           **{key: item[key] for key in ['psnr', 'ssim', 'lpips']},
                           gaussian_N=item['gaussian_count'],
                           **{key: item[key] for key in ['actual_K_mean', 'actual_K_min', 'actual_K_max']},
                           evaluated_frames=item['frames'], evaluation_protocol=item['evaluation_protocol'],
                           evaluation_path=item['output'], checkpoint=group['trained_model'],
                           vanilla_repeat='', fine_steps=group['fine_steps'], fine_attempt=group['fine_attempt']))
    return result


def plot_group(group):
    identity = group['id'].replace('/', '__')
    target = FIGURES/group['dataset']
    target.mkdir(parents=True, exist_ok=True)
    vanilla = group['vanilla']
    ours_full, = [r for r in group['ours'] if r['scope'] == 'full']
    global_lod = sorted([r for r in group['ours'] if r['scope'] == 'global'], key=lambda r:r['ratio'])
    local_lod = sorted([r for r in group['ours'] if r['scope'] == 'view_ratio'], key=lambda r:r['ratio'])
    for rows in [global_lod, local_lod]:
        assert [r['ratio'] for r in rows] == RATIOS
        assert all(np.isfinite(r[key]) for r in rows for key in ['psnr', 'ssim', 'lpips', 'actual_K_mean'])
    assert len({r['frames'] for r in group['ours']}) == 1
    x = np.asarray(RATIOS, dtype=float)
    fig = plt.figure(figsize=(13.8, 12.7), facecolor='white')
    grid = fig.add_gridspec(3, 2, left=.072, right=.975, top=.865, bottom=.075,
                           height_ratios=[1., 1., 1.07], hspace=.41, wspace=.21)
    mode = {'mono': '单目', 'rgbd': 'RGB-D', 'stereo': '双目'}[group['mode']]
    fig.suptitle(f"{group['dataset']} · {group['scene']} · {mode}", y=.982,
                 fontsize=19, fontweight='bold', color='#203040')
    coverage = '帧数不同，未作共同帧重聚合' if not group['frames_match'] else '帧数相同，不代表轨迹和RGB关联逐帧相同'
    fig.text(.5, .949, f"CaRtGS 未校正原图口径 · 各自全轨迹均值 · 原版 {vanilla['frames']:,}帧 / 我们 {ours_full['frames']:,}帧 · {coverage}",
             ha='center', fontsize=9.2, color='#555555')
    handles = [Line2D([], [], color=COLORS['global_lod'], marker='o', lw=2., label='我们：全局 LoD'),
               Line2D([], [], color=COLORS['local_lod'], marker='o', lw=2., label='我们：局部视角 LoD'),
               Line2D([], [], color=COLORS['ours_full'], linestyle='-.', lw=1.8,
                      label=f"我们 Full（N={ours_full['gaussian_count']:,}）"),
               Line2D([], [], color=COLORS['vanilla'], linestyle='--', lw=1.8,
                      label=f"CaRtGS 原版 Full（N={vanilla['gaussians']:,}）")]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .932), ncol=2,
               frameon=False, fontsize=10, columnspacing=2.5, handlelength=3.)
    panels = [('psnr', 'PSNR ↑', 'dB'), ('ssim', 'SSIM ↑', ''),
              ('lpips', 'LPIPS ↓', ''), ('actual_K_mean', '实际送入 Renderer 的 Gaussian 数量', '点')]
    for index, (key, title, ylabel) in enumerate(panels):
        ax = fig.add_subplot(grid[index//2, index%2])
        ax.set_facecolor('#FAFBFC')
        for rows, color in [(global_lod, COLORS['global_lod']), (local_lod, COLORS['local_lod'])]:
            y = np.array([r[key] for r in rows])
            ax.plot(x, y, color=color, lw=1.9)
            for trained in [True, False]:
                mask = np.array([r in TRAIN for r in RATIOS]) == trained
                ax.scatter(x[mask], y[mask], color=color, edgecolor=color, s=26,
                           facecolor=color if trained else 'white', linewidth=1.3, zorder=4)
        full_value = ours_full[key] if key != 'actual_K_mean' else ours_full['gaussian_count']
        vanilla_value = vanilla[key] if key != 'actual_K_mean' else vanilla['gaussians']
        ax.axhline(full_value, color=COLORS['ours_full'], ls='-.', lw=1.6)
        ax.axhline(vanilla_value, color=COLORS['vanilla'], ls='--', lw=1.6)
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=9)
        ax.set_xticks(x, [f'{ratio:g}' for ratio in RATIOS])
        ax.set_xlim(0., .72)
        ax.set_xlabel('Ratio（实际数值线性轴）', fontsize=9, labelpad=5)
        ax.tick_params(axis='x', labelrotation=55)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8.7)
        ax.grid(axis='y', alpha=.24, linewidth=.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#BCC5CC')
        if key == 'actual_K_mean':
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value/1000:g}k' if value >= 1000 else f'{value:g}'))
            ax.text(.03, .07, '局部点数为跨视角平均 K；纵轴为对数', transform=ax.transAxes,
                    fontsize=8.2, color='#596670')
        else:
            precision = 3 if key != 'psnr' else 2
            ax.text(.985, .045, f"原版 {vanilla_value:.{precision}f}  |  我们 Full {full_value:.{precision}f}",
                    transform=ax.transAxes, ha='right', fontsize=8.3, color='#44515B',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=3))

    ax = fig.add_subplot(grid[2, :])
    ax.axis('off')
    table_rows = []
    def fmt_count(value):
        return f'{value:,.1f}' if value != round(value) else f'{value:,.0f}'
    for label, row, count in [('原版 Full', vanilla, vanilla['gaussians']),
                               ('我们 Full', ours_full, ours_full['gaussian_count'])]:
        table_rows.append([label, fmt_count(count), '—', f"{row['psnr']:.3f}", '—',
                           f"{row['ssim']:.4f}", '—', f"{row['lpips']:.4f}", '—'])
    for global_row, local_row in zip(global_lod, local_lod):
        ratio = global_row['ratio']
        table_rows.append([f"{ratio:g}{' *' if ratio not in TRAIN else ''}",
                           fmt_count(global_row['actual_K_mean']), fmt_count(local_row['actual_K_mean']),
                           f"{global_row['psnr']:.3f}", f"{local_row['psnr']:.3f}",
                           f"{global_row['ssim']:.4f}", f"{local_row['ssim']:.4f}",
                           f"{global_row['lpips']:.4f}", f"{local_row['lpips']:.4f}"])
    columns = ['Ratio / Full', '全局 K / Full N', '局部平均 K',
               '全局 PSNR', '局部 PSNR', '全局 SSIM', '局部 SSIM', '全局 LPIPS', '局部 LPIPS']
    table = ax.table(cellText=table_rows, colLabels=columns, colLoc='center', cellLoc='center',
                     bbox=[0., -.02, 1., 1.14], colWidths=[.13,.14,.13,.1,.1,.1,.1,.1,.1])
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#D9E0E5')
        cell.set_linewidth(.5)
        if i == 0:
            cell.set_facecolor('#E9EFF4')
            cell.set_text_props(weight='bold', color='#293E4C')
        elif i in [1, 2]:
            cell.set_facecolor('#F0F4F3' if i == 2 else '#F0F0F0')
        else:
            cell.set_facecolor('#FAFBFC' if i%2 == 0 else 'white')
    fig.text(.072, .035, '* / 空心点：未见 Ratio；其余为训练 Ratio。点数是送入 Renderer 的 K，不是实际参与 Alpha 合成的可见点数。',
             fontsize=8.8, color='#56636D')
    fig.text(.072, .018, '原版按本场景本相机模式 Full PSNR 最高重复取值，SSIM / LPIPS / N 绑定同一次；完整精度、局部 K 范围及来源见同名 CSV。',
             fontsize=8.4, color='#56636D')
    png, svg = target/(identity+'.png'), target/(identity+'.svg')
    fig.savefig(png, dpi=450)
    fig.savefig(svg)
    plt.close(fig)
    rows = records(group)
    write_csv(target/(identity+'.csv'), rows)
    return dict(id=group['id'], png=str(png.relative_to(ROOT)), svg=str(svg.relative_to(ROOT)),
                csv=str((target/(identity+'.csv')).relative_to(ROOT)), rows=len(rows),
                dimensions=list(Image.open(png).size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(ROOT/'data/ALL_SCENE_RESULTS.json'))
    parser.add_argument('--group', default=None)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding='utf-8'))
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei', 'axes.unicode_minus':False,
                         'svg.fonttype':'none', 'savefig.facecolor':'white'})
    groups = [g for g in data['groups'] if not args.group or g['id'] == args.group]
    outputs = []
    for group in groups:
        outputs.append(plot_group(group))
        print('PLOTTED', group['id'], flush=True)
    if args.group:
        return
    assert len(outputs) == 32
    rows = sum([records(group) for group in groups], [])
    assert len(rows) == 768
    write_csv(ROOT/'ALL_SCENE_METRICS.csv', rows)
    (ROOT/'FIGURES_INDEX.json').write_text(json.dumps(outputs, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# 各场景结果图', '',
             '每个数据集、场景、相机模式单独一张图，共32张，包含CaRtGS Vanilla Full及当前双网络真实视角＋LI方案的Full、全局LoD、局部视角LoD。', '',
             '统一采用CaRtGS未校正原图评价口径。VECtor三个双目场景的69条结果重新渲染评价；其余667条复用已完成的同口径评价。没有重新训练或保存渲染图片。', '',
             '训练Ratio：0.01、0.05、0.1、0.3、0.5、0.7；未见Ratio：0.03、0.08、0.2、0.4、0.6。图中空心点和表中*表示未见Ratio。', '',
             '横轴按实际Ratio数值使用线性坐标：0.01至0.1的距离为0.1至0.2距离的90%。低Ratio刻度倾斜以避免重叠，不对横轴做非线性拉伸，所有指标保持原值。', '',
             'Full点数为地图总数N，全局LoD为实际选中K，局部LoD为跨视角平均K；CSV另有最小／最大K。这些数值不是Alpha合成可见点数，不能当作统一same-K比较。', '',
             'CaRtGS按每个场景及相机模式Full全轨迹PSNR最高的一次重复取值，SSIM／LPIPS／N绑定同次。corridors-dolly双目原版只有两次完成，其余组为三次。', '',
             '图中的CaRtGS原版来自vanilla_all_datasets_20261031_2011归档实验，指当前项目关闭扩展模块后的Vanilla算法路径；来源checkpoint及评价目录逐行记录在CSV，不将其标为未修改上游仓库重新构建的结果。', '',
             '评价使用各自完整估计轨迹，未对共同RGB帧重聚合。图中明确展示两方评价帧数；相同帧数也不证明轨迹及RGB关联相同。只作各自全轨迹成绩展示，不作为共同帧配对显著性检验。', '',
             '当前训练模型的Full起点与选模记录保持不变。双目原图评价会包含训练校正图像与未校正评价图像之间的域差异，不将其解释为训练退化。', '',
             '| 数据集／场景／模式 | 结果图 | 矢量图 | 精确数值 |', '|---|---|---|---|']
    for output in outputs:
        lines.append(f"| {output['id']} | [PNG]({output['png']}) | [SVG]({output['svg']}) | [CSV]({output['csv']}) |")
    (ROOT/'README_结果说明.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    changes_file = ROOT/'data/ORIGINAL_RGB_REEVALUATION.json'
    if changes_file.exists():
        changes = json.loads(changes_file.read_text(encoding='utf-8'))
        change_rows = []
        report = ['# CaRtGS 未校正原图评价更新', '',
                  '已对当前双网络真实视角＋LI模型的VECtor三个双目场景重做69条全轨迹评价，包含Full、11档全局Ratio及11档局部Ratio。其余29组667条原有评价已使用相同原图口径，复用并核验。', '',
                  '只移除评价命令中的stereo-camera-config及stereo-native-maps，不改变渲染相机、网络、checkpoint、GT时间关联或选点。全部69条评价帧关联、实际K一致，66条LoD选点身份一致。', '',
                  'PSNR／SSIM／LPIPS保留CaRtGS相同的TorchMetrics参数、输入排列和逐帧调用；不作尺度对齐，不保存渲染或GT图片。', '',
                  '| 双目场景 Full | 校正域 PSNR | 原图 PSNR | 校正域 SSIM | 原图 SSIM | 校正域 LPIPS | 原图 LPIPS |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for change in sorted(changes, key=lambda row:(row['id'], row['scope'], row['ratio'])):
            old, new = change['previous_result'], change['result']
            change_rows.append(dict(id=change['id'], scope=change['scope'], ratio=change['ratio'],
                                   **{prefix+key:value[key] for prefix, value in [('old_',old),('new_',new)]
                                      for key in ['psnr','ssim','lpips']},
                                   frames=new['evaluated_frames'], actual_K_mean=new['actual_K_mean'],
                                   previous_output=change['previous_output'], current_output=new['output']))
            if change['scope']=='full':
                report.append(f"| {change['id']} | {old['psnr']:.6f} | {new['psnr']:.6f} | {old['ssim']:.6f} | {new['ssim']:.6f} | {old['lpips']:.6f} | {new['lpips']:.6f} |")
        with (ROOT/'EVALUATION_PROTOCOL_CHANGES.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, list(change_rows[0]))
            writer.writeheader()
            writer.writerows(change_rows)
        report += ['', '新值已写回服务器原批次LOD_PIPELINE.json、reports/LOD_METRICS.csv；原校正域逐帧评价完整保留，更新前汇总备份在before_original_rgb。训练模型哈希未改变。', '',
                   '这次变化反映评价图像域改变。模型训练仍采用原有训练域，不将原图指标下降解释为模型权重或训练行为退化。Full起点选模记录仍保留历史口径，不重新挑选或训练模型。', '',
                   '全部32图和768条汇总记录见ALL_SCENE_METRICS.csv。逐ratio新旧差异及原路径见EVALUATION_PROTOCOL_CHANGES.csv。']
        (ROOT/'EVALUATION_PROTOCOL_UPDATE.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    manifest = ['# 图表数据来源', '',
                '真实实验数据。源JSON逐条对应原始metric_summary.json及per_frame_metrics.csv，736条我们方法评价已核对均值；原版使用PER_RUN.csv并核对原逐帧txt。', '',
                '| 图 | 数据 | 来源 | 脚本 | 输出 |', '|---|---|---|---|---|']
    for output in outputs:
        manifest.append(f"| {output['id']} | {output['csv']} | data/ALL_SCENE_RESULTS.json的同名group | plot_results.py | {output['png']}；{output['svg']} |")
    (FIGURES/'data-manifest.md').write_text('\n'.join(manifest)+'\n', encoding='utf-8')
    # 联系图只用于快速查找；阅读具体指标请打开单场景450dpi原图。
    thumb_w, thumb_h = 345, 318
    contact = Image.new('RGB', (thumb_w*4, thumb_h*8), 'white')
    for index, output in enumerate(outputs):
        with Image.open(ROOT/output['png']) as image:
            image = ImageOps.contain(image.convert('RGB'), (thumb_w, thumb_h))
            contact.paste(image, ((index%4)*thumb_w, (index//4)*thumb_h))
    contact.save(FIGURES/'全部场景索引.png')
    verification = dict(groups=len(groups), datasets=dict(Counter(g['dataset'] for g in groups)),
                        metric_rows=len(rows), figures=len(outputs), png_dpi=450,
                        ratio_axis='linear_actual_ratio', ratio_positions=RATIOS,
                        input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    (ROOT/'PLOT_VERIFICATION.json').write_text(json.dumps(verification, indent=2)+'\n', encoding='utf-8')
    archive = ROOT/'ALL_SCENE_RESULTS_ORIGINAL_RGB.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as packed:
        for path in sorted(FIGURES.rglob('*')):
            if path.is_file():
                packed.write(path, path.relative_to(ROOT))
        for name in ['ALL_SCENE_METRICS.csv', 'README_结果说明.md', 'FIGURES_INDEX.json',
                     'PLOT_VERIFICATION.json', 'plot_results.py', 'collect_results.py',
                     'data/ALL_SCENE_RESULTS.json', 'data/ORIGINAL_RGB_REEVALUATION.json',
                     'REEVALUATION_COMPLETE.json', 'EVALUATION_PROTOCOL_UPDATE.md',
                     'EVALUATION_PROTOCOL_CHANGES.csv', 'data/METRIC_SOURCE_COMPARISON.json']:
            if (ROOT/name).exists():
                packed.write(ROOT/name, name)
    print('DELIVERY_COMPLETE', json.dumps(verification), 'ZIP_BYTES', archive.stat().st_size, flush=True)


if __name__ == '__main__':
    main()

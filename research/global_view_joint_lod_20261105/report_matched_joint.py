"""生成新联合模型三种推理方式的中文报告、逐视角曲线与配对统计。"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE/'matched_visible_joint_results'
FIGURES = ROOT/'figures'
FIGURES.mkdir(exist_ok=True)
REFERENCE = HERE.parents[2]/'mycartgs/research/fr1xyz_viewer_budget_comparison_20261105/results/reports/VANILLA_ALL_VIEWS_WITH_PSNR.csv'
RATIOS = [.1, .2, .3, .4, .5]
COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']
LABELS = {'global': '全局 ratio 导出', 'view_ratio': '单视角 ratio：名义 Anchor 视锥候选池',
          'view_matched': '单视角预算＝全局该帧实际 Alpha 贡献点数'}
assert json.loads((ROOT/'STATUS.json').read_text())['state'] == 'complete'
assert json.loads((ROOT/'INPUTS_BEFORE.json').read_text()) == json.loads((ROOT/'INPUTS_AFTER.json').read_text())
records = list(csv.DictReader((ROOT/'PER_VIEW.csv').open(encoding='utf-8')))
assert len(records) == 12672
by = defaultdict(list)
for row in records:
    row['source_ratio'] = float(row['source_ratio']) if row['source_ratio'] else None
    for key in ['frame', 'N', 'candidate_N', 'eligible_N', 'requested_K', 'K', 'projected_K', 'contributing_K',
                'budget_shortfall', 'global_reference_contributing_K']:
        row[key] = int(row[key])
    for key in ['psnr', 'global_psnr', 'delta_psnr_to_global', 'photo_l1']:
        row[key] = float(row[key])
    row['network_ratio'] = float(row['network_ratio']) if row['network_ratio'] else None
    assert 0 <= row['contributing_K'] <= row['projected_K'] <= row['K']
    if row['method'] == 'view_matched':
        assert row['requested_K'] == row['global_reference_contributing_K']
        assert row['K'] == min(row['requested_K'], row['eligible_N'])
        assert row['budget_shortfall'] == row['requested_K']-row['K']
    by[(row['method'], row['source_ratio'])].append(row)
for group in by.values():
    group.sort(key=lambda r: r['frame'])
    assert len(group) == 792 and [r['frame'] for r in group] == list(range(792))

reference = [r for r in csv.DictReader(REFERENCE.open(encoding='utf-8')) if r['run'] == 'rgbd_repeat1']
ref_by_image = {Path(r['rgb_path']).name: r for r in reference}
base = by[('full', None)]
reference = [ref_by_image[Path(r['image']).name] for r in base]
x = np.array([int(r['rgb_index']) for r in reference])
distance = x[:, None]-x[None, :]
weights = np.exp(-.5*(distance/30.)**2)
weights[np.abs(distance) > 90] = 0.
weights /= weights.sum(1, keepdims=True)
assert np.max(abs(weights.sum(1)-1)) < 1e-12
font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})


def values(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def describe(array):
    return {key: float(value) for key, value in [('mean', array.mean()), ('median', np.median(array)),
        ('p10', np.percentile(array, 10)), ('p90', np.percentile(array, 90)), ('min', array.min()), ('max', array.max())]}


def finish(fig, name):
    for extension in ['png', 'pdf']:
        fig.savefig(FIGURES/(name+'.'+extension), dpi=200, facecolor='white')
    plt.close(fig)


summary = []
for (method, ratio), group in by.items():
    for partition in ['all', 'bank', 'nonbank']:
        selected = [r for r in group if partition == 'all' or (r['bank'] == 'True') == (partition == 'bank')]
        result = {'method': method, 'source_ratio': ratio, 'partition': partition, 'frames': len(selected)}
        for key in ['psnr', 'delta_psnr_to_global', 'K', 'projected_K', 'contributing_K', 'requested_K', 'candidate_N', 'eligible_N']:
            result.update({key+'_'+k: v for k, v in describe(values(selected, key)).items()})
        result.update(improved_fraction=float(np.mean(values(selected, 'delta_psnr_to_global') > 0)),
            budget_shortfall_frames=sum(r['budget_shortfall'] > 0 for r in selected),
            exact_target_budget_fraction=float(np.mean(values(selected, 'K') == values(selected, 'requested_K'))),
            exact_global_contribution_fraction=float(np.mean(values(selected, 'contributing_K') == values(selected, 'global_reference_contributing_K'))),
            contribution_gap_mean=float(np.mean(values(selected, 'contributing_K')-values(selected, 'global_reference_contributing_K'))))
        if ratio is not None:
            effective = values(selected, 'network_ratio')
            result.update({f'network_ratio_{k}': v for k, v in describe(effective).items()})
            result['outside_training_ratio_frames'] = int(((effective < .01-1e-8) | (effective > .7+1e-8)).sum())
        summary.append(result)
(ROOT/'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
fields = list(dict.fromkeys(k for r in summary for k in r))
with (ROOT/'SUMMARY.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
index = {(r['method'], r['source_ratio'], r['partition']): r for r in summary}
raw_reference = np.array([float(r['psnr_db']) for r in reference])
reference_count = np.array([int(r['contributing_gaussians']) for r in reference])
reference_projected = np.array([int(r['projected_gaussians']) for r in reference])
display = []
all_psnr = [weights@values(g, 'psnr') for g in by.values()]+[weights@raw_reference]
limits = (np.floor(min(v.min() for v in all_psnr)*2)/2-.2, np.ceil(max(v.max() for v in all_psnr)*2)/2+.2)

for method, title in LABELS.items():
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(left=.085, right=.985, top=.84, bottom=.12, hspace=.48)
    for ratio, color in zip(RATIOS, COLORS):
        group = by[(method, ratio)]; quality = values(group, 'psnr'); count = values(group, 'contributing_K')
        filtered = weights@quality
        axes[0].plot(x, count/1000, color=color, lw=1.4, label=f'{ratio:g}｜贡献均值 {count.mean():,.0f} 点')
        axes[0].plot(x, values(group, 'projected_K')/1000, color=color, lw=.9, ls='--', alpha=.6)
        if method == 'view_matched':
            axes[0].plot(x, values(group, 'requested_K')/1000, color=color, lw=.8, ls=':', alpha=.8)
        axes[1].plot(x, filtered, color=color, lw=1.7, label=f'{ratio:g}｜原始均值 {quality.mean():.3f} dB')
        display.extend({'method': method, 'source_ratio': ratio, 'frame': r['frame'], 'rgb_index': int(at),
                        'psnr_raw': float(raw), 'psnr_smoothed': float(smooth)}
                       for r, at, raw, smooth in zip(group, x, quality, filtered))
    axes[0].plot(x, reference_count/1000, color='#202020', lw=2.1, label=f'CaRtGS原版｜贡献均值 {reference_count.mean():,.0f} 点')
    axes[0].plot(x, reference_projected/1000, color='#707070', lw=.9, ls='--')
    for total, label in [(35301, 'CaRtGS整图 N=35,301'), (13298, '联合模型整图 N=13,298')]:
        axes[0].axhline(total/1000, color='#999999', ls='--', lw=1.)
        axes[0].text(x[-1], total/1000+.25, label, color='#777777', ha='right', fontsize=9)
    axes[1].plot(x, weights@raw_reference, color='#202020', lw=2.2, label=f'CaRtGS原版｜原始均值 {raw_reference.mean():.3f} dB')
    axes[0].set(ylabel='Gaussian数量（千）', ylim=(0, 38))
    axes[1].set(ylabel='平滑后PSNR（dB）', xlabel='数据集RGB帧序号', ylim=limits)
    for ax in axes:
        ax.grid(color='#e5e7eb', alpha=.8); ax.set_xlim(x[0], x[-1])
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.035 if ax is axes[0] else 1.29), ncol=3, frameon=False, fontsize=9)
    fig.suptitle('fr1_xyz RGB-D · 新联合模型 · 792共同视角\n'+title, fontsize=16, fontweight='bold', y=.985)
    fig.text(.5, .885, '同色为同一来源ratio；上排实线＝实际Alpha贡献，虚线＝投影有效；下排图例均值来自未平滑PSNR。', ha='center', fontsize=10)
    note = '匹配组标签为来源全局ratio，网络实际ratio=B/N随视角变化；点线为目标预算，最终贡献数可能更低。' if method == 'view_matched' else '全局选择后补偿的数据包固定跨视角；局部预算使用原名义Anchor视锥及Slot资格。'
    fig.text(.5, .035, 'PSNR高斯平滑：σ=30个RGB帧，半窗90帧，边界归一化；点数不平滑。\n'+note, ha='center', fontsize=9)
    finish(fig, method.upper()+'_COUNTS_PSNR')

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
fig.subplots_adjust(left=.08, right=.985, top=.83, bottom=.1, hspace=.28)
for ratio, color in zip(RATIOS, COLORS):
    global_rows, matched = by[('global', ratio)], by[('view_matched', ratio)]
    global_psnr, matched_psnr = values(global_rows, 'psnr'), values(matched, 'psnr')
    axes[0].plot(x, values(global_rows, 'contributing_K')/1000, color=color, lw=1.2)
    axes[0].plot(x, values(matched, 'contributing_K')/1000, color=color, lw=1.2, ls='--')
    axes[1].plot(x, weights@global_psnr, color=color, lw=1.5)
    axes[1].plot(x, weights@matched_psnr, color=color, lw=1.5, ls='--')
    axes[2].plot(x, weights@(matched_psnr-global_psnr), color=color, lw=1.6,
                 label=f'{ratio:g}｜配对Δ均值 {(matched_psnr-global_psnr).mean():+.3f} dB')
axes[0].set_ylabel('实际Alpha贡献点数（千）'); axes[1].set_ylabel('平滑PSNR（dB）')
axes[2].set(ylabel='匹配局部 − 全局（dB）', xlabel='数据集RGB帧序号')
axes[2].axhline(0, color='#475569', lw=1.)
for ax in axes:
    ax.grid(color='#e5e7eb'); ax.set_xlim(x[0], x[-1])
handles, labels = axes[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .915), ncol=3, fontsize=9, frameon=False)
fig.suptitle('新联合模型：全局ratio 与 逐视角匹配贡献预算的局部推理\n实线＝全局；虚线＝局部匹配预算；同色为来源ratio', fontsize=15, y=.98)
fig.text(.5, .025, '预算B取全局实际Alpha贡献数，局部提交K=min(B,E)；最终实际贡献数并未强行凑齐，不能称严格同贡献数比较。\nPSNR平滑σ=30、半窗90；图例Δ均值和报告统计均使用未平滑原值。', ha='center', fontsize=10)
finish(fig, 'GLOBAL_VS_MATCHED_PAIRED')
with (ROOT/'PSNR_DISPLAY_SMOOTHED.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(display[0])); writer.writeheader(); writer.writerows(display)

lines = ['# 新联合模型：全局与单视角逐帧预算比较', '',
    '仅评价新联合 fr1_xyz RGB-D checkpoint `runs/full_fine_joint/23560_shutdown/ply`；不重新训练、不重新渲染原基线。', '',
    '模型13,298个Gaussian、2,790个Anchor，792共同RGB帧。三方法五档ratio，共11,880条LoD记录，另有792条Full参考。原版CaRtGS黑线只复用历史RGB-D repeat1数值，按RGB文件名对齐。', '',
    '## 口径', '',
    '- 全局：全图ratio选择和补偿一次、固定数据包跨视角；五个FP32导出包见 `exports/`。',
    '- 单视角ratio：Viewer原名义Anchor AABB视锥候选N，目标floor(rN)，Slot资格可能造成不足。',
    '- 匹配贡献预算：B=全局该帧实际Alpha贡献点数，局部提交K=min(B,E)，网络条件r_eff=min(1,B/N)。图例ratio只是预算来源。',
    '- 实际Alpha贡献点数指至少一个像素通过Alpha阈值及透射率早停检查的独立Gaussian数量；投影有效数为最终已补偿点的radii>0。',
    '- B匹配不保证最终实际贡献数匹配；候选集、网络ratio及补偿结果均可能不同。该实验回答当前实际推理流程下的质量，不能完全隔离Selector排序效果。', '',
    '## 三方法PSNR与实际贡献数', '',
    '| 来源ratio | 全局PSNR | 单视角ratio PSNR | 匹配局部PSNR | 匹配−全局 | 全局贡献均值 | 同ratio局部贡献 | 匹配局部贡献 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for ratio in RATIOS:
    a, b, c = [index[(method, ratio, 'all')] for method in LABELS]
    lines.append(f'| {ratio:g} | {a["psnr_mean"]:.4f} | {b["psnr_mean"]:.4f} | {c["psnr_mean"]:.4f} | {c["delta_psnr_to_global_mean"]:+.4f} | {a["contributing_K_mean"]:,.1f} | {b["contributing_K_mean"]:,.1f} | {c["contributing_K_mean"]:,.1f} |')
lines += ['', '## 匹配预算是否兑现', '',
    '| 来源ratio | 请求预算B均值 | 提交K均值 | 最终贡献与全局差 | 最终贡献严格相同帧 | 候选不足帧 | r_eff均值／范围 | PSNR改善视角 |',
    '|---|---:|---:|---:|---:|---:|---|---:|']
for ratio in RATIOS:
    s = index[('view_matched', ratio, 'all')]
    lines.append(f'| {ratio:g} | {s["requested_K_mean"]:,.1f} | {s["K_mean"]:,.1f} | {s["contribution_gap_mean"]:+.1f} | {s["exact_global_contribution_fraction"]:.1%} | {s["budget_shortfall_frames"]}/792 | {s["network_ratio_mean"]:.4f} / [{s["network_ratio_min"]:.4f}, {s["network_ratio_max"]:.4f}] | {s["improved_fraction"]:.1%} |')
delta = np.array([index[('view_matched', r, 'all')]['delta_psnr_to_global_mean'] for r in RATIOS])
ratio_delta = np.array([index[('view_ratio', r, 'all')]['delta_psnr_to_global_mean'] for r in RATIOS])
positive = [f'{r:g}' for r, d in zip(RATIOS, delta) if d > 0]
lines += ['', '## 结果解读', '',
    f'五档等权的同ratio局部−全局PSNR均值为 {ratio_delta.mean():+.4f} dB，改为匹配贡献预算后缩小到 {delta.mean():+.4f} dB；改善档位：'+('、'.join(positive) if positive else '无')+'。各档结论以表中原始配对均值为准，不以平滑曲线判断。',
    '如果匹配局部较差，不能只归因于Selector：它可能实际使用更少的贡献点，并且网络输入ratio已从来源ratio变为B/N。全局补偿后可见的点也可能来自名义视锥外Anchor，局部候选池无法选择这些点。',
    '如果匹配局部较好，说明当前局部预算流程在该模型、轨迹和预算下有效；单场景单checkpoint不能证明跨场景稳定性。Bank/non-Bank及mean/median/p10/p90详见SUMMARY.csv。', '',
    '## 曲线', '',
    '![全局ratio](figures/GLOBAL_COUNTS_PSNR.png)', '',
    '![单视角ratio](figures/VIEW_RATIO_COUNTS_PSNR.png)', '',
    '![匹配贡献预算](figures/VIEW_MATCHED_COUNTS_PSNR.png)', '',
    '![逐帧配对比较](figures/GLOBAL_VS_MATCHED_PAIRED.png)', '',
    '## 验证', '',
    '- 冻结模型及运行Python输入前后哈希相同，旧评价模型身份与本次一致。',
    '- 前两组在首/中/末帧逐ratio复现选点身份与投影/贡献点数；PSNR归约核查误差见VALIDATION.json。',
    '- 新局部渲染在首/中/末帧逐ratio与普通Rasterizer颜色和radii逐bit相同；共15组。原结果复现30组，PSNR归约最大差1.90735e-6 dB。',
    '- 所有局部预算及贡献⊆投影⊆提交的计数关系通过，未增加训练步数、不保存渲染图片。',
    '- 点数不平滑；PSNR显示按RGB帧距高斯平滑σ30、半窗90、端点归一化。所有数值结论使用原始PSNR。', '',
    '服务器：`/home/prmi/zwj/experiments/global_view_joint_lod_20261105/evaluation/matched_visible_joint_20261105`。']
(ROOT/'MATCHED_VISIBLE_JOINT_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
manifest = {'rows': len(records), 'frames': 792, 'source_sha256': hashlib.sha256((ROOT/'PER_VIEW.csv').read_bytes()).hexdigest(),
    'reference_sha256': hashlib.sha256(REFERENCE.read_bytes()).hexdigest(), 'psnr_smoothing_sigma': 30,
    'psnr_smoothing_radius': 90, 'means_from_raw': True, 'figures': [p.name for p in FIGURES.glob('*.png')]}
(ROOT/'PLOT_VALIDATION.json').write_text(json.dumps(manifest, indent=2))
with ZipFile(HERE/'MATCHED_VISIBLE_JOINT_RESULTS.zip', 'w', ZIP_DEFLATED) as archive:
    for p in ROOT.rglob('*'):
        if p.is_file(): archive.write(p, p.relative_to(ROOT))
print(json.dumps({'five_ratio_mean_delta_psnr': float(delta.mean()), 'per_ratio_delta': delta.tolist(), 'root': str(ROOT)}, ensure_ascii=False))

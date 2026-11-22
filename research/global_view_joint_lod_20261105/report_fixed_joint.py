"""绘制新联合模型固定1k至10k预算的点数及PSNR曲线。"""
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE/'fixed_budget_joint_results'
REFERENCE = HERE.parents[2]/'mycartgs/research/fr1xyz_viewer_budget_comparison_20261105/results/reports/VANILLA_ALL_VIEWS_WITH_PSNR.csv'
BUDGETS = [1000, 2000, 3000, 4000, 5000, 10000]
COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9']
assert json.loads((ROOT/'STATUS.json').read_text())['state'] == 'complete'
assert json.loads((ROOT/'INPUTS_BEFORE.json').read_text()) == json.loads((ROOT/'INPUTS_AFTER.json').read_text())
rows = list(csv.DictReader((ROOT/'PER_VIEW.csv').open()))
assert len(rows) == 4752
groups = {}
for budget in BUDGETS:
    group = sorted([r for r in rows if int(r['budget']) == budget], key=lambda r: int(r['frame']))
    assert len(group) == 792 and [int(r['frame']) for r in group] == list(range(792))
    for row in group:
        assert 0 <= int(row['contributing_K']) <= int(row['projected_K']) <= int(row['K'])
        assert int(row['K']) == min(budget, int(row['eligible_N']))
        assert int(row['budget_shortfall']) == budget-int(row['K'])
    groups[budget] = group
reference_by_image = {Path(r['rgb_path']).name: r for r in csv.DictReader(REFERENCE.open(encoding='utf-8')) if r['run'] == 'rgbd_repeat1'}
reference = [reference_by_image[Path(r['image']).name] for r in groups[1000]]
x = np.array([int(r['rgb_index']) for r in reference])
distance = x[:, None]-x[None, :]
weights = np.exp(-.5*(distance/30.)**2); weights[abs(distance) > 90] = 0.
weights /= weights.sum(1, keepdims=True)
assert np.max(abs(weights.sum(1)-1)) < 1e-12


def values(group, key):
    return np.array([float(r[key]) for r in group])


summary = []
for budget, group in groups.items():
    for partition in ['all', 'bank', 'nonbank']:
        selected = [r for r in group if partition == 'all' or (r['bank'] == 'True') == (partition == 'bank')]
        result = dict(budget=budget, partition=partition, frames=len(selected))
        for key in ['K', 'projected_K', 'contributing_K', 'psnr', 'candidate_N', 'eligible_N', 'network_ratio']:
            data = values(selected, key)
            for label, v in [('mean', data.mean()), ('median', np.median(data)), ('p10', np.percentile(data, 10)),
                             ('p90', np.percentile(data, 90)), ('min', data.min()), ('max', data.max())]:
                result[key+'_'+label] = float(v)
        result['budget_shortfall_frames'] = sum(int(r['budget_shortfall']) > 0 for r in selected)
        result['outside_training_ratio_frames'] = sum(r['ratio_in_training_range'] == 'False' for r in selected)
        summary.append(result)
(ROOT/'SUMMARY.json').write_text(json.dumps(summary, indent=2))
with (ROOT/'SUMMARY.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
    'axes.unicode_minus': False, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.subplots_adjust(left=.085, right=.985, top=.84, bottom=.13, hspace=.75)
display = []
for budget, color in zip(BUDGETS, COLORS):
    group = groups[budget]; counts = values(group, 'contributing_K'); quality = values(group, 'psnr')
    axes[0].plot(x, counts/1000, color=color, lw=1.4, label=f'预算={budget:,}｜贡献均值 {counts.mean():,.0f} 点')
    axes[0].plot(x, values(group, 'projected_K')/1000, color=color, lw=.9, ls='--', alpha=.6)
    filtered = weights@quality
    axes[1].plot(x, filtered, color=color, lw=1.65, label=f'预算={budget:,}｜原始均值 {quality.mean():.3f} dB')
    display.extend(dict(budget=budget, frame=int(r['frame']), rgb_index=int(at), psnr_raw=float(a), psnr_smoothed=float(b))
                   for r, at, a, b in zip(group, x, quality, filtered))
ref_count = values(reference, 'contributing_gaussians'); ref_psnr = values(reference, 'psnr_db')
axes[0].plot(x, ref_count/1000, color='#202020', lw=2.1, label=f'CaRtGS原版｜均值 {ref_count.mean():,.0f} 点')
axes[0].plot(x, values(reference, 'projected_gaussians')/1000, color='#707070', lw=.9, ls='--')
axes[1].plot(x, weights@ref_psnr, color='#202020', lw=2.2, label=f'CaRtGS原版｜均值 {ref_psnr.mean():.3f} dB')
for total, text in [(35301, 'CaRtGS整图 N=35,301'), (13298, '联合模型整图 N=13,298')]:
    axes[0].axhline(total/1000, color='#999999', lw=1., ls='--')
    axes[0].text(x[-1], total/1000+.25, text, ha='right', color='#777777', fontsize=9)
axes[0].set(ylabel='Gaussian数量（千）', ylim=(0, 38))
axes[1].set(ylabel='平滑后PSNR（dB）', xlabel='数据集RGB帧序号')
for ax in axes:
    ax.grid(color='#e5e7eb'); ax.set_xlim(x[0], x[-1])
axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.035), ncol=3, fontsize=9, frameon=False)
axes[1].legend(loc='upper center', bbox_to_anchor=(.5, 1.31), ncol=3, fontsize=9, frameon=False)
fig.suptitle('fr1_xyz RGB-D · 新联合模型 · 792个共同视角\n单视角固定点数预算：名义Anchor视锥候选池', fontsize=16, fontweight='bold', y=.985)
fig.text(.5, .885, '同色对应同一预算；上排实线＝实际Alpha贡献，虚线＝投影有效；下排图例均值使用原始PSNR。', ha='center', fontsize=10)
fig.text(.5, .035, 'PSNR高斯平滑：σ=30个RGB帧，半窗90帧，边界归一化；点数不平滑。\n实际K=min(请求预算, Slot可用候选数)，网络ratio=min(1, 请求预算/N)；4k/5k部分候选不足，10k全帧候选不足且ratio外推。', ha='center', fontsize=9)
for extension in ['png', 'pdf']:
    fig.savefig(ROOT/('FIXED_BUDGET_COUNTS_PSNR.'+extension), dpi=200, facecolor='white')
plt.close(fig)
with (ROOT/'PSNR_DISPLAY_SMOOTHED.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(display[0])); writer.writeheader(); writer.writerows(display)
all_results = [r for r in summary if r['partition'] == 'all']
lines = ['# 新联合模型：单视角固定点数预算', '',
    'fr1_xyz RGB-D，792共同视角，使用新联合模型 `full_fine_joint/23560_shutdown/ply`。不进行训练、不重新渲染旧基线，不保存渲染帧图片。', '',
    '名义Anchor AABB视锥决定候选人口N，Slot可用人口E；提交K=min(B,E)，网络ratio=min(1,B/N)。实际Alpha贡献数不等于请求预算或提交K。', '',
    '| 请求预算 | 提交K均值 | 实际贡献均值 | PSNR均值(dB) | 预算不足帧 | 超训练ratio范围帧 |',
    '|---:|---:|---:|---:|---:|---:|']
for r in all_results:
    lines.append(f'| {r["budget"]:,} | {r["K_mean"]:,.1f} | {r["contributing_K_mean"]:,.1f} | {r["psnr_mean"]:.4f} | {r["budget_shortfall_frames"]}/792 | {r["outside_training_ratio_frames"]}/792 |')
lines += ['', '![单视角固定预算点数与PSNR](FIXED_BUDGET_COUNTS_PSNR.png)', '',
    'PSNR曲线按RGB帧距离高斯平滑σ30、半窗90、端点归一化；图例和表格使用未平滑均值。黑线只读复用原版CaRtGS RGB-D repeat1数值，按RGB文件名对齐。', '',
    '新模型训练ratio范围为0.01–0.7。若固定预算导致ratio超出此范围，记录为外推，不修改模型或强制截到0.7。10k候选不足时，不能把该档称作实际渲染了10,000个点。', '',
    '首/中/末帧×六预算，共18组计数版与普通Rasterizer颜色、radii逐bit一致；所有4,752条记录满足贡献数≤投影有效数≤提交K，输入哈希前后不变。逐帧数据见PER_VIEW.csv，Bank/non-Bank及分位数见SUMMARY.csv。']
(ROOT/'FIXED_BUDGET_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
(ROOT/'PLOT_VALIDATION.json').write_text(json.dumps({'rows':len(rows), 'source_sha256':hashlib.sha256((ROOT/'PER_VIEW.csv').read_bytes()).hexdigest(),
    'reference_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(), 'smoothing_sigma':30, 'smoothing_radius':90, 'means_from_raw':True}, indent=2))
with ZipFile(HERE/'FIXED_BUDGET_JOINT_RESULTS.zip', 'w', ZIP_DEFLATED) as archive:
    for p in ROOT.rglob('*'):
        if p.is_file(): archive.write(p, p.relative_to(ROOT))
print(json.dumps([{key:r[key] for key in ['budget', 'K_mean', 'contributing_K_mean', 'psnr_mean',
    'budget_shortfall_frames', 'outside_training_ratio_frames']} for r in all_results], ensure_ascii=False))

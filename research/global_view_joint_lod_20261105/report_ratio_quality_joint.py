"""汇总训练/未见ratio质量；横轴为真实ratio，纵轴为全轨迹平均PSNR。"""
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
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE/'ratio_quality_joint_results'
TRAIN = [.01, .05, .1, .3, .5, .7]
UNSEEN = [.03, .08, .2, .4, .6]
RATIOS = sorted(TRAIN+UNSEEN)
LABELS = {'global': '全局 ratio 导出', 'view': '单视角 ratio：名义 Anchor 视锥预算'}
COLORS = {'global': '#0072B2', 'view': '#D55E00'}
state = json.loads((ROOT/'STATUS.json').read_text())
assert state['state'] == 'complete' and state['rows'] == 18216
assert json.loads((ROOT/'INPUTS_BEFORE.json').read_text()) == json.loads((ROOT/'INPUTS_AFTER.json').read_text())
rows = list(csv.DictReader((ROOT/'PER_VIEW.csv').open()))
assert len(rows) == 18216
groups = defaultdict(list)
for row in rows:
    ratio = float(row['ratio']) if row['ratio'] else None
    expected_split = 'full' if ratio is None else 'train' if ratio in TRAIN else 'unseen'
    assert row['group'] == 'joint' and row['ratio_split'] == expected_split
    assert 0 <= int(row['contributing_K']) <= int(row['projected_K']) <= int(row['K'])
    groups[(row['scope'], ratio)].append(row)
reference_keys = None
for group in groups.values():
    group.sort(key=lambda r: int(r['frame']))
    assert len(group) == 792 and [int(r['frame']) for r in group] == list(range(792))
    keys = [(r['image'], r['timestamp']) for r in group]
    if reference_keys is None:
        reference_keys = keys
    assert keys == reference_keys
summary = []
for (scope, ratio), group in groups.items():
    for partition in ['all', 'bank', 'nonbank']:
        selected = [r for r in group if partition == 'all' or (r['bank'] == 'True') == (partition == 'bank')]
        assert len(selected) == {'all':792, 'bank':72, 'nonbank':720}[partition]
        result = dict(scope=scope, ratio=ratio, ratio_split=group[0]['ratio_split'], partition=partition, frames=len(selected))
        for key in ['psnr', 'ssim', 'lpips', 'photo_l1', 'K', 'candidate_N', 'projected_K', 'contributing_K']:
            values = np.array([float(r[key]) for r in selected])
            assert np.isfinite(values).all()
            for label, value in [('mean',values.mean()), ('median',np.median(values)),
                                 ('p10',np.percentile(values,10)), ('p90',np.percentile(values,90))]:
                result[key+'_'+label] = float(value)
        result['budget_shortfall_frames'] = sum(int(r['budget_shortfall']) > 0 for r in selected)
        summary.append(result)
(ROOT/'SUMMARY.json').write_text(json.dumps(summary, indent=2))
with (ROOT/'SUMMARY.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
index = {(r['scope'], r['ratio'], r['partition']):r for r in summary}
full = index[('full', None, 'all')]
table = [dict(ratio=r, split='训练' if r in TRAIN else '未见',
    global_psnr=index[('global',r,'all')]['psnr_mean'], view_psnr=index[('view',r,'all')]['psnr_mean'],
    global_ssim=index[('global',r,'all')]['ssim_mean'], view_ssim=index[('view',r,'all')]['ssim_mean'],
    global_lpips=index[('global',r,'all')]['lpips_mean'], view_lpips=index[('view',r,'all')]['lpips_mean']) for r in RATIOS]
with (ROOT/'RATIO_QUALITY.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
    'axes.unicode_minus':False, 'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
fig, ax = plt.subplots(figsize=(15, 7))
fig.subplots_adjust(left=.065, right=.985, top=.79, bottom=.19)
for scope, label in LABELS.items():
    quality = np.array([index[(scope,r,'all')]['psnr_mean'] for r in RATIOS])
    ax.plot(RATIOS, quality, color=COLORS[scope], lw=2., label=label)
    for chosen, filled in [(TRAIN, True), (UNSEEN, False)]:
        indices = [i for i,r in enumerate(RATIOS) if r in chosen]
        ax.scatter(np.array(RATIOS)[indices], quality[indices], s=58,
                   edgecolor=COLORS[scope], facecolor=COLORS[scope] if filled else 'white', lw=1.8, zorder=4)
    for r, q in zip(RATIOS, quality):
        ax.annotate(f'{q:.2f}', (r,q), xytext=(0,10 if scope=='global' else -17),
                    textcoords='offset points', ha='center', color=COLORS[scope], fontsize=9)
ax.axhline(full['psnr_mean'], color='#64748B', ls='--', lw=1.5, label=f'未补偿 Full：{full["psnr_mean"]:.3f} dB')
ax.set(xlabel='Ratio', ylabel='全轨迹平均 PSNR（dB，越高越好）', xlim=(-.005,.72),
       ylim=(min(r['view_psnr'] for r in table)-.4,full['psnr_mean']+.45),
       xticks=RATIOS, xticklabels=[f'{r:g}' for r in RATIOS])
ax.tick_params(axis='x', labelrotation=40)
ax.grid(color='#E2E8F0')
handles, labels = ax.get_legend_handles_labels()
handles += [Line2D([],[],ls='',marker='o',color='#475569',label='训练 ratio：实心'),
            Line2D([],[],ls='',marker='o',color='#475569',markerfacecolor='white',label='未见 ratio：空心')]
labels += ['训练 ratio：实心','未见 ratio：空心']
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.925),ncol=3,frameon=False,fontsize=11)
fig.suptitle('fr1_xyz RGB-D · 新联合模型 · 792共同视角\n训练与未见 ratio 的渲染质量', fontsize=17, fontweight='bold', y=.985)
fig.text(.5,.035,'横轴为真实ratio；点为逐帧PSNR的原始均值，连线仅辅助观察，不做平滑。\n0.08已完整补评；0.7已参与训练，不能归为未见ratio。全局与单视角同ratio的实际点数不同。',ha='center',fontsize=10)
for extension in ['png','pdf']:
    fig.savefig(ROOT/('RATIO_PSNR_JOINT.'+extension), dpi=220, facecolor='white')
plt.close(fig)

lines = ['# fr1_xyz新联合模型：训练与未见ratio质量', '',
    '使用联合模型 `runs/full_fine_joint/23560_shutdown/ply`，13,298个Gaussian、2,790个Anchor；792共同RGB帧，Bank72、非Bank720。不新增训练、不重新评价旧基线、不保存渲染帧图片。', '',
    '训练ratio：[0.01, 0.05, 0.1, 0.3, 0.5, 0.7]。真正未见ratio：[0.03, 0.08, 0.2, 0.4, 0.6]。用户列表中的0.7同时出现在训练列表，因此只作为训练ratio展示一次，不作为未见ratio统计。', '',
    '除0.08外复用同一冻结checkpoint的既有全轨迹评价；0.08在792帧上对全局和单视角分别新增渲染，共1,584次，未用0.075近似或插值。', '',
    '全局沿用原全图ratio导出、固定补偿数据包跨视角；单视角沿用Viewer名义Anchor AABB视锥候选人口N，目标floor(rN)、Slot约束的严格点数预算，网络ratio仍为r。同ratio并不保证两种方式实际K或贡献点数相同。', '',
    '| Ratio | 类型 | 全局PSNR(dB) | 单视角PSNR(dB) | 全局SSIM | 单视角SSIM | 全局LPIPS↓ | 单视角LPIPS↓ |',
    '|---:|---|---:|---:|---:|---:|---:|---:|']
for r in table:
    lines.append(f'| {r["ratio"]:g} | {r["split"]} | {r["global_psnr"]:.4f} | {r["view_psnr"]:.4f} | {r["global_ssim"]:.5f} | {r["view_ssim"]:.5f} | {r["global_lpips"]:.5f} | {r["view_lpips"]:.5f} |')
lines += ['',f'未补偿Full：PSNR {full["psnr_mean"]:.4f} dB，SSIM {full["ssim_mean"]:.5f}，LPIPS {full["lpips_mean"]:.5f}。', '',
    '![Ratio与PSNR](RATIO_PSNR_JOINT.png)', '',
    '质量指标沿用原评价：FP32 RGB裁到[0,1]、逐帧PSNR后求均值；TorchMetrics 11×11 Gaussian SSIM，AlexNet LPIPS(normalize=True)，批大小8、TF32关闭。图不做平滑或均值重采样。', '',
    '0.08首/中/末帧×两种推理，共6组颜色/radii逐bit核对通过；全部18,216条记录覆盖相同相机与RGB文件，计数满足贡献数≤投影有效数≤实际K，输入前后SHA相同。完整分位数和Bank/non-Bank划分见SUMMARY.csv。', '',
    '服务器目录：`/home/prmi/zwj/experiments/global_view_joint_lod_20261105/evaluation/ratio_quality_joint_20261105`。']
(ROOT/'RATIO_QUALITY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(ROOT/'PLOT_VALIDATION.json').write_text(json.dumps({'source_sha256':hashlib.sha256((ROOT/'PER_VIEW.csv').read_bytes()).hexdigest(),
    'frames':792,'ratios':RATIOS,'train':TRAIN,'unseen':UNSEEN,'x_axis':'actual_ratio_linear',
    'y_axis':'mean_of_per_frame_psnr','smoothed':False,'ratio_008_measured':True,'ratio_07_is_train':True},indent=2))
with ZipFile(HERE/'RATIO_QUALITY_JOINT_RESULTS.zip','w',ZIP_DEFLATED) as archive:
    for p in ROOT.rglob('*'):
        if p.is_file():archive.write(p,p.relative_to(ROOT))
print(json.dumps(table,ensure_ascii=False))

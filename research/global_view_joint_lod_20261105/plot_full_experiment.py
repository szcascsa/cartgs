"""绘制全局／局部质量与配对差值，点数标注来自实际评价结果。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent/'full_results/evaluation/reports'
OUT = ROOT/'figures'; OUT.mkdir(exist_ok=True)
font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.facecolor': 'white'})
rows = json.loads((ROOT/'SUMMARY.json').read_text())
lookup = {(r['group'], r['scope'], r['ratio']): r for r in rows if r['partition'] == 'all'}
ratios = [.01, .03, .05, .075, .1, .2, .3, .4, .5, .6, .7]
train = [.01, .05, .1, .3, .5, .7]; x = np.arange(len(ratios))
colors = {'baseline': '#334155', 'joint': '#d55e00', 'global_control': '#009e73'}
labels = {'baseline': '原基线：Full＋全局', 'joint': '主实验：Full＋全局＋局部', 'global_control': '对照：Full＋两路全局'}


def line(ax, values, group, label=True):
    ax.plot(x, values, color=colors[group], lw=2, label=labels[group] if label else None)
    for selected, filled in [(train, True), ([r for r in ratios if r not in train], False)]:
        at = [i for i, r in enumerate(ratios) if r in selected]
        ax.scatter(x[at], np.asarray(values)[at], color=colors[group], facecolors=colors[group] if filled else 'white', s=34, zorder=3)


fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
for ax, scope, title in zip(axes, ['global', 'view'], ['全局 ratio 导出', '名义 Anchor 视锥内严格预算推理']):
    for group in labels:
        values = [lookup[(group, scope, ratio)]['psnr_mean'] for ratio in ratios]
        line(ax, values, group)
    ticks = [f'{100*r:g}%\nK≈{lookup[("joint",scope,r)]["K_mean"]:,.0f}' for r in ratios]
    ax.set(xticks=x, xticklabels=ticks, title=title, xlabel='Ratio（按档位排列）')
    ax.grid(axis='y', color='#e2e8f0'); ax.tick_params(axis='x', labelsize=9)
axes[0].set_ylabel('全轨迹平均 PSNR（dB，越高越好）')
fig.suptitle('fr1_xyz RGB-D｜Full checkpoint 继续 Fine 11,580步｜792共同RGB帧', fontsize=15, y=.99)
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
fig.text(.5, .015, 'K为联合方案实际选中点数的逐帧均值；实心点为训练ratio，空心点为插值ratio；不额外进行same-K重选。', ha='center', fontsize=10)
fig.tight_layout(rect=(.01, .05, .99, .86))
for extension in ['png', 'pdf']:
    fig.savefig(OUT/('PSNR_GLOBAL_VIEW.'+extension), dpi=220)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for row, scope in enumerate(['global', 'view']):
    for column, metric, title in [(0, 'psnr', 'ΔPSNR（dB，↑）'), (1, 'ssim', 'ΔSSIM（↑）'), (2, 'lpips', 'ΔLPIPS（↓）')]:
        ax = axes[row, column]
        for group in ['joint', 'global_control']:
            values = [lookup[(group,scope,r)][metric+'_mean']-lookup[('baseline',scope,r)][metric+'_mean'] for r in ratios]
            line(ax, values, group)
        ax.axhline(0, color='#334155', lw=1)
        if row == 0 and column == 0:
            ax.axhline(-.2, color='#b91c1c', ls='--', lw=1)
        ax.set(xticks=x, xticklabels=[f'{100*r:g}%' for r in ratios], title=('全局' if row == 0 else '局部')+' '+title)
        ax.grid(axis='y', color='#e2e8f0'); ax.tick_params(axis='x', labelsize=8)
fig.suptitle('相对匹配基线的质量变化｜相同Full起点、相机序列和Adam步数', fontsize=15, y=.99)
handles, names = axes[0,0].get_legend_handles_labels()
fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.5,.95), ncol=2, frameon=False)
fig.text(.5,.015,'全局PSNR红虚线为训练档预设−0.20 dB门槛，20%插值略低于此参考线；单场景单seed，曲线不表示置信区间。',ha='center',fontsize=10)
fig.tight_layout(rect=(.01,.04,.99,.89))
for extension in ['png','pdf']:
    fig.savefig(OUT/('METRIC_DELTAS.'+extension),dpi=220)
plt.close(fig)
print(OUT)

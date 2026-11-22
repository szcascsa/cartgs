"""用原始均值绘制LI消融；逐帧诊断曲线仅对PSNR显示作平滑。"""
import csv
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/'results'
REPORTS=ROOT/'evaluation/reports'
FIGURES=REPORTS/'figures';FIGURES.mkdir(exist_ok=True)
TRAIN=[.01,.05,.1,.3,.5,.7]
RATIOS=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]
summary=json.loads((REPORTS/'SUMMARY.json').read_text())
index={(r['group'],r['case']):r for r in summary if r['partition']=='all'}
paired=json.loads((REPORTS/'PAIRED_SUMMARY.json').read_text())
pair_index={r['case']:r for r in paired if r['partition']=='all'}
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
    'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
colors={'no_li':'#334155','li01':'#D55E00'}
names={'no_li':'联合方案：无LI','li01':'联合方案：LI=0.1'}


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(FIGURES/(name+'.'+ext),dpi=220,facecolor='white')
    plt.close(fig)


fig,axes=plt.subplots(1,2,figsize=(16,6),sharey=True)
for ax,scope,title in zip(axes,['global','view'],['全局 ratio 导出','名义Anchor视锥：单视角ratio预算']):
    for group in names:
        values=[index[(group,f'{scope}_{r:g}')]['psnr_mean'] for r in RATIOS]
        ax.plot(RATIOS,values,color=colors[group],lw=2,label=names[group])
        for filled in [True,False]:
            at=[i for i,r in enumerate(RATIOS) if (r in TRAIN)==filled]
            ax.scatter(np.array(RATIOS)[at],np.array(values)[at],edgecolor=colors[group],
                       facecolor=colors[group] if filled else 'white',s=36,zorder=3)
        full=index[(group,'full')]['psnr_mean']
        ax.axhline(full,color=colors[group],lw=1,ls='--',alpha=.6)
    ax.set(title=title,xlabel='Ratio',xticks=RATIOS,xticklabels=[f'{r:g}' for r in RATIOS])
    ax.tick_params(axis='x',labelrotation=45,labelsize=9);ax.grid(color='#E2E8F0')
axes[0].set_ylabel('792帧平均PSNR（dB）')
handles,labels=axes[0].get_legend_handles_labels()
handles += [Line2D([],[],marker='o',color='#475569',ls=''),Line2D([],[],marker='o',color='#475569',markerfacecolor='white',ls='')]
labels += ['实心＝训练ratio','空心＝未见ratio']
fig.suptitle('fr1_xyz RGB-D｜局部LI配对消融｜相同Full起点、11,580步、一次Adam',fontsize=15,y=.99)
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=4,frameon=False)
fig.text(.5,.015,'横轴为真实ratio，均值不平滑；水平虚线为各自Full PSNR。相同ratio不意味着全局与局部点数相同。',ha='center')
fig.tight_layout(rect=(.01,.055,.99,.86));save(fig,'LI_RATIO_PSNR')

fig,axes=plt.subplots(2,3,figsize=(16,8))
for row,scope in enumerate(['global','view']):
    for col,key,label in [(0,'psnr','ΔPSNR（dB，↑）'),(1,'ssim','ΔSSIM（↑）'),(2,'lpips','ΔLPIPS（↓）')]:
        ax=axes[row,col];values=[pair_index[f'{scope}_{r:g}'][f'delta_{key}_mean'] for r in RATIOS]
        ax.plot(RATIOS,values,color='#D55E00',lw=1.8,marker='o',ms=4)
        ax.axhline(0,color='#475569',lw=1);ax.grid(color='#E2E8F0')
        ax.set(title=('全局' if row==0 else '局部')+' '+label,xticks=RATIOS,xticklabels=[f'{r:g}' for r in RATIOS])
        ax.tick_params(axis='x',labelrotation=45,labelsize=8)
fig.suptitle('LI=0.1 − 无LI｜相同RGB帧的配对质量变化',fontsize=15)
fig.text(.5,.015,'所有数值来自逐帧原始指标；单场景单seed，未按non-Bank指标挑选终点或权重。',ha='center')
fig.tight_layout(rect=(0,.05,1,.95));save(fig,'LI_METRIC_DELTAS')

fig,axes=plt.subplots(1,2,figsize=(14,5.5))
for group in names:
    budgets=[1000,2000,3000,4000,5000,10000]
    axes[0].plot(budgets,[index[(group,f'points_{b}')]['psnr_mean'] for b in budgets],color=colors[group],marker='o',label=names[group])
    axes[1].plot([index[(group,f'points_{b}')]['contributing_K_mean'] for b in budgets],
                 [index[(group,f'points_{b}')]['psnr_mean'] for b in budgets],color=colors[group],marker='o',label=names[group])
axes[0].set(xlabel='请求预算B',ylabel='平均PSNR（dB）',title='固定点数预算',xticks=budgets)
axes[1].set(xlabel='实际Alpha贡献点数均值',ylabel='平均PSNR（dB）',title='实际贡献数量与质量')
for ax in axes:ax.grid(color='#E2E8F0');ax.legend(frameon=False)
fig.suptitle('fr1_xyz RGB-D｜单视角固定点数：无LI vs LI=0.1',fontsize=14)
fig.text(.5,.015,'严格提交K=min(B,E)，网络ratio=min(1,B/N)；预算不足和超训练范围不得视为实际达到请求点数。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.94));save(fig,'LI_FIXED_BUDGET_PSNR')

fig,axes=plt.subplots(1,2,figsize=(14,5.5),sharey=True)
ratios=[.1,.2,.3,.4,.5]
for ax,group in zip(axes,names):
    for scope,label,style in [('global','全局导出','-'),('matched','局部预算＝该帧全局贡献数','--')]:
        ax.plot(ratios,[index[(group,f'{scope}_{r:g}')]['psnr_mean'] for r in ratios],style,marker='o',label=label)
    ax.set(title=names[group],xlabel='来源全局ratio',xticks=ratios);ax.grid(color='#E2E8F0');ax.legend(frameon=False)
axes[0].set_ylabel('平均PSNR（dB）')
fig.suptitle('匹配逐帧全局贡献预算后的局部质量',fontsize=14)
fig.text(.5,.015,'匹配提交预算不保证最终Alpha贡献数量严格相同；局部网络ratio按B/N计算。',ha='center')
fig.tight_layout(rect=(0,.06,1,.94));save(fig,'LI_GLOBAL_MATCHED_PSNR')
with ZipFile(HERE/'LOCAL_IMPORTANCE_LI_RESULTS.zip','w',ZIP_DEFLATED) as archive:
    for p in ROOT.rglob('*'):
        if p.is_file():archive.write(p,p.relative_to(ROOT))
print(json.dumps({'figures':len(list(FIGURES.glob('*.png'))),'quality_gates':json.loads((REPORTS/'QUALITY_GATES.json').read_text())},ensure_ascii=False))

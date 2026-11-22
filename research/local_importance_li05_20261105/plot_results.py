"""在Windows绘制三权重原始均值曲线；不平滑ratio均值。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parent/'results'
REPORTS=ROOT/'evaluation/reports'
FIGURES=REPORTS/'figures';FIGURES.mkdir(exist_ok=True)
RATIOS=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]
TRAIN=[.01,.05,.1,.3,.5,.7]
summary=json.loads((REPORTS/'SUMMARY.json').read_text())
index={(r['group'],r['case']):r for r in summary if r['partition']=='all'}
paired=json.loads((REPORTS/'PAIRED_SUMMARY.json').read_text())
pair_index={(r['reference'],r['group'],r['case']):r for r in paired if r['partition']=='all'}
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
    'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
COLORS={'no_li':'#475569','li01':'#0072B2','li05':'#D55E00'}
NAMES={'no_li':'无LI','li01':'LI=0.1','li05':'LI=0.5'}


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(FIGURES/(name+'.'+ext),dpi=200,facecolor='white')
    plt.close(fig)


fig,axes=plt.subplots(1,2,figsize=(16,6),sharey=True)
for ax,scope,title in zip(axes,['global','view'],['全局ratio导出','单视角：名义Anchor视锥预算']):
    for group in NAMES:
        values=[index[(group,f'{scope}_{r:g}')]['psnr_mean'] for r in RATIOS]
        ax.plot(RATIOS,values,color=COLORS[group],lw=2,label=NAMES[group])
        for filled in [True,False]:
            at=[i for i,r in enumerate(RATIOS) if (r in TRAIN)==filled]
            ax.scatter(np.array(RATIOS)[at],np.array(values)[at],edgecolor=COLORS[group],
                facecolor=COLORS[group] if filled else 'white',s=35,zorder=3)
        ax.axhline(index[(group,'full')]['psnr_mean'],color=COLORS[group],lw=1,ls='--',alpha=.6)
    ax.set(title=title,xlabel='Ratio',xticks=RATIOS,xticklabels=[f'{r:g}' for r in RATIOS])
    ax.tick_params(axis='x',labelrotation=45,labelsize=9);ax.grid(color='#E2E8F0')
axes[0].set_ylabel('792帧平均PSNR（dB）')
handles,labels=axes[0].get_legend_handles_labels()
handles.extend([Line2D([],[],marker='o',color='#475569',ls=''),Line2D([],[],marker='o',color='#475569',markerfacecolor='white',ls='')])
labels.extend(['实心：训练ratio','空心：未见ratio'])
fig.suptitle('fr1_xyz RGB-D｜LI权重对比｜同Full起点、同相机、11,580步',fontsize=15,y=.99)
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=5,frameon=False)
fig.text(.5,.015,'真实ratio横轴，均值不平滑；虚线为各自Full。相同ratio不代表全局与局部点数相同。',ha='center')
fig.tight_layout(rect=(.01,.055,.99,.86));save(fig,'LI_WEIGHT_RATIO_PSNR')

fig,axes=plt.subplots(2,3,figsize=(16,8))
for row,scope in enumerate(['global','view']):
    for col,key,label in [(0,'psnr','ΔPSNR（dB，↑）'),(1,'ssim','ΔSSIM（↑）'),(2,'lpips','ΔLPIPS（↓）')]:
        ax=axes[row,col]
        for reference,group,color,style in [('no_li','li01','#0072B2',':'),('no_li','li05','#D55E00','-'),('li01','li05','#009E73','--')]:
            values=[pair_index[(reference,group,f'{scope}_{r:g}')][f'delta_{key}_mean'] for r in RATIOS]
            ax.plot(RATIOS,values,color=color,ls=style,lw=1.8,marker='o',ms=3,label=NAMES[group]+' − '+NAMES[reference])
        ax.axhline(0,color='#475569',lw=1);ax.grid(color='#E2E8F0')
        ax.set(title=('全局' if row==0 else '局部')+' '+label,xticks=RATIOS,xticklabels=[f'{r:g}' for r in RATIOS])
        ax.tick_params(axis='x',labelrotation=45,labelsize=8)
fig.suptitle('三权重配对变化｜相同792帧，逐帧指标差值再平均',fontsize=15,y=.995)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.96),ncol=3,frameon=False)
fig.text(.5,.015,'单场景单seed；同seed训练不保证多步逐bit可重复，不按non-Bank结果选择checkpoint。',ha='center')
fig.tight_layout(rect=(0,.05,1,.90));save(fig,'LI_WEIGHT_METRIC_DELTAS')

fig,axes=plt.subplots(1,2,figsize=(14,5.5))
budgets=[1000,2000,3000,4000,5000,10000]
for group in NAMES:
    q=[index[(group,f'points_{b}')]['psnr_mean'] for b in budgets]
    axes[0].plot(budgets,q,color=COLORS[group],marker='o',label=NAMES[group])
    axes[1].plot([index[(group,f'points_{b}')]['contributing_K_mean'] for b in budgets],q,color=COLORS[group],marker='o',label=NAMES[group])
axes[0].set(xlabel='请求预算B',ylabel='平均PSNR（dB）',title='固定点数预算',xticks=budgets)
axes[1].set(xlabel='实际Alpha贡献点数均值',ylabel='平均PSNR（dB）',title='实际贡献数量与质量')
for ax in axes:ax.grid(color='#E2E8F0');ax.legend(frameon=False)
fig.suptitle('fr1_xyz RGB-D｜局部固定预算：LI=0 / 0.1 / 0.5',fontsize=14)
fig.text(.5,.015,'实际K=min(B,E)，网络ratio=min(1,B/N)；保留候选不足和超训练ratio范围的原Viewer口径。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.94));save(fig,'LI_WEIGHT_FIXED_BUDGET_PSNR')

fig,axes=plt.subplots(1,3,figsize=(17,5),sharey=True)
ratios=[.1,.2,.3,.4,.5]
for ax,group in zip(axes,NAMES):
    for scope,label,style in [('global','全局导出','-'),('matched','局部预算＝本组全局贡献数','--')]:
        ax.plot(ratios,[index[(group,f'{scope}_{r:g}')]['psnr_mean'] for r in ratios],style,marker='o',label=label)
    ax.set(title=NAMES[group],xlabel='来源全局ratio',xticks=ratios);ax.grid(color='#E2E8F0');ax.legend(frameon=False,fontsize=9)
axes[0].set_ylabel('平均PSNR（dB）')
fig.suptitle('各自逐帧全局贡献数量匹配预算下的局部质量',fontsize=14)
fig.text(.5,.015,'使用各组自身全局贡献数量，不是三组same-K，也不保证最后Alpha贡献数量严格相同。',ha='center')
fig.tight_layout(rect=(0,.06,1,.94));save(fig,'LI_WEIGHT_GLOBAL_MATCHED_PSNR')
print('FIGURES_COMPLETE',len(list(FIGURES.glob('*.png'))))

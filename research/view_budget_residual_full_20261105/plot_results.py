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
COLORS={'baseline':'#475569','view_no_li':'#0072B2','view_li01':'#D55E00'}
NAMES={'baseline':'原联合无LI','view_no_li':'视角残差无LI','view_li01':'视角残差＋LI=0.1'}


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
fig.suptitle('fr1_xyz RGB-D｜视角残差对比｜同Full起点、同相机、11,580步',fontsize=15,y=.99)
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=5,frameon=False)
fig.text(.5,.015,'真实ratio横轴，均值不平滑；虚线为各自Full。相同ratio不代表全局与局部点数相同。',ha='center')
fig.tight_layout(rect=(.01,.055,.99,.86));save(fig,'VIEW_RESIDUAL_RATIO_PSNR')

fig,axes=plt.subplots(2,3,figsize=(16,8))
for row,scope in enumerate(['global','view']):
    for col,key,label in [(0,'psnr','ΔPSNR（dB，↑）'),(1,'ssim','ΔSSIM（↑）'),(2,'lpips','ΔLPIPS（↓）')]:
        ax=axes[row,col]
        for reference,group,color,style in [('baseline','view_no_li','#0072B2',':'),('baseline','view_li01','#D55E00','-'),('view_no_li','view_li01','#009E73','--')]:
            values=[pair_index[(reference,group,f'{scope}_{r:g}')][f'delta_{key}_mean'] for r in RATIOS]
            ax.plot(RATIOS,values,color=color,ls=style,lw=1.8,marker='o',ms=3,label=NAMES[group]+' − '+NAMES[reference])
        ax.axhline(0,color='#475569',lw=1);ax.grid(color='#E2E8F0')
        ax.set(title=('全局' if row==0 else '局部')+' '+label,xticks=RATIOS,xticklabels=[f'{r:g}' for r in RATIOS])
        ax.tick_params(axis='x',labelrotation=45,labelsize=8)
fig.suptitle('视角残差配对变化｜相同792帧，逐帧指标差值再平均',fontsize=15,y=.995)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.96),ncol=3,frameon=False)
fig.text(.5,.015,'单场景单seed；同seed训练不保证多步逐bit可重复，不按non-Bank结果选择checkpoint。',ha='center')
fig.tight_layout(rect=(0,.05,1,.90));save(fig,'VIEW_RESIDUAL_METRIC_DELTAS')

fig,axes=plt.subplots(1,2,figsize=(14,5.5))
budgets=[1000,2000,3000,4000,5000,10000]
for group in NAMES:
    q=[index[(group,f'points_{b}')]['psnr_mean'] for b in budgets]
    axes[0].plot(budgets,q,color=COLORS[group],marker='o',label=NAMES[group])
    axes[1].plot([index[(group,f'points_{b}')]['contributing_K_mean'] for b in budgets],q,color=COLORS[group],marker='o',label=NAMES[group])
axes[0].set(xlabel='请求预算B',ylabel='平均PSNR（dB）',title='固定点数预算',xticks=budgets)
axes[1].set(xlabel='实际Alpha贡献点数均值',ylabel='平均PSNR（dB）',title='实际贡献数量与质量')
for ax in axes:ax.grid(color='#E2E8F0');ax.legend(frameon=False)
fig.suptitle('fr1_xyz RGB-D｜局部固定预算：原联合／视角残差／视角残差＋LI',fontsize=14)
fig.text(.5,.015,'实际K=min(B,E)，网络ratio=min(1,B/N)；保留候选不足和超训练ratio范围的原Viewer口径。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.94));save(fig,'VIEW_RESIDUAL_FIXED_BUDGET_PSNR')

fig,axes=plt.subplots(1,3,figsize=(17,5),sharey=True)
ratios=[.1,.2,.3,.4,.5]
for ax,group in zip(axes,NAMES):
    for scope,label,style in [('global','全局导出','-'),('matched','局部预算＝本组全局贡献数','--')]:
        ax.plot(ratios,[index[(group,f'{scope}_{r:g}')]['psnr_mean'] for r in ratios],style,marker='o',label=label)
    ax.set(title=NAMES[group],xlabel='来源全局ratio',xticks=ratios);ax.grid(color='#E2E8F0');ax.legend(frameon=False,fontsize=9)
axes[0].set_ylabel('平均PSNR（dB）')
fig.suptitle('各自逐帧全局贡献数量匹配预算下的局部质量',fontsize=14)
fig.text(.5,.015,'使用各组自身全局贡献数量，不是三组same-K，也不保证最后Alpha贡献数量严格相同。',ha='center')
fig.tight_layout(rect=(0,.06,1,.94));save(fig,'VIEW_RESIDUAL_GLOBAL_MATCHED_PSNR')
diagnostics=json.loads((REPORTS/'RESIDUAL_ON_OFF.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(16,5))
cases=['view_0.01','view_0.05','view_0.1','points_1000','points_2000']
for ax,key,label in zip(axes,['psnr','ssim','lpips'],['ΔPSNR（dB，↑）','ΔSSIM（↑）','ΔLPIPS（↓）']):
    for j,group in enumerate(['view_no_li','view_li01']):
        values=[next(r for r in diagnostics if r['group']==group and r['case']==c and r['partition']=='all')['delta_'+key+'_mean'] for c in cases]
        ax.bar(np.arange(5)+(j-.5)*.36,values,width=.36,color=COLORS[group],label=NAMES[group])
    ax.axhline(0,color='#475569',lw=1);ax.set(title=label,xticks=np.arange(5),xticklabels=['1%','5%','10%','1k','2k']);ax.grid(axis='y',color='#E2E8F0');ax.set_axisbelow(True)
fig.suptitle('同checkpoint、同视角、同K｜视角残差开启 − 关闭',fontsize=14)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,frameon=False)
fig.tight_layout(rect=(0,.02,1,.83));save(fig,'VIEW_RESIDUAL_ON_OFF')
print('FIGURES_COMPLETE',len(list(FIGURES.glob('*.png'))))

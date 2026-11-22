"""绘制三组质量、固定点数、配对变化和补偿幅度。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT=Path(__file__).resolve().parent/'results/evaluation/reports'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
summary=json.loads((OUT/'SUMMARY.json').read_text())
index={(v['group'],v['case']):v for v in summary if v['partition']=='all'}
paired=json.loads((OUT/'PAIRED_SUMMARY.json').read_text())
pairs={(v['reference'],v['group'],v['case']):v for v in paired if v['partition']=='all'}
groups=['baseline','mode','view'];names=['原无视角联合','仅模式标记','模式＋真实视角'];colors=['#64748B','#0072B2','#D55E00']
ratios=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7];trained=[.01,.05,.1,.3,.5,.7]
font='C:/Windows/Fonts/msyh.ttc';font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':10,
    'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=190,facecolor='white')
    plt.close(fig)


fig,axes=plt.subplots(1,2,figsize=(13,5.6))
for ax,scope,title in zip(axes,['global','view'],['全局导出：mode=0','局部名义Anchor预算：mode=1']):
    for group,label,color in zip(groups,names,colors):
        y=np.array([index[group,f'{scope}_{r:g}']['psnr_mean'] for r in ratios]);ax.plot(ratios,y,lw=2,color=color,label=label)
        for train in [True,False]:
            mask=np.array([(r in trained)==train for r in ratios]);ax.scatter(np.array(ratios)[mask],y[mask],s=30,
                facecolor=color if train else 'white',edgecolor=color,zorder=3)
    ax.set(title=title,xlabel='Ratio',ylabel='792帧平均PSNR（dB）');ax.grid(color='#E2E8F0');ax.legend(frameon=False)
    ax.set_xticks(ratios,[f'{r:g}' for r in ratios],rotation=45,fontsize=8)
fig.suptitle('fr1_xyz RGB-D｜Selector＋Compensator共享条件｜LI=0',fontsize=14)
fig.text(.5,.015,'相同Full起点、seed0、11,580步；N=13,298。实心为训练ratio，空心为未见ratio。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.055,1,.94));save(fig,'JOINT_VIEW_RATIO_PSNR')

fig,axes=plt.subplots(1,2,figsize=(13,5.6))
budgets=[1000,2000,3000,4000,5000,10000]
for group,label,color in zip(groups,names,colors):
    axes[0].plot(budgets,[index[group,f'points_{b}']['psnr_mean'] for b in budgets],color=color,lw=2,marker='o',label=label)
    axes[1].plot(budgets,[index[group,f'points_{b}']['K_mean'] for b in budgets],color=color,lw=2,marker='o',label=label)
axes[0].set(title='局部固定请求预算',xlabel='请求点数B',ylabel='平均PSNR（dB）')
axes[1].set(title='实际提交Gaussian数量',xlabel='请求点数B',ylabel='平均实际K')
for ax in axes:ax.set_xticks(budgets,[str(b) for b in budgets],rotation=40);ax.grid(color='#E2E8F0');ax.legend(frameon=False)
fig.text(.5,.012,'实际K=min(B,合格候选点数)；10k可能候选不足，不能称实际10k对照。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,1));save(fig,'JOINT_VIEW_FIXED_POINTS')

fig,axes=plt.subplots(2,3,figsize=(15,8.2))
for row,scope in enumerate(['global','view']):
    for col,(key,label) in enumerate([('psnr','ΔPSNR↑（dB）'),('ssim','ΔSSIM↑'),('lpips','ΔLPIPS↓')]):
        ax=axes[row,col]
        for reference,group,color,name in [('baseline','mode',colors[1],'仅模式−原联合'),('baseline','view',colors[2],'真实视角−原联合'),('mode','view','#009E73','真实视角−仅模式')]:
            ax.plot(ratios,[pairs[reference,group,f'{scope}_{r:g}']['delta_'+key+'_mean'] for r in ratios],color=color,lw=1.6,marker='o',ms=3,label=name)
        ax.axhline(0,color=colors[0],lw=1);ax.grid(color='#E2E8F0');ax.legend(frameon=False,fontsize=8)
        ax.set(title=('全局' if row==0 else '局部')+' '+label)
        ax.set_xticks(ratios,[f'{r:g}' for r in ratios],rotation=45,fontsize=8)
fig.suptitle('相同792帧配对变化｜SSIM越高越好，LPIPS越低越好',fontsize=14)
fig.tight_layout(rect=(0,0,1,.95));save(fig,'JOINT_VIEW_METRIC_DELTAS')

geo=json.loads((OUT/'GEOMETRY_SUMMARY.json').read_text());gi={(v['group'],v['case']):v for v in geo}
baseline_geo=json.loads((OUT/'BASELINE_GEOMETRY_SUMMARY.json').read_text())['summary']
gi.update({(v['group'],v['case']):v for v in baseline_geo})
fig,axes=plt.subplots(2,2,figsize=(13,9))
for row,scope in enumerate(['global','view']):
    for col,(key,label) in enumerate([('xyz_p99_mean','XYZ/参考尺度 p99'),('scale_p99_mean','最大轴Scale倍率 p99')]):
        ax=axes[row,col]
        for group,name,color in zip(groups[1:],names[1:],colors[1:]):
            ax.plot(ratios,[gi[group,f'{scope}_{r:g}'][key] for r in ratios],color=color,lw=2,marker='o',ms=4,label=name)
        if scope=='view':ax.plot([.01,.05,.1],[gi['baseline',f'view_{r:g}'][key] for r in [.01,.05,.1]],
            color=colors[0],lw=2,ls='--',marker='s',label='原联合：仅前向统计')
        ax.set(title=('全局' if row==0 else '局部')+'补偿幅度',ylabel='各视角'+label+'的均值',xlabel='Ratio')
        ax.grid(color='#E2E8F0');ax.legend(frameon=False);ax.set_xticks(ratios,[f'{r:g}' for r in ratios],rotation=45,fontsize=8)
fig.suptitle('新模型的补偿长尾｜幅度变化不单独代表质量收益',fontsize=14)
fig.text(.5,.012,'基线仅补充三档局部前向几何统计，未重新渲染；Scale按最终渲染输入相对Master计算。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.96));save(fig,'JOINT_VIEW_COMPENSATION_TAILS')

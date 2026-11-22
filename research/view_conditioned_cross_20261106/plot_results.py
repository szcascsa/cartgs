"""五组匹配实验的质量、固定预算与几何曲线；不生成渲染帧。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT=Path(__file__).resolve().parent/'results/evaluation/reports'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
groups=['baseline','mode','selector_view','compensator_view','view']
names=['原无视角联合','两网络仅模式','Selector视角／Comp模式','Selector模式／Comp视角','两网络真实视角']
colors=['#64748B','#0072B2','#009E73','#CC79A7','#D55E00']
ratios=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7];trained=[.01,.05,.1,.3,.5,.7]
summary=json.loads((OUT/'SUMMARY.json').read_text())
index={(v['group'],v['case']):v for v in summary if v['partition']=='all'}
paired=json.loads((OUT/'PAIRED_SUMMARY.json').read_text())
pairs={(v['reference'],v['group'],v['case']):v for v in paired if v['partition']=='all'}
geometry=json.loads((OUT/'GEOMETRY_SUMMARY.json').read_text())
gi={(v['group'],v['case']):v for v in geometry}
gi.update({(v['group'],v['case']):v for v in json.loads((OUT/'BASELINE_GEOMETRY_SUMMARY.json').read_text())['summary']})
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
    'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=190,facecolor='white')
    plt.close(fig)


def decorate(ax,title,ylabel):
    ax.set(title=title,xlabel='Ratio',ylabel=ylabel)
    ax.set_xticks(ratios,[f'{r:g}' for r in ratios],rotation=45,fontsize=8)
    ax.grid(color='#E2E8F0');ax.legend(frameon=False,fontsize=8)


fig,axes=plt.subplots(1,2,figsize=(14,5.6))
for ax,scope,title in zip(axes,['global','view'],['全局导出：两网络条件为零','局部名义Anchor候选池']):
    for g,n,c in zip(groups,names,colors):
        y=np.array([index[g,f'{scope}_{r:g}']['psnr_mean'] for r in ratios])
        ax.plot(ratios,y,lw=1.8,color=c,label=n)
        mask=np.array([r in trained for r in ratios]);ax.scatter(np.array(ratios)[mask],y[mask],s=22,color=c)
    decorate(ax,title,'792帧平均PSNR（dB）')
fig.suptitle('fr1_xyz RGB-D｜视角输入交叉消融｜LI=0',fontsize=14)
fig.text(.5,.012,'相同Full起点、seed0、11,580步；N=13,298。只新增两组训练，旧三组只读复用。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.94));save(fig,'CROSS_VIEW_RATIO_PSNR')

fig,axes=plt.subplots(1,2,figsize=(14,5.6));budgets=[1000,2000,3000,4000,5000,10000]
for g,n,c in zip(groups,names,colors):
    y=np.array([index[g,f'points_{b}']['psnr_mean'] for b in budgets])
    axes[0].plot(budgets,y,lw=1.8,color=c,marker='o',label=n)
    if g!='mode':
        base=np.array([index['mode',f'points_{b}']['psnr_mean'] for b in budgets])
        axes[1].plot(budgets,y-base,lw=1.8,color=c,marker='o',label=n+'−仅模式')
for ax,title,label in zip(axes,['固定请求点数','相对两网络仅模式'],['平均PSNR（dB）','ΔPSNR（dB）']):
    ax.set(title=title,xlabel='请求预算B',ylabel=label);ax.set_xticks(budgets,[str(b) for b in budgets],rotation=40)
    ax.grid(color='#E2E8F0');ax.legend(frameon=False,fontsize=8)
axes[1].axhline(0,color='#64748B',lw=1)
k=[index['selector_view',f'points_{b}']['K_mean'] for b in budgets]
fig.text(.5,.012,'新两组平均实际K：'+', '.join(f'{b}→{v:.0f}' for b,v in zip(budgets,k))+'；实际K=min(B,合格候选)。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,1));save(fig,'CROSS_VIEW_FIXED_POINTS')

fig,axes=plt.subplots(2,3,figsize=(15,8.3))
for row,scope in enumerate(['global','view']):
    for col,(key,label) in enumerate([('psnr','ΔPSNR↑（dB）'),('ssim','ΔSSIM↑'),('lpips','ΔLPIPS↓')]):
        ax=axes[row,col]
        for g,n,c in zip(groups[1:],names[1:],colors[1:]):
            ax.plot(ratios,[pairs['baseline',g,f'{scope}_{r:g}']['delta_'+key+'_mean'] for r in ratios],color=c,lw=1.6,marker='o',ms=3,label=n)
        ax.axhline(0,color='#64748B',lw=1)
        decorate(ax,('全局 ' if row==0 else '局部 ')+label,label)
fig.suptitle('相对原无视角联合｜相同792帧配对变化',fontsize=14)
fig.tight_layout(rect=(0,0,1,.95));save(fig,'CROSS_VIEW_METRIC_DELTAS')

fig,axes=plt.subplots(2,2,figsize=(14,9))
for row,scope in enumerate(['global','view']):
    for col,(key,label) in enumerate([('xyz_p99_mean','XYZ／Anchor参考尺度 p99'),('scale_p99_mean','最大轴Scale相对Master倍率 p99')]):
        ax=axes[row,col]
        for g,n,c in zip(groups[1:],names[1:],colors[1:]):
            ax.plot(ratios,[gi[g,f'{scope}_{r:g}'][key] for r in ratios],color=c,lw=1.8,marker='o',ms=3,label=n)
        if scope=='view':
            ax.plot([.01,.05,.1],[gi['baseline',f'view_{r:g}'][key] for r in [.01,.05,.1]],ls='--',color=colors[0],marker='s',label=names[0]+'：既有前向统计')
        decorate(ax,('全局' if row==0 else '局部')+'补偿长尾','逐视角'+label+'的均值')
fig.suptitle('视角输入由哪一分支读取时出现几何长尾',fontsize=14)
fig.text(.5,.012,'参考尺度为Anchor边长2.5／5／10 cm；混合层级不能统一换算为米。原基线没有重渲染。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.96));save(fig,'CROSS_VIEW_COMPENSATION_TAILS')

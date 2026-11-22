"""三组LI相对相同结构无LI的配对质量与XYZ曲线。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT=Path(__file__).resolve().parent/'results/evaluation/reports'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
groups=['selector_li','compensator_li','both_li']
controls={'selector_li':'selector_view','compensator_li':'compensator_view','both_li':'view'}
names=['Selector视角／Comp模式','Selector模式／Comp视角','两网络真实视角']
colors=['#009E73','#CC79A7','#D55E00']
ratios=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7];trained=[.01,.05,.1,.3,.5,.7]
index={(v['group'],v['case']):v for v in json.loads((OUT/'SUMMARY.json').read_text()) if v['partition']=='all'}
pairs={(v['reference'],v['group'],v['case']):v for v in json.loads((OUT/'PAIRED_SUMMARY.json').read_text()) if v['partition']=='all'}
geo={(v['group'],v['case']):v for v in json.loads((OUT/'GEOMETRY_SUMMARY.json').read_text())}
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
    'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=190,facecolor='white')
    plt.close(fig)


def decorate(ax,title,ylabel):
    ax.set(title=title,xlabel='Ratio',ylabel=ylabel)
    ax.set_xticks(trained,[f'{100*r:g}%' for r in trained],rotation=30,fontsize=8)
    ax.grid(color='#E2E8F0');ax.legend(frameon=False,fontsize=8)


fig,axes=plt.subplots(2,3,figsize=(16,9))
for row,scope in enumerate(['global','view']):
    for col,(g,n,c) in enumerate(zip(groups,names,colors)):
        ax=axes[row,col]
        for name,label,color,style in [(controls[g],'同结构无LI','#64748B','--'),(g,'LI=0.1',c,'-')]:
            y=np.asarray([index[name,f'{scope}_{r:g}']['psnr_mean'] for r in ratios])
            ax.plot(ratios,y,color=color,lw=2,ls=style,label=label)
            mask=np.asarray([r in trained for r in ratios]);ax.scatter(np.asarray(ratios)[mask],y[mask],color=color,s=20)
        decorate(ax,('全局｜' if row==0 else '局部｜')+n,'792帧平均PSNR（dB）')
fig.suptitle('fr1_xyz RGB-D｜加入LI是否改善同结构结果',fontsize=14)
fig.text(.5,.012,'同一Full起点、seed0、11,580步；N=13,298。旧无LI结果只读复用；实心标记为训练ratio。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.95));save(fig,'LI_VIEW_RATIO_PSNR')

fig,axes=plt.subplots(1,3,figsize=(16,5.4));budgets=[1000,2000,3000,4000,5000,10000]
for ax,g,n,c in zip(axes,groups,names,colors):
    for name,label,color,style in [(controls[g],'同结构无LI','#64748B','--'),(g,'LI=0.1',c,'-')]:
        ax.plot(budgets,[index[name,f'points_{b}']['psnr_mean'] for b in budgets],color=color,lw=2,ls=style,marker='o',label=label)
    ax.set(title=n,xlabel='请求点数B',ylabel='平均PSNR（dB）');ax.grid(color='#E2E8F0');ax.legend(frameon=False)
    ax.set_xticks(budgets,[f'{b/1000:g}k' for b in budgets],rotation=35)
fig.text(.5,.015,'实际K=min(B,合格候选E)；候选不足及网络实际ratio逐帧记录，10k不代表实际保留10k。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,1));save(fig,'LI_VIEW_FIXED_POINTS')

fig,axes=plt.subplots(2,3,figsize=(16,9))
for row,scope in enumerate(['global','view']):
    for col,(metric,label) in enumerate([('psnr','ΔPSNR↑（dB）'),('ssim','ΔSSIM↑'),('lpips','ΔLPIPS↓')]):
        ax=axes[row,col]
        for g,n,c in zip(groups,names,colors):
            ax.plot(ratios,[pairs[controls[g],g,f'{scope}_{r:g}']['delta_'+metric+'_mean'] for r in ratios],color=c,lw=1.8,marker='o',ms=3,label=n)
        ax.axhline(0,color='#64748B',lw=1);decorate(ax,('全局 ' if row==0 else '局部 ')+label,label)
fig.suptitle('LI=0.1 − 同结构无LI｜同792帧配对变化',fontsize=14)
fig.tight_layout(rect=(0,0,1,.95));save(fig,'LI_VIEW_METRIC_DELTAS')

fig,axes=plt.subplots(2,3,figsize=(16,9))
for row,scope in enumerate(['global','view']):
    for col,(g,n,c) in enumerate(zip(groups,names,colors)):
        ax=axes[row,col]
        for name,label,color,style in [(controls[g],'同结构无LI','#64748B','--'),(g,'LI=0.1',c,'-')]:
            ax.plot(ratios,[geo[name,f'{scope}_{r:g}']['xyz_p99_mean'] for r in ratios],color=color,lw=2,ls=style,marker='o',ms=3,label=label)
        decorate(ax,('全局｜' if row==0 else '局部｜')+n,'XYZ／参考尺度p99的视角均值')
fig.suptitle('LI对XYZ补偿长尾的影响',fontsize=14)
fig.text(.5,.012,'参考尺度为Anchor边长2.5／5／10 cm；先逐视角计算p99再平均，不能统一换算为米。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.95));save(fig,'LI_VIEW_XYZ_TAILS')

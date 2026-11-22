"""生成可导出的质量、几何与速度对照图；不输出渲染帧。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT=Path(__file__).resolve().parent/'results'
OUT=ROOT/'evaluation/reports';FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
    'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
ratios=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7];train=[.01,.05,.1,.3,.5,.7]
groups=[('four_head','原四Head','#64748B'),('no_other','移除Other Head','#D55E00')]
index={(r['group'],r['case']):r for r in json.loads((OUT/'SUMMARY.json').read_text()) if r['partition']=='all'}
geo={(r['group'],r['case']):r for r in json.loads((OUT/'GEOMETRY_SUMMARY.json').read_text())}


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=180,facecolor='white',bbox_inches='tight')
    plt.close(fig)


fig,axes=plt.subplots(2,3,figsize=(15,8))
for row,scope in enumerate(['global','view']):
    for col,(metric,label) in enumerate([('psnr','PSNR↑（dB）'),('ssim','SSIM↑'),('lpips','LPIPS↓')]):
        ax=axes[row,col]
        for g,name,color in groups:
            y=[index[g,f'{scope}_{r:g}'][metric+'_mean'] for r in ratios]
            ax.plot(ratios,y,color=color,lw=2,marker='o',ms=3,label=name)
        ax.set_xscale('log')
        ax.set(title=('全局导出' if row==0 else '当前视角预算')+'｜'+label,xlabel='Ratio（对数间距）',ylabel=label)
        ax.set_xticks(train,[f'{r*100:g}%' for r in train],rotation=25);ax.grid(alpha=.25);ax.legend(frameon=False,fontsize=9)
fig.suptitle('fr1_xyz RGB-D｜两网络真实视角＋LI=0.1｜Other Head消融',fontsize=14)
fig.text(.5,.005,'同一Full起点、seed0、11,580步、相同792帧；N=13,298。局部K=floor(r×名义可见Anchor成员数)。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.95));save(fig,'NO_OTHER_RATIO_QUALITY')

fig,axes=plt.subplots(1,3,figsize=(15,4.6));budgets=[500,1000,2000,3000,4000,5000,10000]
for ax,(metric,label) in zip(axes,[('psnr','PSNR↑（dB）'),('ssim','SSIM↑'),('lpips','LPIPS↓')]):
    for g,name,color in groups:
        ax.plot(budgets,[index[g,f'points_{b}'][metric+'_mean'] for b in budgets],color=color,lw=2,marker='o',ms=4,label=name)
    ax.set_xscale('log');ax.set(xlabel='局部请求预算B（对数间距）',ylabel=label);ax.set_xticks(budgets,[f'{b/1000:g}k' for b in budgets],rotation=30)
    ax.grid(alpha=.25);ax.legend(frameon=False,fontsize=9)
fig.text(.5,.005,'实际K=min(B,合格候选E)，候选不足时不补点；10k请求不等于每个视角实际10k。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,1));save(fig,'NO_OTHER_FIXED_BUDGET_QUALITY')

fig,axes=plt.subplots(2,2,figsize=(12,8))
for row,scope in enumerate(['global','view']):
    for col,(key,label) in enumerate([('xyz_p99_mean','XYZ／参考尺度p99的视角均值'),('scale_p99_mean','最大轴Scale倍率p99的视角均值')]):
        ax=axes[row,col]
        for g,name,color in groups:ax.plot(ratios,[geo[g,f'{scope}_{r:g}'][key] for r in ratios],color=color,lw=2,marker='o',ms=3,label=name)
        ax.set_xscale('log')
        ax.set(title='全局' if row==0 else '局部',xlabel='Ratio（对数间距）',ylabel=label)
        ax.set_xticks(train,[f'{r*100:g}%' for r in train],rotation=25);ax.grid(alpha=.25);ax.legend(frameon=False)
fig.tight_layout();save(fig,'NO_OTHER_GEOMETRY')

speed=json.loads((ROOT/'speed/SUMMARY.json').read_text())
si={(r['group'],r['budget']):r for r in speed if r['repeat']=='all'}
fig,axes=plt.subplots(1,3,figsize=(15,5))
for ax,budget,title in zip(axes[:2],['points500','ratio50'],['局部500点','局部50%']):
    bottom=np.zeros(2)
    for key,label,color in [('selector_ms_mean','Selector／选择','#0072B2'),('compensation_ms_mean','Compensation','#D55E00'),
                            ('other_prepare_ms_mean','其他准备','#E69F00'),('render_ms_mean','Rasterizer','#009E73')]:
        y=np.array([si[g,budget][key] for g,_,_ in groups]);ax.bar([0,1],y,bottom=bottom,color=color,label=label);bottom+=y
    for x,value in enumerate(bottom):ax.text(x,value+.08,f'{value:.3f} ms',ha='center',fontsize=10)
    ax.set(title=title,ylabel='逐帧同步墙钟均值（ms）',xticks=[0,1],xticklabels=['原四Head','去Other']);ax.grid(axis='y',alpha=.2)
axes[0].legend(frameon=False,fontsize=8)
micro=json.loads((ROOT/'speed/COMPENSATION_SAME_INPUT_SUMMARY.json').read_text())
labels=[];a=[];b=[]
for k in [500,4096]:
    for ratio in [.05,.5]:
        labels.append(f'K={k}\nr={ratio:g}')
        for group,target in [('four_head',a),('no_other',b)]:target.append(next(r['cuda_stream_ms_mean']*1000 for r in micro if r['K']==k and r['ratio']==ratio and r['group']==group))
x=np.arange(4);axes[2].bar(x-.18,a,.36,color=groups[0][2],label='原四Head');axes[2].bar(x+.18,b,.36,color=groups[1][2],label='去Other')
axes[2].set(title='同输入、同保留权重的纯Compensation',ylabel='CUDA Event均值（μs）',xticks=x,xticklabels=labels);axes[2].grid(axis='y',alpha=.2);axes[2].legend(frameon=False,fontsize=9)
fig.text(.5,.005,'端到端包含选点／补偿／渲染；微测仅网络前向。792帧×3轮交错配对，全部轮次保留。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,1));save(fig,'NO_OTHER_LATENCY')

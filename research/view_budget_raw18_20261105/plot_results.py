"""Windows绘制18D/39D匹配LI对照；曲线为原始逐帧指标均值。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT=Path(__file__).resolve().parent/'results/evaluation/reports'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
RATIOS=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]
TRAIN=[.01,.05,.1,.3,.5,.7]
SUMMARY=json.loads((OUT/'SUMMARY.json').read_text())
INDEX={(r['group'],r['case']):r for r in SUMMARY if r['partition']=='all'}
PAIRS=json.loads((OUT/'PAIRED_SUMMARY.json').read_text())
PAIR={(r['reference'],r['group'],r['case']):r for r in PAIRS if r['partition']=='all'}
font='C:/Windows/Fonts/msyh.ttc';font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':10,
    'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
COLORS=['#64748B','#0072B2','#D55E00']
PANELS=[('view_no_li','raw_no_li','无LI'),('view_li01','raw_li01','LI=0.1')]


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=190,facecolor='white')
    plt.close(fig)


fig,axes=plt.subplots(2,2,figsize=(14,10))
for col,(old,new,label) in enumerate(PANELS):
    for row,(scope,title) in enumerate([('global','全局ratio导出'),('view','单视角名义Anchor预算')]):
        ax=axes[row,col]
        for group,name,color in zip(['baseline',old,new],['原联合无LI','39D中间特征','18D原始属性'],COLORS):
            v=np.array([INDEX[(group,f'{scope}_{r:g}')]['psnr_mean'] for r in RATIOS])
            ax.plot(RATIOS,v,color=color,lw=2,label=name)
            for trained in [True,False]:
                ids=[i for i,r in enumerate(RATIOS) if (r in TRAIN)==trained]
                ax.scatter(np.array(RATIOS)[ids],v[ids],s=29,edgecolor=color,facecolor=color if trained else 'white',zorder=3)
        ax.set(title=title+'｜'+label,xlabel='Ratio',ylabel='792帧平均PSNR（dB）')
        ax.set_xticks(RATIOS,[f'{r:g}' for r in RATIOS],rotation=45,fontsize=8)
        ax.grid(color='#E2E8F0');ax.legend(frameon=False)
fig.suptitle('fr1_xyz RGB-D｜18D与39D视角残差｜相同Full起点、11,580步',fontsize=15)
fig.text(.5,.012,'实心为训练ratio、空心为未见ratio；均值不平滑；单场景seed0。',ha='center')
fig.tight_layout(rect=(0,.04,1,.96));save(fig,'RAW18_RATIO_PSNR')

fig,axes=plt.subplots(1,2,figsize=(14,5.8))
budgets=[1000,2000,3000,4000,5000,10000]
for ax,(old,new,label) in zip(axes,PANELS):
    for group,name,color in zip(['baseline',old,new],['原联合无LI','39D中间特征','18D原始属性'],COLORS):
        ax.plot(budgets,[INDEX[(group,f'points_{b}')]['psnr_mean'] for b in budgets],color=color,lw=2,marker='o',label=name)
    ax.set(title='局部固定请求预算｜'+label,xlabel='请求点数B',ylabel='平均PSNR（dB）',xticks=budgets)
    ax.grid(color='#E2E8F0');ax.legend(frameon=False)
fig.suptitle('18D／39D局部固定点数质量',fontsize=15)
fig.text(.5,.015,'实际K=min(B,E)；10k可能候选不足。不同模型的地图、候选及最终Alpha贡献数可不同。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.055,1,.94));save(fig,'RAW18_FIXED_BUDGET_PSNR')

fig,axes=plt.subplots(2,3,figsize=(16,8))
for row,scope in enumerate(['global','view']):
    for col,(key,label) in enumerate([('psnr','ΔPSNR↑（dB）'),('ssim','ΔSSIM↑'),('lpips','ΔLPIPS↓')]):
        ax=axes[row,col]
        for (old,new,name),color in zip(PANELS,COLORS[1:]):
            ax.plot(RATIOS,[PAIR[(old,new,f'{scope}_{r:g}')]['delta_'+key+'_mean'] for r in RATIOS],
                color=color,lw=1.8,marker='o',ms=3,label=name)
        ax.axhline(0,color=COLORS[0],lw=1);ax.grid(color='#E2E8F0')
        ax.set(title=('全局' if row==0 else '局部')+' 18D−39D '+label)
        ax.set_xticks(RATIOS,[f'{r:g}' for r in RATIOS],rotation=45,fontsize=8);ax.legend(frameon=False)
fig.suptitle('同LI配置：18D−39D配对变化｜相同792帧',fontsize=15)
fig.text(.5,.012,'SSIM越高越好，LPIPS越低越好；单seed不能证明稳定收益。',ha='center')
fig.tight_layout(rect=(0,.045,1,.95));save(fig,'RAW18_METRIC_DELTAS')

diagnostics=json.loads((OUT/'RESIDUAL_ON_OFF.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(13,5.8))
cases=['view_0.01','view_0.05','view_0.1','points_1000','points_2000']
for ax,(old,new,label) in zip(axes,PANELS):
    for j,group in enumerate([old,new]):
        v=[next(r for r in diagnostics if r['group']==group and r['case']==c and r['partition']=='all')['delta_psnr_mean'] for c in cases]
        ax.bar(np.arange(5)+(j-.5)*.34,v,width=.34,color=COLORS[j+1],label=['39D','18D'][j])
    ax.axhline(0,color=COLORS[0],lw=1);ax.set(title=label,ylabel='残差开−关ΔPSNR（dB）',xticks=np.arange(5),xticklabels=['1%','5%','10%','1k','2k'])
    ax.grid(axis='y',color='#E2E8F0');ax.set_axisbelow(True);ax.legend(frameon=False)
fig.suptitle('同checkpoint、同相机、同K：残差本身在推理时的作用',fontsize=14)
fig.tight_layout(rect=(0,.02,1,.94));save(fig,'RAW18_RESIDUAL_ON_OFF')
print('FIGURES_COMPLETE',len(list(FIG.glob('*.png'))))

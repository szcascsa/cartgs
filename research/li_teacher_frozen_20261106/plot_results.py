"""Windows绘制冻结LI教师诊断的真实配对结果。"""
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
pairs=json.loads((OUT/'PAIRED_SUMMARY.json').read_text())
si={(r['ratio'],r['selection'],r['compensation']):r for r in summary if r['partition']=='all'}
pi={(r['ratio'],r['reference'],r['selection'],r['compensation']):r for r in pairs if r['partition']=='all'}
font='C:/Windows/Fonts/msyh.ttc';font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':11,
    'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
colors=['#64748B','#0072B2','#D55E00'];names=['原Selector／残差关','18D视角残差','Full LI教师']


def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=180,facecolor='white')
    plt.close(fig)


fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax,r in zip(axes,[.01,.05]):
    x=np.arange(3)
    for j,c in enumerate([False,True]):
        v=[si[(r,s,c)]['psnr_mean'] for s in ['base','view','teacher']]
        bars=ax.bar(x+(j-.5)*.34,v,width=.34,color=colors,alpha=.5 if not c else 1,hatch='//' if not c else '',label='补偿关' if not c else '补偿开')
        for bar,value in zip(bars,v):ax.annotate(f'{value:.3f}',(bar.get_x()+bar.get_width()/2,bar.get_height()),ha='center',va='bottom',fontsize=9,xytext=(0,3),textcoords='offset points')
    ax.set(title=f'局部{r:.0%}预算',xticks=x,xticklabels=names,ylabel='792帧平均PSNR（dB）')
    ax.grid(axis='y',color='#E2E8F0');ax.set_axisbelow(True);ax.legend(frameon=False)
    vals=[si[(r,s,c)]['psnr_mean'] for s in ['base','view','teacher'] for c in [False,True]]
    ax.set_ylim(min(vals)-.9,max(vals)+1.)
fig.suptitle('固定18D＋LI模型｜教师集合是否真正改善渲染',fontsize=15)
fig.text(.5,.012,'三种集合使用相同候选、实际K和网络ratio；补偿开关前后选点ID完全相同。',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.06,1,.94));save(fig,'LI_TEACHER_PSNR')

fig,axes=plt.subplots(1,3,figsize=(16,5.5))
for ax,key,label in zip(axes,['psnr','ssim','lpips'],['ΔPSNR（dB，↑）','ΔSSIM（↑）','ΔLPIPS（↓）']):
    x=np.arange(2)
    for j,(ref,c,name,color) in enumerate([('base',False,'教师−原Selector，补偿关','#94A3B8'),('base',True,'教师−原Selector，补偿开','#D55E00'),('view',True,'教师−视角残差，补偿开','#0072B2')]):
        v=[pi[(r,ref,'teacher',c)]['delta_'+key+'_mean'] for r in [.01,.05]]
        ax.bar(x+(j-1)*.24,v,width=.24,color=color,label=name)
    ax.axhline(0,color='#475569',lw=1);ax.set(title=label,xticks=x,xticklabels=['1%','5%']);ax.grid(axis='y',color='#E2E8F0');ax.set_axisbelow(True)
fig.suptitle('相同792帧：Full LI教师的配对指标变化',fontsize=15)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.92),ncol=3,frameon=False,fontsize=10)
fig.text(.5,.01,'固定地图诊断；Teacher读取当前Full贡献，不是可部署免费推理或全局最优点集。',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.06,1,.80));save(fig,'LI_TEACHER_DELTAS')
print('FIGURES_COMPLETE')

"""正式交替计时；误差线仅表示两轮p50范围，不能当作置信区间。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;a=pd.read_csv(D/'evidence_final/phase2_online_tail_performance.csv');a=a[a.scope.str.startswith('fixed')]
fig,axes=plt.subplots(1,3,figsize=(14,4));colors=['#31688e','#35b779','#e6a33b']
for ax,resolution,cases,variants,title in [
    (axes[0],1.,['Full'],['B','C','E'],'Full 640x480'),
    (axes[1],.5,['lod_0.010000','lod_0.050000','lod_0.100000','lod_0.150000'],['B','D','E'],'Online LoD 320x240'),
    (axes[2],1.,['lod_0.010000','lod_0.050000','lod_0.100000','lod_0.150000'],['B','D','E'],'Tail LoD 640x480')]:
    x=np.arange(len(cases));width=.23
    for j,(v,color) in enumerate(zip(variants,colors)):
        groups=[a[(a.variant==v)&(a.resolution==resolution)&a.case.eq(case)].p50 for case in cases]
        mean=np.array([g.mean() for g in groups]);low=np.array([g.min() for g in groups]);high=np.array([g.max() for g in groups])
        ax.bar(x+(j-1)*width,mean,width,label=v,color=color,yerr=np.array([mean-low,high-mean]),capsize=3)
    ax.set_xticks(x,['Full'] if len(cases)==1 else ['1%','5%','10%','15%']);ax.set_title(title);ax.set_ylabel('Core replay time (ms)');ax.set_ylim(0,4.6);ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Fixed checkpoint; shared RTX3090; error bars = two-run p50 range (not CI)',fontsize=12)
fig.tight_layout();fig.savefig(D/'phase2_latency_summary.png',dpi=170);print('PLOT_PASS')

"""从已完成CSV生成静态研究图，不改变实验。"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def rows(n):return list(csv.DictReader(open(ROOT/n)))
arms=['A_Raw','B_FixedMarginal','C_Hierarchical','D_RepresentativeOnly','E_L1Only']
names=['Raw SI','Fixed h8 Marginal','Hierarchical SI','Representative only','L1 only'];colors=['#747b86','#2869ba','#c74236','#a077b1','#b59b3b']
q=rows('quality_summary.csv');red=rows('selected_redundancy.csv');comp=rows('hierarchical_topk_composition.csv')
fig,ax=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
for arm,name,color in zip(arms,names,colors):
    a=sorted([r for r in q if r['arm']==arm],key=lambda r:float(r['ratio']));x=np.array([100*float(r['ratio'])for r in a]);style='-'if arm in arms[:3]else'--'
    ax[0,0].plot(x,[float(r['psnr'])for r in a],style,marker='o',label=name,color=color)
    ax[0,2].plot(x,[float(r['lpips'])for r in a],style,marker='o',label=name,color=color)
    if arm in arms[:3]:
        a=sorted([r for r in red if r['arm']==arm],key=lambda r:float(r['ratio']));ax[1,1].plot(x,[float(r['mean'])for r in a],marker='o',label=name,color=color)
for label,color in [('C_vs_Raw',colors[0]),('C_vs_FixedMarginal',colors[1])]:
    a=[r for r in rows('quality_deltas.csv')if r['comparison']==label];ax[0,1].plot([100*float(r['ratio'])for r in a],[float(r['delta_psnr'])for r in a],marker='o',label=label,color=color)
a=[r for r in comp if r['arm']=='C_Hierarchical'];ax[1,0].plot([100*float(r['ratio'])for r in a],[100*float(r['representative_fraction'])for r in a],marker='o',color=colors[2]);ax[1,0].set_ylim(0,105)
detail=rows('representative_delta_statistics.csv');raw=np.array([float(r['raw_I'])for r in detail]);delta=np.array([float(r['Delta'])for r in detail]);valid=(raw>0)&(delta>0)
ax[1,2].scatter(raw[valid],delta[valid],s=4,alpha=.15,c=[int(r['level'])for r,v in zip(detail,valid)if v],cmap='viridis',rasterized=True);ax[1,2].set_xscale('log');ax[1,2].set_yscale('log');ax[1,2].set_xlabel('Raw SI of representative');ax[1,2].set_ylabel('Representative Delta')
for panel,title in zip(ax.flat,['PSNR: same K, frozen TF','Hierarchical minus baseline PSNR','LPIPS: lower is better','Final-leaf representative share (%)','Uniform h8 geometry redundancy','Representative Delta versus Raw SI']):panel.set_title(title);panel.grid(alpha=.2)
for panel in [ax[0,0],ax[0,1],ax[0,2],ax[1,0],ax[1,1]]:panel.set_xlabel('Budget ratio (%)');panel.set_xticks([1,5,10,15])
ax[0,0].set_ylabel('PSNR (dB)');ax[0,1].set_ylabel('Delta PSNR (dB)');ax[0,1].axhline(0,color='black',lw=.8);ax[0,0].legend(fontsize=8);ax[0,1].legend(fontsize=8);ax[1,1].legend(fontsize=8)
fig.suptitle('fr1_room: 161 fixed non-Bank archived evaluation views',fontsize=14);fig.savefig(ROOT/'quality_and_mechanism.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for axis,metric,title in zip(axes.flat,['psnr','ssim','lpips','photo'],['Delta PSNR (higher is better)','Delta SSIM (higher is better)','Delta LPIPS (lower is better)','Delta photo (lower is better)']):
    for label,color in [('C_vs_Raw',colors[2]),('C_vs_FixedMarginal',colors[1])]:
        a=[r for r in rows('quality_deltas.csv')if r['comparison']==label]
        axis.plot([100*float(r['ratio'])for r in a],[float(r['delta_'+metric])for r in a],marker='o',label=label,color=color)
    axis.axhline(0,color='black',lw=.8);axis.set_title(title);axis.set_xlabel('Budget ratio (%)');axis.set_xticks([1,5,10,15]);axis.grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.suptitle('All four metrics: high-budget SSIM / photo tradeoff');fig.savefig(ROOT/'metric_tradeoffs.png',dpi=170);plt.close(fig)
r=np.load(ROOT/'arm_rankings.npz');N=len(r['row_ids']);a=np.empty(N,int);b=np.empty(N,int);a[r['A_Raw_order']]=np.arange(1,N+1);b[r['C_Hierarchical_order']]=np.arange(1,N+1)
fig,ax=plt.subplots(figsize=(6,6),layout='constrained');ax.scatter(a,b,s=2,alpha=.12,c=np.where(r['rep_mask'],'#bf4338','#2869ba'),rasterized=True);ax.plot([1,N],[1,N],color='black',lw=.8);ax.set(xlabel='Raw SI rank',ylabel='Hierarchical SI rank',title='Complete ranking: red representatives, blue residuals');fig.savefig(ROOT/'raw_vs_hierarchical_rank.png',dpi=160);plt.close(fig)
print('PLOTS_PASS',flush=True)

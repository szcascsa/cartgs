from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
q=list(csv.DictReader(open(ROOT/'progressive_marginal_quality_summary.csv')));d=list(csv.DictReader(open(ROOT/'quality_deltas.csv')));red=list(csv.DictReader(open(ROOT/'selected_redundancy.csv')))
arms=['B_FixedMarginal','C_Hierarchical','F1_MarginalTail','F2_ProgressiveResidual','F3_ProgressiveMarginal'];labels=['Fixed h8','Current C','F1 marginal tail','F2 progressive','F3 combined'];colors=['#717780','#ce4b3b','#319c78','#be922e','#2e6db5']
fig,ax=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
for arm,label,color in zip(arms,labels,colors):
    a=[v for v in q if v['arm']==arm];x=np.array([100*float(v['ratio'])for v in a])
    for panel,m in [(ax[0,0],'psnr'),(ax[0,2],'lpips')]:panel.plot(x,[float(v[m])for v in a],marker='o',color=color,label=label)
    a=[v for v in red if v['arm']==arm];ax[1,0].plot(x,[float(v['mean_redundancy'])for v in a],marker='o',color=color,label=label)
    if arm.startswith('F'):
        a=[v for v in d if v['comparison']==arm+'_minus_C_Hierarchical']
        for panel,m in [(ax[0,1],'psnr'),(ax[1,1],'ssim'),(ax[1,2],'photo')]:panel.plot(x,[float(v['delta_'+m])for v in a],marker='o',color=color,label=label)
for panel,title in zip(ax.flat,['PSNR: same K, frozen TF','Delta PSNR versus C','LPIPS: lower is better','Uniform h8 selected redundancy','Delta SSIM versus C','Delta production photo versus C']):panel.set_title(title);panel.set_xlabel('Ratio (%)');panel.set_xticks([1,5,10,15]);panel.grid(alpha=.2)
for panel in [ax[0,1],ax[1,1],ax[1,2]]:panel.axhline(0,color='black',lw=.7)
ax[0,0].set_ylabel('dB');ax[0,1].set_ylabel('dB');ax[0,0].legend(fontsize=8);fig.suptitle('fr1_room: marginal tail and progressive ranking, frozen 161 archived views');fig.savefig(ROOT/'progressive_quality.png',dpi=170);plt.close(fig)
comp=list(csv.DictReader(open(ROOT/'progressive_composition.csv')))
fig,ax=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
for panel,arm in zip(ax,['F1_MarginalTail','F2_ProgressiveResidual','F3_ProgressiveMarginal']):
    a=[v for v in comp if v['arm']==arm];bottom=np.zeros(4);x=np.arange(4)
    for level,color in [(1,'#3b79aa'),(2,'#4ca694'),(3,'#d8a34b')]:
        value=np.array([100*float(v['L'+str(level)+'_fraction'])for v in a]);panel.bar(x,value,bottom=bottom,color=color,label='L'+str(level));bottom+=value
    panel.set_xticks(x,['1%','5%','10%','15%']);panel.set_ylim(0,103);panel.set_title(arm);panel.set_ylabel('Emitted depth share (%)');panel.legend()
fig.suptitle('Progressive depth composition: emitted node depth, not original C rep identity');fig.savefig(ROOT/'progressive_depth.png',dpi=170);plt.close(fig)
print('PLOTS_PASS',flush=True)

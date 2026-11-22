from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def load(n):return list(csv.DictReader(open(ROOT/n)))
scenes=['fr1_room','fr1_xyz','fr2_xyz'];ratios=[.01,.05,.1,.15];seeds=[2026,2027];d=load('quality_deltas.csv')
fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
for y,seed in enumerate(seeds):
    for x,reference in enumerate(['PG-ST','Raw','C']):
        ax=axes[y,x];a=np.array([[float(next(z['psnr']for z in d if z['scene']==s and float(z['ratio'])==r and int(z['seed'])==seed and z['reference']==reference and z['split']=='eval'))for r in ratios]for s in scenes]);limit=max(abs(a).max(),.001);im=ax.imshow(a,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto');ax.set_title(f'MLP-ST - {reference}, seed {seed} (dB)');ax.set_xticks(range(4),['1%','5%','10%','15%']);ax.set_yticks(range(3),scenes)
        for i in range(3):
            for j in range(4):ax.text(j,i,f'{a[i,j]:+.3f}',ha='center',va='center',color='white'if abs(a[i,j])>.6*limit else'black')
        fig.colorbar(im,ax=ax,shrink=.7)
fig.savefig(ROOT/'quality_deltas.png',dpi=170);plt.close(fig)
hard=load('hard_training_dynamics.csv');q=load('quality_summary.csv');fig,axes=plt.subplots(3,4,figsize=(14,9),constrained_layout=True)
for y,name in enumerate(scenes):
    for x,r in enumerate(ratios):
        ax=axes[y,x]
        for seed in seeds:
            z=sorted([z for z in hard if z['scene']==name and float(z['ratio'])==r and int(z['seed'])==seed],key=lambda z:int(z['step']));base=float(z[0]['psnr']);ax.plot([int(z['step'])for z in z],[float(z['psnr'])-base for z in z],label=f'MLP {seed}')
            st=float(next(z['psnr']for z in q if z['scene']==name and float(z['ratio'])==r and int(z['seed'])==seed and z['arm']=='PG-ST'and z['split']=='train'));ax.axhline(st-base,ls='--',alpha=.6,label=f'PG final {seed}')
        ax.axhline(0,color='gray',lw=.6);ax.set_title(f'{name} {int(r*100)}%');ax.set_xlabel('step');ax.set_ylabel('Bank hard PSNR - Raw (dB)');ax.legend(fontsize=6)
fig.savefig(ROOT/'hard_training_curves.png',dpi=160);plt.close(fig)
sm=load('spatial_smoothness_statistics.csv');fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
for ax,name in zip(axes,scenes):
    for arm in ['MLP-ST','PG-ST']:
        rows=[z for z in sm if z['scene']==name and float(z['ratio'])==.05 and int(z['seed'])==2026 and z['arm']==arm];ax.plot([float(z['mean_distance'])for z in rows],[float(z['mean'])for z in rows],'o-',label=arm)
    ax.set_title(name+' 5%, seed 2026');ax.set_xlabel('normalized spatial pair distance');ax.set_ylabel('mean absolute correction difference');ax.legend()
fig.savefig(ROOT/'spatial_smoothness.png',dpi=170);plt.close(fig)
ind=load('indirect_update_statistics.csv');fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
for ax,name in zip(axes,scenes):
    for seed in seeds:
        rows=[next(z for z in ind if z['scene']==name and float(z['ratio'])==r and int(z['seed'])==seed and z['diagnostic']=='cumulative_never_selected'and int(z['step'])==2000)for r in ratios]
        ax.plot([1,5,10,15],[float(z['unselected_N'])/int(z['N'])for z in rows],'o-',label=f'never selected share {seed}');ax.plot([1,5,10,15],[float(z['unselected_changed_fraction'])for z in rows],'--',label=f'never selected changed {seed}')
    ax.set_title(name);ax.set_xlabel('budget (%)');ax.set_ylabel('fraction');ax.set_ylim(0,1.05);ax.legend(fontsize=7)
fig.savefig(ROOT/'indirect_updates.png',dpi=170);plt.close(fig)

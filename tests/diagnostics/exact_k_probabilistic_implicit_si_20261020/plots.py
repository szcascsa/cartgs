from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def load(n):return list(csv.DictReader(open(ROOT/n)))
scenes=['fr1_room','fr1_xyz','fr2_xyz'];ratios=[.01,.05,.1,.15];seeds=[2026,2027];d=load('quality_deltas.csv')
fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
for y,seed in enumerate(seeds):
    for x,ref in enumerate(['PG-ST','Raw','C']):
        ax=axes[y,x];a=np.array([[float(next(z['psnr']for z in d if z['scene']==s and float(z['ratio'])==r and int(z['seed'])==seed and z['reference']==ref and z['split']=='eval'))for r in ratios]for s in scenes]);lim=max(abs(a).max(),.001)
        im=ax.imshow(a,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto');ax.set_xticks(range(4),['1%','5%','10%','15%']);ax.set_yticks(range(3),scenes);ax.set_title(f'PG-PL - {ref}, seed {seed} (dB)')
        for i in range(3):
            for j in range(4):ax.text(j,i,f'{a[i,j]:+.3f}',ha='center',va='center',color='white'if abs(a[i,j])>.6*lim else'black')
        fig.colorbar(im,ax=ax,shrink=.65)
fig.savefig(ROOT/'quality_deltas.png',dpi=170);plt.close(fig)
hard=load('hard_training_dynamics.csv');q=load('quality_summary.csv');fig,axes=plt.subplots(3,4,figsize=(14,9),constrained_layout=True)
for y,scene in enumerate(scenes):
    for x,r in enumerate(ratios):
        ax=axes[y,x]
        for seed in seeds:
            z=sorted([z for z in hard if z['scene']==scene and float(z['ratio'])==r and int(z['seed'])==seed],key=lambda z:int(z['step']));b=float(z[0]['psnr']);ax.plot([int(a['step'])for a in z],[float(a['psnr'])-b for a in z],label=f'PL {seed}')
            st=float(next(z['psnr']for z in q if z['scene']==scene and float(z['ratio'])==r and z['arm']=='PG-ST'and int(z['seed'])==seed and z['split']=='train'))
            ax.axhline(st-b,ls='--',alpha=.6,label=f'ST final {seed}')
        ax.axhline(0,color='gray',lw=.6);ax.set_title(f'{scene} {int(r*100)}%');ax.set_xlabel('step');ax.set_ylabel('Bank hard PSNR - Raw (dB)');ax.legend(fontsize=6)
fig.savefig(ROOT/'hard_training_curves.png',dpi=160);plt.close(fig)
cov=load('gradient_coverage.csv');fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
for ax,scene in zip(axes,scenes):
    for seed in seeds:
        z=[next(z for z in cov if z['scene']==scene and float(z['ratio'])==r and int(z['seed'])==seed)for r in ratios]
        ax.plot([1,5,10,15],[float(a['ever_nonzero_fraction'])for a in z],'o-',label=f'PL {seed}');ax.plot([1,5,10,15],[float(a['legacy_ever_nonzero_fraction'])for a in z],'o--',label=f'ST {seed}')
    ax.set_title(scene);ax.set_xlabel('budget (%)');ax.set_ylabel('ever nonzero selection / ST gradient fraction');ax.set_ylim(0,1.05);ax.legend(fontsize=8)
fig.savefig(ROOT/'gradient_coverage.png',dpi=170);plt.close(fig)
prom=load('promotion_comparison.csv');fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
for ax,scene in zip(axes,scenes):
    for seed in seeds:
        z=[next(z for z in prom if z['scene']==scene and float(z['ratio'])==r and int(z['seed'])==seed)for r in ratios]
        ax.plot([1,5,10,15],[float(a['PL_Raw_percentile_p90'])for a in z],'o-',label=f'PL {seed}');ax.plot([1,5,10,15],[float(a['ST_Raw_percentile_p90'])for a in z],'o--',label=f'ST {seed}')
    ax.set_title(scene);ax.set_xlabel('budget (%)');ax.set_ylabel('promoted rows: p90 Raw rank percentile');ax.legend(fontsize=8)
fig.savefig(ROOT/'raw_rank_promotion.png',dpi=170);plt.close(fig)

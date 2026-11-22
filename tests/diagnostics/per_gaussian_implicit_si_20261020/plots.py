"""静态研究图；仅显示已完成且固定的实验结果。"""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def load(n):return list(csv.DictReader(open(ROOT/n)))
scenes=['fr1_room','fr1_xyz','fr2_xyz'];ratios=[.01,.05,.1,.15];seeds=[2026,2027]
d=load('quality_deltas.csv');curves=load('hard_training_dynamics.csv')
fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
for j,ref in enumerate(['A','B','C']):
    for i,seed in enumerate(seeds):
        ax=axes[i,j];a=np.array([[float(next(z['psnr']for z in d if z['scene']==s and float(z['ratio'])==r and z['split']=='eval'and z['reference']==ref and int(z['seed'])==seed))for r in ratios]for s in scenes]);limit=max(abs(a).max(),.001)
        im=ax.imshow(a,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto');ax.set_xticks(range(4),['1%','5%','10%','15%']);ax.set_yticks(range(3),scenes);ax.set_title(f'PG - {dict(A="Raw",B="Fixed h8",C="C")[ref]}, seed {seed} (dB)')
        for y in range(3):
            for x in range(4):ax.text(x,y,f'{a[y,x]:+.3f}',ha='center',va='center',color='white'if abs(a[y,x])>.55*limit else'black')
        fig.colorbar(im,ax=ax,shrink=.7)
fig.savefig(ROOT/'eval_quality_deltas.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(3,4,figsize=(14,9),constrained_layout=True)
for y,scene in enumerate(scenes):
    for x,ratio in enumerate(ratios):
        ax=axes[y,x]
        for seed in seeds:
            z=sorted([z for z in curves if z['scene']==scene and float(z['ratio'])==ratio and int(z['seed'])==seed],key=lambda z:int(z['step']))
            ax.plot([int(a['step'])for a in z],[float(a['train_hard_psnr_minus_Raw'])for a in z],marker='o',label=str(seed))
        ax.axhline(0,color='gray',lw=.8);ax.set_title(f'{scene} {int(ratio*100)}%');ax.set_xlabel('step');ax.set_ylabel('Bank hard PSNR - Raw (dB)');ax.legend()
fig.savefig(ROOT/'training_hard_curves.png',dpi=160);plt.close(fig)
states=json.loads((ROOT/'selected_before_quality.json').read_text())['models'];fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for ax,scene in zip(axes,scenes):
    for seed in seeds:
        v=[states[f'{scene}_PG_{int(r*100)}_seed_{seed}_step_2000']['gradient_top10percent_root_mass_fraction']for r in ratios]
        b=[states[f'{scene}_PG_{int(r*100)}_seed_{seed}_step_2000']['gradient_top10percent_roots_candidate_fraction']for r in ratios]
        ax.plot([1,5,10,15],v,'o-',label=f'gradient mass {seed}');ax.plot([1,5,10,15],b,'--',label=f'candidate share {seed}')
    ax.set_title(scene);ax.set_xlabel('budget (%)');ax.set_ylabel('share in top 10% gradient-mass roots');ax.legend(fontsize=7)
fig.savefig(ROOT/'gradient_spatial_concentration.png',dpi=170);plt.close(fig)

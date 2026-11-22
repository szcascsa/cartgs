"""科学静态图：匹配固定集合结果与全部seed，不选择最佳曲线。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
NAMES=['Reference59TrainEncoder','SpatialFiLM','SpatialOutputKnots']
COLORS=['#666666','#297fb8','#df783b']


def main():
    data=pd.read_csv(ROOT/'quality_by_ratio.csv');out=ROOT/'figures';out.mkdir(exist_ok=True)
    q=data[(data.policy=='same_k')&(data.split=='test')]
    fig,axs=plt.subplots(1,3,figsize=(14,4.1))
    for ax,scene,tiers in zip(axs,['fr1_xyz','fr2_xyz','fr1_xyz'],[['confirm','reference_control'],['fr2_confirm'],['baseline_sets']]):
        pool=q[(q.scene==scene)&q.tier.isin(tiers)]
        for name,color in zip(NAMES,COLORS):
            g=pool[pool.config==name].groupby('ratio').psnr.agg(['mean','min','max']).reset_index()
            if g.empty:continue
            ax.plot(g.ratio*100,g['mean'],'o-',color=color,label=name,markersize=4)
            ax.fill_between(g.ratio*100,g['min'],g['max'],color=color,alpha=.1)
        family='Baseline59 sets' if tiers==['baseline_sets'] else 'R3 sets'
        if family=='R3 sets':
            source=3100 if scene=='fr1_xyz' else 4100
            ref=pd.read_csv(ROOT/'references'/scene/'R3'/str(source)/'test_per_view.csv')
            ref=ref[ref.policy=='same_k'].groupby('ratio').psnr.mean()
            ax.plot(ref.index*100,ref.values,'--',color='#202020',label='R3 reference')
        ax.set(title=scene+' '+family,xlabel='Ratio (%)',ylabel='PSNR (dB)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'quality_ratios.png',dpi=180);plt.close(fig)
    frames=[]
    for p in (ROOT/'results').glob('*/*/*/*/validation_curve.csv'):
        tier,scene,name,seed=p.parts[-5:-1]
        if scene!='fr1_xyz' or tier not in ['confirm','reference_control','long36k']:continue
        if not any(n in name for n in NAMES):continue
        f=pd.read_csv(p);f['config']=name.replace('Long','');f['seed']=int(seed);f['total_step']=f.step+(12000 if tier=='long36k' else 0);frames.append(f)
    if frames:
        curves=pd.concat(frames);fig,ax=plt.subplots(figsize=(7.5,4.3))
        for name,color in zip(NAMES,COLORS):
            for seed,style in [(5100,'-'),(5101,'--')]:
                g=curves[(curves.config==name)&(curves.seed==seed)].groupby('total_step').psnr.mean()
                ax.plot(g.index,g.values,style,color=color,label=name+f' {seed}')
        ax.set(xlabel='Total real photo steps',ylabel='Bank holdout seven-ratio PSNR (dB)');ax.grid(alpha=.2);ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(out/'convergence.png',dpi=180);plt.close(fig)
    print('PLOTS_WRITTEN',flush=True)


if __name__=='__main__':main()

"""真实CSV绑定：逐场景K缩放、归一化与画质Pareto。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

HERE=Path(__file__).resolve().parent
DATA=HERE/'reports/RENDER_AGGREGATE';OUT=HERE/'figures'
COLORS={'F2':'#4E79A7','tile_safe_fast':'#59A14F','tile_safe_bitexp_stop':'#E15759',
    'tile_safe_bitexp':'#F28E2B','tile_safe_bitexp_stop2':'#B279A2'}
LABELS={'F2':'冻结 F2','tile_safe_fast':'快速 exp（近乎无损）','tile_safe_bitexp_stop':'局部多项式＋快速 exp＋0.01 早停',
    'tile_safe_bitexp':'局部多项式＋快速 exp','tile_safe_bitexp_stop2':'0.02 早停（未通过）'}
METHODS=['F2','tile_safe_fast','tile_safe_bitexp_stop']
PRIMARY=[f'local{x:g}' for x in [.1,.2,.3,.4,.5]]


def style():
    for path in ['C:/Windows/Fonts/msyh.ttc','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
        if Path(path).exists():fm.fontManager.addfont(path);plt.rcParams['font.family']=fm.FontProperties(fname=path).get_name();break
    plt.rcParams.update({'font.size':9,'axes.titlesize':11,'axes.unicode_minus':False,'axes.spines.top':False,
        'axes.spines.right':False,'legend.frameon':False,'grid.alpha':.2,'savefig.dpi':450,'svg.fonttype':'none'})


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=450,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)


def main():
    style();OUT.mkdir(exist_ok=True)
    samples=pd.read_csv(DATA/'samples.csv.gz');perf=pd.read_csv(DATA/'PER_SCENE_RATIO_ROUND.csv')
    normalized=pd.read_csv(DATA/'NORMALIZED_SCALING.csv');quality=pd.read_csv(DATA/'QUALITY_GATE.csv')
    primary=pd.read_csv(DATA/'PRIMARY_RATIO_GAIN.csv')
    timing=samples.groupby(['workload','scene','condition','variant']).agg(K=('K','median'),ms=('ms','median'),
        p95=('ms',lambda x:x.quantile(.95))).reset_index()
    timing.to_csv(OUT/'K_TIMING_CURVE_DATA.csv',index=False)
    norm=normalized.groupby(['workload','scene','condition','variant']).agg(K=('K_normalized','median'),
        T=('T_normalized','median'),T25=('T_normalized',lambda x:x.quantile(.25)),T75=('T_normalized',lambda x:x.quantile(.75))).reset_index()
    norm.to_csv(OUT/'NORMALIZED_CURVE_DATA.csv',index=False)
    conditions=[f'local{x:g}' for x in [.01,.03,.05,.1,.2,.3,.4,.5,.7]]+['full']
    for scene in sorted(timing.scene.unique()):
        fig,axes=plt.subplots(2,2,figsize=(9.8,7.1))
        for col,workload in enumerate(['nested','lod']):
            title='固定属性嵌套删点' if workload=='nested' else '真实 LoD 补偿输出'
            for name in METHODS:
                q=timing[(timing.scene==scene)&(timing.workload==workload)&timing.condition.isin(conditions)&(timing.variant==name)].sort_values('K')
                axes[0,col].plot(q.K,q.ms*1000,'o-',ms=3,lw=1.4,color=COLORS[name],label=LABELS[name])
                q=norm[(norm.scene==scene)&(norm.workload==workload)&norm.condition.isin(conditions)&(norm.variant==name)].sort_values('K')
                axes[1,col].plot(q.K,q['T'],'o-',ms=3,lw=1.4,color=COLORS[name])
            axes[0,col].set(title=title,xlabel='实际输入 K',ylabel='同步墙钟 / μs')
            axes[1,col].plot([0,1],[0,1],'--',color='#555555',lw=1,label='理想 T/Tfull = K/Kfull')
            axes[1,col].set(xlabel='K / Kfull',ylabel='T / Tfull',xlim=(0,1.03))
            for ax in axes[:,col]:ax.grid(True)
        handles,labels=axes[0,0].get_legend_handles_labels();h,l=axes[1,0].get_legend_handles_labels()
        fig.legend(handles+h,labels+l,loc='lower center',ncol=2,bbox_to_anchor=(.5,-.015),fontsize=8)
        fig.suptitle(scene+'｜两轮独立进程，10个移动／跳转视角',fontsize=12)
        fig.tight_layout(rect=[0,.07,1,.96]);save(fig,'SCENE_'+scene.replace('/','__'))
    fig,axes=plt.subplots(1,2,figsize=(9.6,4.1))
    for ax,workload in zip(axes,['nested','lod']):
        for name in METHODS:
            sub=norm[(norm.workload==workload)&norm.condition.isin(conditions)&(norm.variant==name)]
            q=sub.groupby('condition').agg(K=('K','median'),T=('T','median'),T25=('T',lambda x:x.quantile(.25)),T75=('T',lambda x:x.quantile(.75))).sort_values('K')
            ax.plot(q.K,q['T'],'o-',ms=3,color=COLORS[name],label=LABELS[name]);ax.fill_between(q.K,q.T25,q.T75,color=COLORS[name],alpha=.10)
        ax.plot([0,1],[0,1],'--',color='#555555',lw=1,label='理想线')
        ax.set(title='固定属性嵌套删点' if workload=='nested' else '真实 LoD',xlabel='实际 K / Full K',ylabel='耗时 / Full耗时',xlim=(0,1.03));ax.grid(True)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,bbox_to_anchor=(.5,-.1),fontsize=8)
    fig.suptitle('32组场景／模式的归一化缩放｜带状区域为场景间四分位区间',fontsize=11)
    fig.tight_layout(rect=[0,.04,1,.94]);save(fig,'NORMALIZED_SCALING_OVERVIEW')
    fig,ax=plt.subplots(figsize=(8.4,4.1));ratio_values=[.01,.03,.05,.1,.2,.3,.4,.5,.7,1.]
    subset=perf[(perf.workload=='lod')&perf.condition.isin(conditions)]
    for name in METHODS[1:]:
        med=[];lo=[];hi=[]
        for c in conditions:
            q=subset[(subset.variant==name)&(subset.condition==c)].speedup
            med.append(float(np.exp(np.log(q).mean())));lo.append(q.quantile(.1));hi.append(q.quantile(.9))
        ax.plot(ratio_values,med,'o-',color=COLORS[name],label=LABELS[name]);ax.fill_between(ratio_values,lo,hi,color=COLORS[name],alpha=.1)
    ax.axhline(1,color='#555555',ls='--',lw=1);ax.set(xlabel='名义 Ratio（1 为 Full）',ylabel='相对 F2 加速倍数',title='真实 LoD：两轮、32组等权几何平均；阴影为10%–90%分位');ax.grid(True);ax.legend(fontsize=8)
    save(fig,'RENDER_SPEEDUP_BY_RATIO')
    pareto=quality.merge(primary,on='variant');pareto.to_csv(OUT/'PARETO_DATA.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for _,row in pareto.iterrows():
        name=row.variant
        for ax,col in zip(axes,['mean_loss','worst_condition_loss']):
            ax.scatter(row[col],row.speedup,s=65,c=COLORS[name],marker='o' if row.passed else 'X',label=LABELS[name])
            ax.annotate(f'{row.speedup:.3f}×',(row[col],row.speedup),xytext=(4,5),textcoords='offset points',fontsize=8)
    for ax,cut,col in zip(axes,[.2,.5],['全部条件等权平均 PSNR 下降 / dB','最差场景×条件平均 PSNR 下降 / dB']):
        ax.axvline(cut,ls='--',lw=1,color='#555555',label=f'门限 {cut:g} dB');ax.grid(True)
        ax.set(xlabel=col,ylabel='0.1–0.5纯渲染加速倍数')
    axes[0].set_xlim(-.018,.215);axes[1].set_xlim(-.025,.75)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h[:4],l[:4],loc='lower center',ncol=2,bbox_to_anchor=(.5,-.10),fontsize=8)
    fig.suptitle('速度—画质 Pareto｜负 PSNR 下降表示测试均值升高，不保证逐帧升高',fontsize=11)
    fig.tight_layout(rect=[0,.05,1,.93]);save(fig,'SPEED_QUALITY_PARETO')
    manifests=[]
    for p in [DATA/'samples.csv.gz',DATA/'PER_SCENE_RATIO_ROUND.csv',DATA/'NORMALIZED_SCALING.csv',DATA/'QUALITY_GATE.csv',DATA/'PRIMARY_RATIO_GAIN.csv']:
        with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
        manifests.append(dict(path=str(p),sha256=h,kind='真实测量',script=str(Path(__file__).resolve())))
    (OUT/'DATA_MANIFEST.json').write_text(json.dumps(manifests,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'data-manifest.md').write_text('所有图均来自真实冻结输入的双轮测量。固定属性删点与真实LoD分开绘制；无渲染图片、无模拟计时。\n\n输入与SHA256见 DATA_MANIFEST.json。每场景10个当前帧，含6个连续视角与4个跳转视角。\n',encoding='utf-8')
    print('RENDER_PLOTS_COMPLETE',len(list(OUT.glob('*.png'))))


if __name__=='__main__':main()

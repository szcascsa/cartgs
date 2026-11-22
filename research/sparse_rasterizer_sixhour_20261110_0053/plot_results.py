"""真实逐样本计时的性能曲线；线性Ratio坐标，保留反例和实际K。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE=Path(__file__).resolve().parent
COLORS={'A':'#777777','B':'#EE7733','C':'#009988','G':'#0077BB'}
LABELS={'A':'原入口 A','B':'上一版 B','C':'新版 CUDA C','G':'新版 Graph G'}


def save(fig,path):
    fig.savefig(str(path)+'.png',dpi=450,bbox_inches='tight')
    fig.savefig(str(path)+'.svg',bbox_inches='tight');plt.close(fig)


def main():
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
        'axes.grid':True,'grid.alpha':.2,'legend.frameon':False,'svg.fonttype':'none'})
    out=HERE/'figures';(out/'scenes').mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(HERE/'RASTER_AND_PIPELINE_RESULTS.csv');gate=pd.read_csv(HERE/'FULL_GATES.csv')
    rounds=json.loads((HERE/'FINAL_VERIFICATION.json').read_text())['formal_names'];round_label=' + '.join(rounds)
    scene_order=[s['id'] for s in json.loads((HERE/'reports/SCENES.json').read_text())]
    sources=[];links=[]
    for scene in scene_order:
        rows=data[data.scene==scene];N=int(rows.N.iloc[0]);local=rows[rows.condition.str.startswith('local')].copy()
        local['ratio']=local.condition.str[5:].astype(float)
        fig,axes=plt.subplots(2,2,figsize=(10.5,6.9),layout='constrained')
        for axis,scope,title in [(axes[0,0],'raster','纯 Rasterizer 同步延迟'),(axes[0,1],'pipeline','每帧重新准备＋渲染')]:
            for name in COLORS:
                series=local[(local.scope==scope)&(local.variant==name)].sort_values('ratio')
                axis.plot(series.ratio,series.median_ms,marker='o',markersize=3.2,lw=1.3,color=COLORS[name],label=LABELS[name])
            full=rows[(rows.condition=='full')&(rows.scope==scope)&(rows.variant=='G')].median_ms.iloc[0]
            axis.axhline(full,lw=.9,ls='--',color=COLORS['G'],alpha=.65,label='G Full（未补偿）')
            axis.set(title=title,xlabel='请求 Ratio（线性坐标）',ylabel='中位延迟 / ms',xlim=(0,.72));axis.legend(fontsize=7.5)
        for name in ['C','G']:
            for scope,style in [('raster','-'),('pipeline','--')]:
                series=local[(local.scope==scope)&(local.variant==name)].sort_values('ratio')
                axes[1,0].plot(series.ratio,series.speedup_B,ls=style,marker='o',markersize=3,color=COLORS[name],
                    label=LABELS[name]+(' 渲染' if scope=='raster' else ' 总延迟'))
        axes[1,0].axhline(1,color='#777777',lw=.9);axes[1,0].set(title='本轮新增收益（相对 B）',xlabel='请求 Ratio',ylabel='加速倍数',xlim=(0,.72))
        axes[1,0].legend(fontsize=7.5)
        k=local[(local.scope=='raster')&(local.variant=='G')].sort_values('ratio')
        axes[1,1].plot(k.ratio,k.K_median,color=COLORS['G'],marker='o',ms=3,label='实际 K 中位数')
        axes[1,1].fill_between(k.ratio,k.K_min,k.K_max,color=COLORS['G'],alpha=.16,label='各视角 K 范围')
        axes[1,1].set(title=f'实际输入点数（整图 N={N:,}）',xlabel='请求 Ratio',ylabel='Gaussian 数量',xlim=(0,.72));axes[1,1].legend(fontsize=7.5)
        fig.suptitle(scene+'  |  两轮独立配对；每帧重新准备；阴影为视角点数范围',fontsize=11)
        name=scene.replace('/','__');target=out/'scenes'/name;save(fig,target)
        links.append(f'- **{scene}，N={N:,}**：[曲线](scenes/{name}.png) · [SVG](scenes/{name}.svg)')
        sources.append(dict(figure='scenes/'+name,data='RASTER_AND_PIPELINE_RESULTS.csv',kind='真实测量',source=round_label,script='plot_results.py'))
    # 全24组Full：两轮各自的门限结论另在CSV，不用汇总均值掩盖失败。
    fig,axes=plt.subplots(1,2,figsize=(12,8.5),gridspec_kw={'width_ratios':[1.1,1]},layout='constrained')
    full=data[(data.condition=='full')&(data.scope=='raster')].set_index(['scene','variant'])
    y=np.arange(len(scene_order))
    for j,name in enumerate(COLORS):
        values=np.array([full.loc[(scene,name),'median_ms'] for scene in scene_order])
        axes[0].plot(values,y+(j-1.5)*.13,marker='o',ms=3,lw=.6,color=COLORS[name],label=LABELS[name])
    labels=[scene+'  N='+str(int(full.loc[(scene,'G'),'N'])) for scene in scene_order]
    axes[0].set(yticks=y,yticklabels=labels,xlabel='Full 中位延迟 / ms',title='所有场景 Full 速度');axes[0].invert_yaxis();axes[0].legend(fontsize=8)
    g=gate[(gate.variant=='G')&(gate.scope=='raster')]
    for offset,(ratio,color,label) in enumerate([('speedup_A',COLORS['G'],'相对原入口 A'),('speedup_B',COLORS['B'],'相对上一版 B')]):
        for i,scene in enumerate(scene_order):
            values=g[g.scene==scene][ratio];axes[1].plot(values,[i+(offset-.5)*.16]*len(values),'o',ms=3,color=color,label=label if i==0 else None)
    axes[1].axvline(1,color='#777777',ls='--',lw=.8);axes[1].set(yticks=y,yticklabels=[],xlabel='加速倍数（每个点对应一轮）',title='逐场景、逐轮 Full 门限');axes[1].invert_yaxis();axes[1].legend(fontsize=8)
    save(fig,out/'FULL_ALL_SCENES')
    # 低Ratio相对B新增收益：所有场景展示，包括任何小于1的条目。
    fig,axes=plt.subplots(1,2,figsize=(12,8.3),layout='constrained')
    for axis,scope in zip(axes,['raster','pipeline']):
        for j,(condition,color) in enumerate([('local0.01','#0077BB'),('local0.05','#009988'),('local0.1','#EE7733')]):
            values=[data[(data.scene==scene)&(data.scope==scope)&(data.variant=='G')&(data.condition==condition)].speedup_B.iloc[0] for scene in scene_order]
            axis.plot(values,y+(j-1)*.17,'o',ms=3.5,color=color,label=condition[5:])
        axis.axvline(1,color='#777777',ls='--',lw=.8);axis.set(yticks=y,yticklabels=scene_order if scope=='raster' else [],
            xlabel='本轮相对上一版 B 的加速倍数',title='纯 Rasterizer' if scope=='raster' else '每帧准备＋渲染');axis.invert_yaxis();axis.legend(title='Ratio',fontsize=8)
    save(fig,out/'LOW_RATIO_ALL_SCENES')
    # 逐视角的真实点数/延迟关系，颜色区分实际Tile实例负载，避免只以K解释时间。
    raw=pd.concat([pd.read_csv(HERE/'reports'/name/'samples.csv') for name in rounds],ignore_index=True)
    render=raw[(raw.variant=='G')&(raw.scope=='raster')].groupby(['scene','condition','frame']).wall_ms.median().reset_index()
    work=pd.read_csv(HERE/'TILE_WORKLOADS.csv');joined=work.merge(render,on=['scene','condition','frame'],validate='one_to_one')
    positive=joined[joined.K>0]
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
    scatter=axes[0].scatter(positive.K,positive.wall_ms,c=np.log10(positive.tile_instances.clip(lower=1)),s=7,cmap='viridis',alpha=.6,rasterized=True)
    axes[0].set(xscale='log',xlabel='实际输入 K（对数坐标）',ylabel='G 渲染中位延迟 / ms',title='K 无法单独决定延迟');fig.colorbar(scatter,ax=axes[0],label='log10(Tile 实例数)')
    for dataset,color in [('Replica','#0077BB'),('TUM_RGBD','#009988'),('VECtor','#EE7733')]:
        points=positive[positive.scene.str.startswith(dataset)];axes[1].scatter(points.tile_instances.clip(lower=1),points.wall_ms,s=7,color=color,alpha=.5,label=dataset,rasterized=True)
    axes[1].set(xscale='log',xlabel='实际 Tile 实例数（对数坐标）',ylabel='G 渲染中位延迟 / ms',title='屏幕覆盖工作量与延迟');axes[1].legend(fontsize=8)
    save(fig,out/'COUNT_AND_FOOTPRINT')
    (out/'SCENE_INDEX.md').write_text('# 逐场景性能曲线\n\n'+chr(10).join(links)+'\n',encoding='utf-8')
    sources.extend([dict(figure=name,data='RASTER_AND_PIPELINE_RESULTS.csv + FULL_GATES.csv + TILE_WORKLOADS.csv',kind='真实测量',source=round_label,script='plot_results.py') for name in ['FULL_ALL_SCENES','LOW_RATIO_ALL_SCENES','COUNT_AND_FOOTPRINT']])
    pd.DataFrame(sources).to_csv(out/'DATA_MANIFEST.csv',index=False,encoding='utf-8-sig')
    (out/'data-manifest.md').write_text('# 图表数据清单\n\n全部图表来自真实计时和计时外工作量诊断；无合成结果、无平滑处理。正式原始数据：'+round_label+'。PNG 450 DPI，同时输出SVG。\n\n'+
        '| 图表 | 数据 | 来源 | 脚本 |\n|---|---|---|---|\n'+chr(10).join(f"| {row['figure']} | {row['data']} | {row['source']} | {row['script']} |" for row in sources),encoding='utf-8')
    print('FIGURES_COMPLETE',len(sources),'PNG + SVG')


if __name__=='__main__':main()

"""最终冻结方案曲线；纯渲染与准备使用各自独立配对批次，不跨批次构造加速比。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE=Path(__file__).resolve().parent
COLORS={'A':'#777777','B':'#EE7733','C':'#009988','G':'#0077BB','F':'#AA4499','Q':'#009988','R':'#332288'}
LABELS={'A':'原入口 A','B':'上一版 B','C':'本轮 CUDA C','G':'渲染 Graph G','F':'G＋固定内参 F','Q':'相机 Graph Q','R':'最终异步方案 R'}


def save(fig,path):
    fig.savefig(str(path)+'.png',dpi=450,bbox_inches='tight');fig.savefig(str(path)+'.svg',bbox_inches='tight');plt.close(fig)


def main():
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':.2,'legend.frameon':False,'axes.unicode_minus':False,'svg.fonttype':'none'})
    out=HERE/'figures';out.mkdir(exist_ok=True);(out/'final_scenes').mkdir(exist_ok=True)
    accepted=HERE/'CAMERA_ASYNC_FINAL_VERIFICATION.json'
    async_passed=accepted.exists() and json.loads(accepted.read_text())['passed']
    final_variant='R' if async_passed else 'Q'
    pipeline_name='FINAL_ASYNC_PIPELINE_RESULTS.csv' if async_passed else 'FINAL_CAMERA_GRAPH_PIPELINE_RESULTS.csv'
    pipeline_variants=['A','B','G','Q','R'] if async_passed else ['A','B','G','F','Q']
    final_rounds='/'.join(json.loads(accepted.read_text())['formal_names']) if async_passed else 'camera_final01/camera_final02'
    raster=pd.read_csv(HERE/'RASTERIZER_RESULTS.csv');pipeline=pd.read_csv(HERE/pipeline_name)
    scenes=json.loads((HERE/'reports/SCENES.json').read_text());links=[];manifest=[]
    for scene in scenes:
        name=scene['id'];slug=scene['slug'];N=int(raster[raster.scene==name].N.iloc[0])
        r=raster[(raster.scene==name)&raster.condition.str.startswith('local')].copy();r['ratio']=r.condition.str[5:].astype(float)
        p=pipeline[(pipeline.scene==name)&pipeline.condition.str.startswith('local')].copy();p['ratio']=p.condition.str[5:].astype(float)
        fig,axes=plt.subplots(2,2,figsize=(10.5,7),layout='constrained')
        for axis,data,variants,title in [(axes[0,0],r,['A','B','C','G'],'纯渲染（复用已验收两轮）'),
            (axes[0,1],p,pipeline_variants,'相机＋准备＋渲染（最终两轮）')]:
            for variant in variants:
                d=data[data.variant==variant].sort_values('ratio')
                axis.plot(d.ratio,d.median_ms,color=COLORS[variant],marker='o',markersize=3,label=LABELS[variant],lw=1.2)
            full_source,full_variant=(raster,'G') if axis==axes[0,0] else (pipeline,final_variant)
            full_ms=full_source[(full_source.scene==name)&(full_source.variant==full_variant)&(full_source.condition=='full')].median_ms.iloc[0]
            axis.axhline(full_ms,color=COLORS[full_variant],lw=.9,ls='--',alpha=.7,label=full_variant+' Full（未补偿）')
            axis.set(title=title,xlabel='请求 Ratio（线性坐标）',ylabel='中位延迟 / ms',xlim=(0,.72));axis.legend(fontsize=7)
        for variant,source,style,label in [('G',r,'-','纯渲染 G'),(final_variant,p,'--','完整延迟 '+final_variant)]:
            d=source[source.variant==variant].sort_values('ratio')
            axes[1,0].plot(d.ratio,d.speedup_B,color=COLORS[variant],marker='o',ms=3,ls=style,label=label+' 相对 B')
        axes[1,0].axhline(1,color='#777777',ls=':',lw=.8)
        axes[1,0].set(title='本轮新增加速（各自同批次配对）',xlabel='请求 Ratio',ylabel='加速倍数',xlim=(0,.72));axes[1,0].legend(fontsize=7.5)
        k=r[r.variant=='G'].sort_values('ratio');axes[1,1].plot(k.ratio,k.K_median,color=COLORS['G'],marker='o',ms=3,label='实际 K 中位数')
        axes[1,1].fill_between(k.ratio,k.K_min,k.K_max,color=COLORS['G'],alpha=.15,label='视角 K 范围')
        axes[1,1].set(title=f'整图 N={N:,}；候选池随视角变化',xlabel='请求 Ratio',ylabel='Gaussian 数量',xlim=(0,.72));axes[1,1].legend(fontsize=7.5)
        fig.suptitle(name+'  |  FP32 RGB/radii 逐位一致；每帧重新准备',fontsize=11)
        save(fig,out/'final_scenes'/slug)
        links.append(f'- **{name}，N={N:,}**：[PNG](final_scenes/{slug}.png) · [SVG](final_scenes/{slug}.svg)')
        manifest.append(dict(figure='final_scenes/'+slug,data='RASTERIZER_RESULTS.csv + '+pipeline_name,
            source='formal02/formal04；'+final_rounds,kind='真实测量',script='plot_final_results.py'))
    order=[s['id'] for s in scenes];y=np.arange(len(order))
    fig,axes=plt.subplots(1,2,figsize=(12,8.3),layout='constrained')
    for axis,data,variant,title in [(axes[0],raster,'G','纯渲染 G 相对上一版 B'),(axes[1],pipeline,final_variant,'完整延迟 '+final_variant+' 相对上一版 B')]:
        for index,(condition,color) in enumerate([('local0.01','#0077BB'),('local0.05','#009988'),('local0.1','#EE7733')]):
            values=[data[(data.scene==name)&(data.variant==variant)&(data.condition==condition)].speedup_B.iloc[0] for name in order]
            axis.plot(values,y+(index-1)*.17,'o',color=color,ms=3.5,label=condition[5:])
        axis.axvline(1,color='#777777',ls='--',lw=.8);axis.set(yticks=y,yticklabels=order if axis==axes[0] else [],
            xlabel='本轮新增加速倍数',title=title);axis.invert_yaxis();axis.legend(title='Ratio',fontsize=8)
    save(fig,out/'FINAL_LOW_RATIO_ALL_SCENES')
    gate=pd.read_csv(HERE/('FINAL_ASYNC_FULL_GATES.csv' if async_passed else 'FINAL_CAMERA_GRAPH_FULL_GATES.csv'))
    fig,axes=plt.subplots(1,2,figsize=(12,8.5),layout='constrained')
    for axis,reference,title in [(axes[0],'A','最终 Full 完整延迟：相对原入口 A'),(axes[1],'B','最终 Full 完整延迟：相对上一版 B')]:
        for index,name in enumerate(order):
            d=gate[gate.scene==name]
            axis.plot(d['speedup_'+reference],[index]*len(d),'o',ms=3.5,color=COLORS[final_variant])
        axis.axvline(1,color='#777777',ls='--',lw=.8);axis.set(yticks=y,yticklabels=order if axis==axes[0] else [],xlabel='加速倍数；每个点为一轮',title=title);axis.invert_yaxis()
    save(fig,out/'FINAL_FULL_GATES')
    # “减少点数”分别检查纯渲染和完整延迟，所有反例保留。
    old=pd.read_csv(HERE/'COUNT_VS_SPEED_PER_VIEW.csv');old=old[(old.scope=='raster')&old.condition.isin(['local0.01','local0.05','local0.1'])]
    new=pd.read_csv(HERE/('FINAL_ASYNC_COUNT_VS_FULL.csv' if async_passed else 'FINAL_CAMERA_GRAPH_COUNT_VS_FULL.csv'));new=new[new.condition.isin(['local0.01','local0.05','local0.1'])]
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for axis,data,title in [(axes[0],old,'纯渲染：低 Ratio / 同视角 Full'),(axes[1],new,'完整延迟：低 Ratio / 同视角 Full')]:
        groups=[data[data.condition==condition].speedup_vs_own_full.values for condition in ['local0.01','local0.05','local0.1']]
        parts=axis.boxplot(groups,labels=['1%','5%','10%'],showfliers=True,patch_artist=True)
        for patch,color in zip(parts['boxes'],['#0077BB','#009988','#EE7733']):patch.set(facecolor=color,alpha=.5)
        axis.axhline(1,color='#777777',ls='--',lw=.8);axis.set(title=title,xlabel='请求 Ratio',ylabel='相对自身 Full 加速倍数')
    save(fig,out/'FINAL_COUNT_VS_FULL')
    (out/'FINAL_SCENE_INDEX.md').write_text('# 最终逐场景性能曲线\n\n'+chr(10).join(links)+'\n',encoding='utf-8')
    manifest.extend([dict(figure=name,data='正式性能CSV及门限CSV',source='各自两轮配对；无跨批次配对',kind='真实测量',script='plot_final_results.py')
        for name in ['FINAL_LOW_RATIO_ALL_SCENES','FINAL_FULL_GATES','FINAL_COUNT_VS_FULL']])
    pd.DataFrame(manifest).to_csv(out/'FINAL_DATA_MANIFEST.csv',index=False,encoding='utf-8-sig')
    print('FINAL_FIGURES_COMPLETE',len(manifest),'PNG + SVG')


if __name__=='__main__':main()

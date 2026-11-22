"""来自正式实测CSV的科研性能图；合成压力数据单独标记。"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DATA=HERE/'reports/AGGREGATE'
OUT=HERE/'figures'
COLORS=['#4E79A7','#F28E2B','#59A14F','#E15759']
for font in (Path('C:/Windows/Fonts/msyh.ttc'),Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')):
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name();break
plt.rcParams.update({'font.size':9,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.18,'legend.frameon':False,'axes.unicode_minus':False,'svg.fonttype':'none'})
MANIFEST=[]


def save(fig,name,data,description):
    fig.savefig(OUT/(name+'.png'),dpi=450,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    MANIFEST.append(dict(figure=name,data=data,type=description,outputs=[name+'.png',name+'.svg']))


def ratios(table):
    table=table[table.condition.str.startswith('local')].copy()
    table['ratio']=table.condition.str.removeprefix('local').astype(float)
    return table.sort_values('ratio')


def main():
    OUT.mkdir(exist_ok=True)
    conditions=pd.read_csv(DATA/'CONDITIONS.csv')
    scales=pd.read_csv(DATA/'COUNT_SCALING_SUMMARY.csv')
    rejects=pd.read_csv(DATA/'SUBTILE_REJECTION.csv')
    fig,axes=plt.subplots(2,2,figsize=(11.4,8.1),layout='constrained')
    for variant,color,label in [('common',COLORS[0],'精确公共乘积'),('exit8',COLORS[1],'每8成员早停')]:
        a=ratios(conditions[conditions.variant==variant])
        axes[0,0].plot(a.ratio,100*(1-1/a.speedup),marker='o',ms=4,color=color,label=label)
    axes[0,0].axhline(0,c='#777777',lw=.8);axes[0,0].set(xlabel='局部预算 Ratio',ylabel='相对 F2 的延迟降低 (%)',
        title='32组 × 两轮：本轮新增收益不足1%');axes[0,0].legend()
    a=ratios(scales)
    for key,color,label in [('K_vs_full',COLORS[0],'输入 K'),('interaction_vs_full',COLORS[1],'Alpha有效交互'),
                            ('latency_vs_full',COLORS[2],'纯渲染延迟')]:
        axes[0,1].plot(a.ratio,a[key],marker='o',ms=4,c=color,label=label)
    axes[0,1].set(xlabel='局部预算 Ratio',ylabel='相对 Full 的比例（逐组比值中位数）',title='点数下降，像素工作量未同比下降')
    axes[0,1].legend()
    for side,color in [(4,COLORS[0]),(8,COLORS[1])]:
        a=ratios(rejects[rejects.side==side])
        axes[1,0].plot(a.ratio,100*a.weighted_reject_fraction,marker='o',ms=4,c=color,label=f'{side}×{side}子块')
    axes[1,0].set(xlabel='局部预算 Ratio',ylabel='可保守筛除的子块 (%)',title='开发集诊断：1%下仅筛除2.51% / 1.21%')
    axes[1,0].legend()
    attr=DATA/'ATTRIBUTION_SUMMARY.csv'
    if attr.exists():
        a=ratios(pd.read_csv(attr))
        for key,color,label in [('rebuilt_vs_F2',COLORS[1],'重新编译原逻辑 / F2'),
                                ('common_vs_rebuilt',COLORS[0],'公共乘积 / 重新编译原逻辑')]:
            axes[1,1].plot(a.ratio,100*(1-1/a[key]),marker='o',c=color,label=label)
        axes[1,1].set(xlabel='局部预算 Ratio',ylabel='延迟降低 (%)',title='追加归因：排除同二进制布局差异')
        axes[1,1].axhline(0,c='#777777',lw=.8);axes[1,1].legend(fontsize=8)
    else:
        axes[1,1].axis('off')
    fig.suptitle('低点数、大 footprint 优化：实测收益与工作量边界',fontsize=15)
    save(fig,'RESULT_OVERVIEW',['CONDITIONS.csv','COUNT_SCALING_SUMMARY.csv','SUBTILE_REJECTION.csv','ATTRIBUTION_SUMMARY.csv'],'真实实验')
    perf=pd.read_csv(DATA/'PER_SCENE_RATIO_ROUND.csv')
    mean=perf.groupby(['scene','condition','variant']).agg(p50_ms=('p50_ms','mean'),N=('N','first'),K_p50=('K_p50','first')).reset_index()
    scenes=sorted(mean.scene.unique())
    for page in range(2):
        fig,axes=plt.subplots(4,4,figsize=(14.2,12),layout='constrained')
        for ax,scene in zip(axes.flat,scenes[page*16:page*16+16]):
            part=mean[mean.scene==scene];n=int(part.N.iloc[0])
            for variant,color,label in [('F2','#666666','F2'),('common',COLORS[0],'精确公共乘积'),('exit8',COLORS[1],'早停8')]:
                a=ratios(part[part.variant==variant])
                ax.plot(a.ratio,a.p50_ms*1000,c=color,label=label,lw=1.1,marker='o',ms=2.3)
                full=part[(part.variant==variant)&(part.condition=='full')]
                ax.scatter([1.],full.p50_ms*1000,c=color,marker='D',s=17)
            ax.set_title(scene+'\nN='+f'{n:,}',fontsize=9)
            ax.set(xlabel='局部 Ratio；◆为全图 Full',ylabel='纯渲染延迟 (μs)',xticks=[0,.1,.3,.5,.7,1])
            ax.tick_params(labelsize=8)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f'全部场景逐组结果 {page+1}/2：两轮 p50 的均值，Full候选域不同',fontsize=14)
        save(fig,f'SCENE_LATENCY_{page+1:02d}',['PER_SCENE_RATIO_ROUND.csv'],'真实实验')
        fig,axes=plt.subplots(4,4,figsize=(14.2,12),layout='constrained')
        for ax,scene in zip(axes.flat,scenes[page*16:page*16+16]):
            part=mean[mean.scene==scene]
            for variant,color,label in [('F2','#666666','F2'),('common',COLORS[0],'精确公共乘积'),('exit8',COLORS[1],'早停8')]:
                a=ratios(part[part.variant==variant]).sort_values('K_p50')
                ax.plot(a.K_p50,a.p50_ms*1000,c=color,label=label,lw=1.1,marker='o',ms=2.3)
                full=part[(part.variant==variant)&(part.condition=='full')]
                ax.scatter(full.K_p50,full.p50_ms*1000,c=color,marker='D',s=17)
            ax.set_title(scene+'\nN='+f'{int(part.N.iloc[0]):,}',fontsize=9)
            ax.set(xlabel='输入 K（条件内中位数）；◆ Full',ylabel='纯渲染延迟 (μs)')
            ax.ticklabel_format(axis='x',style='sci',scilimits=(4,4));ax.tick_params(labelsize=8)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f'实际Gaussian点数与延迟 {page+1}/2：线性坐标，不预设正比关系',fontsize=14)
        save(fig,f'SCENE_COUNT_LATENCY_{page+1:02d}',['PER_SCENE_RATIO_ROUND.csv'],'真实实验')
    modules=pd.read_csv(HERE/'reports/PROFILE/MODULES.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.5),layout='constrained')
    module_groups={'输入/预处理/清理':['input','preprocess','clear'],'Depth排序':['sort'],
                   'Footprint任务':['footprint'],'成员压缩':['compact'],'逐像素合成':['pixel']}
    palette=['#BAB0AC','#B07AA1','#F28E2B','#59A14F','#4E79A7']
    for ax,scene in zip(axes,['Replica__office2__rgbd','VECtor__corner-slow__stereo']):
        part=modules[(modules.scene==scene)&(modules.variant=='F2')]
        ticks=['local0.01','local0.05','local0.1','local0.3','full'];bottom=np.zeros(len(ticks))
        for (name,members),color in zip(module_groups.items(),palette):
            series=part[part.module.isin(members)].groupby('condition').us_per_render.sum().reindex(ticks).fillna(0)
            ax.bar(np.arange(len(ticks)),series,bottom=bottom,color=color,label=name,width=.66);bottom+=series.to_numpy()
        ax.set(title=scene.replace('__','/'),ylabel='Nsight Kernel累计时间 (μs)',xticks=range(len(ticks)),xticklabels=['1%','5%','10%','30%','Full'])
    axes[1].legend(loc='upper left',fontsize=8)
    fig.suptitle('真实热点：像素合成占主导；Profile时间不代替独占墙钟',fontsize=13)
    save(fig,'KERNEL_BREAKDOWN',['../PROFILE/MODULES.csv'],'真实Profile')
    stress=pd.read_csv(DATA/'SYNTHETIC_STRESS.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.5),layout='constrained')
    for ax,opacity in zip(axes,[.02,.5]):
        part=stress[(stress.width==1280)&(stress.scale==3)&(stress.opacity==opacity)]
        for variant,color,label in [('previous_F2','#666666','F2'),('direct_tile128',COLORS[1],'直接成员遍历'),('exact_common32',COLORS[0],'精确公共乘积')]:
            a=part[part.variant==variant].sort_values('K')
            ax.plot(a.K,a.p50_ms*1000,marker='o',c=color,label=label)
        ax.set(title=f'合成压力：Scale=3，Opacity={opacity}',xlabel='输入 Gaussian 数量 K',ylabel='纯渲染延迟 (μs)',xticks=[1,32,128,512])
        ax.legend(fontsize=8)
    fig.suptitle('合成实测：大 footprint 且强遮挡可提前结束；不能外推到真实场景',fontsize=13)
    save(fig,'SYNTHETIC_FOOTPRINT_STRESS',['SYNTHETIC_STRESS.csv'],'合成输入的真实计时，非规划数据')
    entries=[]
    for item in MANIFEST:
        for source in item['data']:
            path=(DATA/source).resolve()
            if path.exists():entries.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT/'DATA_MANIFEST.json').write_text(json.dumps(dict(figures=MANIFEST,inputs=entries),indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# 图表数据清单','','全部性能数据来自本轮实测。合成压力图使用人工构造输入的真实计时，未使用规划占位数值。',
           'PNG为450 DPI，同时保存SVG；中文字体为系统微软雅黑或Noto CJK。','',
           '|图|数据文件（reports/AGGREGATE下）|性质|生成脚本|','|---|---|---|---|']
    for item in MANIFEST:lines.append('|'+item['figure']+'|'+', '.join(item['data'])+'|'+item['type']+'|plot_results.py|')
    (OUT/'data-manifest.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('FIGURES_COMPLETE',len(MANIFEST))


if __name__=='__main__':main()

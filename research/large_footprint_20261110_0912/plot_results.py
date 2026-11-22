"""真实配对数据作图，450dpi PNG/SVG，保留线性Ratio坐标与所有场景。"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager

HERE=Path(__file__).resolve().parent;OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.is_file():fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':9,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.2,'legend.frameon':False,'svg.fonttype':'none','savefig.dpi':450})
COLORS={'G':'#4E79A7','C':'#59A14F','D':'#F28E2B'}
LABELS={'G':'上一轮 G/R','C':'C：预处理融合任务','D':'D：四 CTA 展开'}
ratios={'local0.01':.01,'local0.03':.03,'local0.05':.05,'local0.1':.1,'local0.3':.3,'local0.7':.7,'full':1.}
d=pd.read_csv(HERE/'reports/RASTER_AND_PIPELINE_RESULTS.csv');a=pd.read_csv(HERE/'reports/AGGREGATE.csv')
paths=[]
def save(fig,name):
    for extension in ['png','svg']:
        path=OUT/(name+'.'+extension);fig.savefig(path,dpi=450,bbox_inches='tight');paths.append(path.name)
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,3.4))
for ax,scope,title in zip(axes,['raster','pipeline'],['纯 Rasterizer','每帧重新准备＋渲染']):
    for name in ['C','D']:
        row=a[(a.scope==scope)&(a.variant==name)&a.condition.isin(ratios)].copy()
        row['ratio']=row.condition.map(ratios);row=row.sort_values('ratio')
        ax.plot(row.ratio,row.geometric_speedup_G,'o-',color=COLORS[name],label=LABELS[name],markersize=4)
    ax.axhline(1,color='#555555',linestyle='--',linewidth=1);ax.set_xlabel('Ratio（1.0 为 Full）')
    ax.set_ylabel('相对上一轮 G/R 的加速比');ax.set_title(title);ax.set_xticks([.01,.1,.3,.5,.7,1.])
axes[0].legend(fontsize=8);fig.suptitle('24 组、两轮配对：本轮新增收益（场景等权几何平均）',fontsize=11)
fig.tight_layout();save(fig,'SUMMARY_SPEEDUP')

fig,axes=plt.subplots(1,2,figsize=(10,3.4))
for ax,scope,title in zip(axes,['raster','pipeline'],['低 Ratio：纯 Rasterizer','低 Ratio：每帧准备＋渲染']):
    for name in ['C','D']:
        row=a[(a.scope==scope)&(a.variant==name)&a.condition.isin(list(ratios)[:4])].copy()
        row['ratio']=row.condition.map(ratios);row=row.sort_values('ratio')
        ax.plot(row.ratio,row.geometric_speedup_G,'o-',color=COLORS[name],label=LABELS[name],markersize=4)
        if name=='C':ax.fill_between(row.ratio,row.min_speedup_G,row.max_speedup_G,color=COLORS[name],alpha=.12,label='C：48个场景轮次范围')
    ax.axhline(1,color='#555555',linestyle='--',linewidth=1);ax.set_xticks([.01,.03,.05,.1]);ax.set_xlabel('Ratio')
    ax.set_ylabel('相对上一轮 G/R 的加速比');ax.set_title(title)
axes[0].legend(fontsize=7);fig.tight_layout();save(fig,'LOW_RATIO_DETAIL')

for dataset,rows,columns,name in [('Replica',4,4,'REPLICA_PER_SCENE'),('other',4,2,'TUM_VECTOR_PER_SCENE')]:
    scenes=[s for s in d.scene.unique() if (s.startswith('Replica/'))==(dataset=='Replica')]
    assert len(scenes)==rows*columns
    fig,axes=plt.subplots(rows,columns,figsize=(columns*3.35,rows*2.7),squeeze=False)
    for ax,scene in zip(axes.ravel(),scenes):
        subset=d[(d.scene==scene)&(d.scope=='raster')&d.variant.isin(['G','C','D'])&d.condition.isin(ratios)]
        for variant in ['G','C','D']:
            row=subset[subset.variant==variant].groupby('condition').median(numeric_only=True).reset_index()
            row['ratio']=row.condition.map(ratios);row=row.sort_values('ratio')
            ax.plot(row.ratio,row.median_ms,'o-',color=COLORS[variant],label=LABELS[variant],markersize=3,linewidth=1.2)
        N=int(subset.N.iloc[0]);K=subset[subset.condition=='local0.01'].K_median.median()
        ax.set_title(scene.replace('TUM_RGBD/','TUM/').replace('freiburg','fr')+f'\nN={N:,}；1% K中位={K:.0f}',fontsize=8)
        ax.set_xlabel('Ratio');ax.set_ylabel('纯渲染延迟（ms）');ax.set_xticks([.01,.1,.3,.7,1.]);ax.tick_params(labelsize=7)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=9)
    fig.tight_layout(rect=[0,0,1,.965]);save(fig,name)

v=pd.read_csv(HERE/'reports/PER_VIEW_RESULTS.csv');w=pd.read_csv(HERE/'reports/WORKLOADS.csv')
scatter=v[(v.variant=='C')&(v.scope=='raster')&(v.condition=='local0.01')].merge(w,on=['scene','condition','frame','motion'],suffixes=('','_diagnostic'))
fig,ax=plt.subplots(figsize=(7.5,4))
for dataset,color in [('Replica','#4E79A7'),('TUM_RGBD','#F28E2B'),('VECtor','#59A14F')]:
    part=scatter[scatter.scene.str.startswith(dataset+'/')]
    ax.scatter(part.bbox_tiles_per_visible,part.speedup_vs_G,s=15,alpha=.55,color=color,label=dataset)
ax.axhline(1,color='#555555',linestyle='--',linewidth=1);ax.set_xlabel('平均每个预处理可见 Gaussian 的 bbox Tile 数')
ax.set_ylabel('C 相对 G 的逐视角加速比');ax.set_title('1% 低预算：footprint 与新增收益（含连续运动和跳转）')
ax.legend();fig.tight_layout();save(fig,'FOOTPRINT_SPEEDUP')

inputs=['RASTER_AND_PIPELINE_RESULTS.csv','AGGREGATE.csv','PER_VIEW_RESULTS.csv','WORKLOADS.csv']
manifest=dict(real_measurements=True,synthetic_planning_data=False,runs=json.loads((HERE/'reports/RESULTS_SUMMARY.json').read_text())['runs'],
    source_csv={n:hashlib.sha256((HERE/'reports'/n).read_bytes()).hexdigest() for n in inputs},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dpi=450,outputs=paths,
    numeric_ratio_axis=True,notes='所有场景保留；图中G为上一轮R准备+G渲染。纯渲染相机输入相同，不含准备。')
(OUT/'DATA_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))

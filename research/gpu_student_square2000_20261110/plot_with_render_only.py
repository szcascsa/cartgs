"""保留已有四条曲线，新增2000方形LoD/Full纯渲染实测。"""
from pathlib import Path
import hashlib,json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties

HERE=Path(__file__).resolve().parent;OUT=HERE/'figures'
paths=[HERE/'reports/SQUARE2000_RESULTS.csv',HERE/'reports/samples.csv']
hashes={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
t=pd.read_csv(paths[0]);samples=pd.read_csv(paths[1])
assert len(t)==24
raw=samples[(samples.width==2000)&(samples.height==2000)&(samples.scope=='raster_event')]
medians=raw.groupby(['scene','condition']).ms.median()
assert len(medians)==24 and raw.groupby(['scene','condition']).size().eq(72).all()
for _,row in t.iterrows():
    np.testing.assert_allclose(row.square_raster_event,medians.loc[row.scene,row.condition],rtol=1e-13,atol=1e-13)
font=Path('C:/Windows/Fonts/msyh.ttc');assert font.is_file()
fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
    'legend.frameon':False,'axes.unicode_minus':False,'svg.fonttype':'none','savefig.dpi':450})
fig,axes=plt.subplots(2,2,figsize=(12,8.6))
scene_order=['Replica/office1/mono','Replica/office2/rgbd','Replica/room0/rgbd','Replica/room2/mono']
plotted=[]
for ax,scene,title in zip(axes.flat,scene_order,['office1 · 单目','office2 · RGB-D','room0 · RGB-D','room2 · 单目']):
    g=t[(t.scene==scene)&(t.condition!='full')].copy();g['ratio']=g.condition.str[5:].astype(float);g=g.sort_values('ratio')
    f=t[(t.scene==scene)&(t.condition=='full')].iloc[0]
    # 原图三条LoD完整曲线与Full完整对照使用相同数据、颜色和线型。
    for field,label,color in [('original_pipeline_wall','1200×680 LoD完整流程','#4E79A7'),
        ('rectangular_pipeline_wall','2000×1133 LoD完整流程','#F28E2B'),
        ('square_pipeline_wall','2000×2000 LoD完整流程','#A23B72')]:
        ax.plot(g.ratio,g[field],'o-',color=color,ms=4,lw=1.8,label=label)
    ax.axhline(f.square_pipeline_wall,color='#6B7280',ls='--',lw=1.5,label='2000×2000 Full完整流程')
    ax.plot(g.ratio,g.square_raster_event,color='#59A14F',marker='s',ls='-.',lw=2,ms=4,label='2000×2000 LoD纯GPU渲染')
    ax.axhline(f.square_raster_event,color='#59A14F',ls='--',lw=1.8,label='2000×2000 Full纯GPU渲染')
    assert len(ax.lines)==6
    for index,field in enumerate(['original_pipeline_wall','rectangular_pipeline_wall','square_pipeline_wall']):
        np.testing.assert_array_equal(ax.lines[index].get_ydata(),g[field].to_numpy())
    np.testing.assert_array_equal(ax.lines[4].get_ydata(),g.square_raster_event.to_numpy())
    for index,value in [(3,f.square_pipeline_wall),(5,f.square_raster_event)]:
        np.testing.assert_array_equal(ax.lines[index].get_ydata(),[value,value])
    plotted.append(dict(scene=scene,curves=6,ratio_values=g.ratio.tolist(),pure_render_ms=g.square_raster_event.tolist(),full_pure_render_ms=f.square_raster_event))
    ax.set_title(f'{title}    N={int(f.N):,}',loc='left',fontsize=12,fontweight='bold',pad=12)
    ax.set_ylim(0,max(f.square_pipeline_wall,g.square_pipeline_wall.max())*1.28);ax.set_xlim(0,.735)
    ticks=[.01,.05,.1,.3,.7];ax.set_xticks(ticks,[f'{r:.0%}' for r in ticks],rotation=35,ha='right')
    ax.set_xlabel('Ratio（线性轴）');ax.set_ylabel('耗时（ms，越低越好）')
    row=g[np.isclose(g.ratio,.1)].iloc[0]
    ax.text(.03,.96,f'10%：完整 {row.square_pipeline_wall:.3f} / 纯渲染 {row.square_raster_event:.3f} ms\nFull：完整 {f.square_pipeline_wall:.3f} / 纯渲染 {f.square_raster_event:.3f} ms',
        transform=ax.transAxes,va='top',fontsize=9,bbox=dict(boxstyle='round,pad=.3',fc='white',ec='#dddddd',alpha=.95))
fig.suptitle('2000×2000 GPU蒸馏版：完整流程与纯渲染耗时',fontsize=17,fontweight='bold',y=.985)
handles,labels=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.945),ncol=2,fontsize=10)
fig.text(.5,.019,'绿色：纯GPU渲染CUDA Event，属性与相机已就绪；不含两网络推理、上传和CPU调度。\n完整流程：含当前相机、候选、两网络、打包、渲染及同步；分项独立计时，不能相加替代完整耗时。\nRTX 3090；18视角 × 双轮；原始实测，无平滑。',ha='center',fontsize=9,color='#555555')
fig.subplots_adjust(left=.08,right=.975,bottom=.15,top=.78,hspace=.56,wspace=.25)
outputs=[]
for ext in ['png','svg']:
    path=OUT/('GPU_STUDENT_SQUARE2000_WITH_RENDER_ONLY.'+ext)
    fig.savefig(path,dpi=450,facecolor='white');outputs.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
plt.close(fig)
for name,sha in hashes.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha
manifest=dict(passed=True,real_data=True,smoothing='none',dpi=450,source_sha256=hashes,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),outputs=outputs,panels=plotted,
    added_curves='2000x2000 LoD and Full raster_event; 72 samples per condition',original_curves_preserved=True)
(OUT/'WITH_RENDER_ONLY_DATA_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'WITH_RENDER_ONLY_DATA_MANIFEST.md').write_text('# 纯渲染曲线数据清单\n\n| Figure | Data file | Real/mock | Source | Script | Outputs |\n|---|---|---|---|---|---|\n| 原图加入方形纯渲染 | reports/SQUARE2000_RESULTS.csv | Real | samples.csv中2000×2000的raster_event，72条/条件中位数 | plot_with_render_only.py | GPU_STUDENT_SQUARE2000_WITH_RENDER_ONLY.png / .svg |\n\n保留原四条曲线，增加绿色LoD纯渲染与Full纯渲染两条曲线。24条绘制曲线逐项核对数据；原CSV哈希未改变。未进行训练、渲染或平滑。纯渲染不含推理/上传/CPU调度；完整流程独立计时。详细SHA及绘制数值见同名JSON。\n',encoding='utf-8')
print(json.dumps(dict(passed=True,panels=4,curves=24,source_unchanged=True,outputs=[p['file'] for p in outputs]),ensure_ascii=False))

"""本轮CPU/GPU分离实测；原始中位数，线性Ratio轴，不叠加独立计时。"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
source=HERE/'reports/CPU_GPU_SEPARATED_RESULTS.csv'
timing=HERE/'reports/TIMING_SUMMARY.csv'
data=pd.read_csv(source);stats=pd.read_csv(timing)
assert len(data)==32 and np.isfinite(data.select_dtypes(include='number')).all().all()
assert data.groupby('scene').ratio.nunique().eq(8).all()
font=Path('C:/Windows/Fonts/msyh.ttc')
assert font.is_file(),'需要可显示中文的微软雅黑字体'
fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
    'legend.frameon':False,'axes.unicode_minus':False,'svg.fonttype':'none','savefig.dpi':450})
COLORS=dict(cpu='#4E79A7',upload='#F28E2B',gpu='#59A14F',pipeline='#A23B72',full='#6B7280')
SCENES=['Replica__office1__mono','Replica__office2__rgbd','Replica__room0__rgbd','Replica__room2__mono']
TITLES=['office1 · 单目','office2 · RGB-D','room0 · RGB-D','room2 · 单目']
outputs=[]
manifest=dict(real_data=True,smoothing='none',aggregation='原正式双轮六视角样本中位数，沿用CSV',
    source_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,timing]},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dpi=450,
    x_axis='线性真实Ratio；仅刻度标签转为百分数',K='六视角实际选择数量中位数，可为半整数',
    warning='分项独立计时；不得相加替代完整流程。Full Master常驻GPU，LoD属性每帧从CPU上传。',outputs=outputs)


def subset(scene):return data[data.scene==scene].sort_values('ratio')


def configure(ax,scene,title):
    n=stats[(stats.scene==scene)&(stats.variant=='student_fast')].N.unique()
    assert len(n)==1
    ax.set_title(f'{title}    N={int(n[0]):,}',loc='left',pad=12,fontweight='bold')
    ax.set_xlabel('Ratio（线性轴）');ax.set_ylabel('耗时（ms，越低越好）')
    ax.set_xlim(0,.735);ax.set_ylim(bottom=0)
    ticks=[.01,.05,.1,.2,.3,.4,.5,.7]
    ax.set_xticks(ticks,[f'{r:.0%}' for r in ticks],rotation=45,ha='right')
    ax.tick_params(labelsize=9)


def save(fig,name):
    for ext in ['png','svg']:
        path=OUT/(name+'.'+ext);fig.savefig(path,dpi=450,facecolor='white')
        outputs.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    plt.close(fig)


fig,axes=plt.subplots(2,2,figsize=(12,8.3))
for ax,scene,title in zip(axes.flat,SCENES,TITLES):
    g=subset(scene)
    for field,label,color,marker in [
        ('cpu_prepare','CPU全部准备（含两网络）',COLORS['cpu'],'o'),
        ('h2d_event','H2D上传（CUDA Event）',COLORS['upload'],'s'),
        ('raster_event','GPU LoD渲染（CUDA Event）',COLORS['gpu'],'^'),
        ('pipeline_wall','完整流程（独立墙钟）',COLORS['pipeline'],'D')]:
        ax.plot(g.ratio,g[field],color=color,marker=marker,ms=4,lw=1.8,label=label)
    assert g.full_pipeline_wall.nunique()==1
    ax.axhline(g.full_pipeline_wall.iloc[0],color=COLORS['full'],ls='--',lw=1.6,label='GPU常驻Full完整流程')
    configure(ax,scene,title)
handles,labels=axes.flat[0].get_legend_handles_labels()
fig.suptitle('CPU蒸馏版＋新渲染器：各阶段耗时与完整延迟',fontsize=17,fontweight='bold',y=.985)
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.945),ncol=3,fontsize=10)
fig.text(.5,.018,'分项独立采样，不能将中位数相加代替完整耗时；相机构造、模型读取和显示不计时。\n硬件：i5-9600K（6线程）＋RTX 3090；4场景 × 8档Ratio；原始实测，无平滑。',ha='center',va='bottom',fontsize=9,color='#555555')
fig.subplots_adjust(left=.08,right=.975,bottom=.13,top=.82,hspace=.57,wspace=.25)
save(fig,'CPU_GPU_SEPARATED_LATENCY')

fig,axes=plt.subplots(2,2,figsize=(12,8.3))
for ax,scene,title in zip(axes.flat,SCENES,TITLES):
    g=subset(scene);full=float(g.full_raster_event.iloc[0])
    assert g.full_raster_event.nunique()==1
    ax.axhline(full,color=COLORS['full'],ls='--',lw=1.8,label='Full渲染（GPU常驻Master）')
    ax.plot(g.ratio,g.raster_event,color=COLORS['gpu'],marker='o',ms=5,lw=2,label='LoD渲染（CPU属性已上传）')
    ax.fill_between(g.ratio,g.raster_event,full,color=COLORS['gpu'],alpha=.09)
    configure(ax,scene,title);ax.set_ylim(0,full*1.25)
    for ratio in [.1,.3,.7]:
        r=g[np.isclose(g.ratio,ratio)].iloc[0]
        ax.annotate(f'K={r.K:,.0f}' if float(r.K).is_integer() else f'K={r.K:,.1f}',
            xy=(ratio,r.raster_event),xytext=(0,-22),textcoords='offset points',ha='center',fontsize=8,color='#356C38')
    r=g[np.isclose(g.ratio,.1)].iloc[0]
    ax.text(.96,.93,f'10%档：{r.raster_speedup:.2f}×\n节省 {full-r.raster_event:.3f} ms',
        transform=ax.transAxes,ha='right',va='top',fontsize=10,color='#356C38',
        bbox=dict(boxstyle='round,pad=.35',facecolor='white',edgecolor='#D4DDD2',alpha=.95))
handles,labels=axes.flat[0].get_legend_handles_labels()
fig.suptitle('降低点数带来的纯GPU渲染收益',fontsize=17,fontweight='bold',y=.985)
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,fontsize=10)
fig.text(.5,.018,'仅含GPU渲染，不含CPU推理、H2D上传；所有属性与相机均已就绪。\nK为六个视角实际选点数中位数，标注10%／30%／70%档；各子图纵轴独立。',ha='center',va='bottom',fontsize=9,color='#555555')
fig.subplots_adjust(left=.08,right=.975,bottom=.13,top=.83,hspace=.57,wspace=.25)
save(fig,'GPU_RENDERING_BENEFIT')

for p,expected in manifest['source_sha256'].items():assert hashlib.sha256((HERE/p).read_bytes()).hexdigest()==expected
(OUT/'DATA_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'data-manifest.md').write_text('''# 可视化数据清单

所有图使用真实实验数据，未平滑、未重采样或重新训练。Ratio为线性数值轴。

| Figure | Data file | Real/mock | Source | Script | Outputs |
|---|---|---|---|---|---|
| CPU/GPU分离耗时 | reports/CPU_GPU_SEPARATED_RESULTS.csv | Real | 本轮双轮正式测速 | plot_results.py | CPU_GPU_SEPARATED_LATENCY.png / .svg |
| GPU渲染收益 | reports/CPU_GPU_SEPARATED_RESULTS.csv | Real | 同输入Full/LoD新G Renderer | plot_results.py | GPU_RENDERING_BENEFIT.png / .svg |

总地图N来自reports/TIMING_SUMMARY.csv；实际K为六视角中位数，可为半整数。原始样本位于reports/samples.csv。各分项独立计时，不是加和分解；完整流程使用独立墙钟样本。Full属性常驻GPU，CPU-LoD每帧上传选中属性。输入、脚本和输出SHA256见DATA_MANIFEST.json。
''',encoding='utf-8')
print(json.dumps(dict(passed=True,figures=2,outputs=[o['file'] for o in outputs],input_rows=len(data),font=plt.rcParams['font.family']),ensure_ascii=False))

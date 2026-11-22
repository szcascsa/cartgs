"""真实双轮CPU/完整渲染结果，中文450dpi PNG/SVG及数据哈希清单。"""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
HERE=Path(__file__).resolve().parent;OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':9,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'legend.frameon':False,'savefig.dpi':450,'svg.fonttype':'none'})
s=pd.read_csv(HERE/'reports/CPU_PAIRED_RESULTS.csv');a=pd.read_csv(HERE/'reports/CPU_AGGREGATE.csv');h=pd.read_csv(HERE/'reports/HANDOFF_PAIRED_RESULTS.csv')
outputs=[];colors=['#4E79A7','#F28E2B','#59A14F']
def save(fig,name):
    for ext in ['png','svg']:
        fig.savefig(OUT/(name+'.'+ext),bbox_inches='tight',dpi=450);outputs.append(name+'.'+ext)
    plt.close(fig)
for replica,nrow,ncol,name in [(True,4,4,'REPLICA_CPU_DEEP_LATENCY'),(False,4,2,'TUM_VECTOR_CPU_DEEP_LATENCY')]:
    scenes=[x for x in s.scene.unique() if x.startswith('Replica/')==replica];fig,axes=plt.subplots(nrow,ncol,figsize=(ncol*3.2,nrow*2.6),squeeze=False)
    for ax,scene in zip(axes.ravel(),scenes):
        g=s[(s.scene==scene)&(s.variant=='combined')&s.condition.str.startswith('local')].groupby('condition').median(numeric_only=True).reset_index();g['ratio']=g.condition.str[5:].astype(float);g=g.sort_values('ratio')
        ax.plot(g.ratio,g.baseline_ms,'o--',color=colors[0],ms=3,label='上一版V4')
        ax.plot(g.ratio,g.ms,'o-',color=colors[2],ms=3,label='原生调度＋AVX4×16')
        ax.set_title(scene.replace('TUM_RGBD','TUM').replace('freiburg','fr'),fontsize=8);ax.set_xlabel('Ratio');ax.set_ylabel('完整CPU准备（ms）');ax.set_xticks([.01,.1,.3,.7]);ax.tick_params(labelsize=7)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2);fig.tight_layout(rect=[0,0,1,.96]);save(fig,name)
fig,ax=plt.subplots(figsize=(7.8,3.9));conditions=['local0.01','local0.05','local0.1','local0.3','local0.7'];x=np.arange(len(conditions));w=.24
for i,(variant,label,color) in enumerate(zip(['schedule','kernel','combined'],['仅调度','仅AVX微核','合用'],colors)):
    values=[100*(1-1/a[(a.variant==variant)&(a.condition==c)].iloc[0].geometric_speedup) for c in conditions]
    ax.bar(x+(i-1)*w,values,w,label=label,color=color)
ax.set_xticks(x,[c[5:] for c in conditions]);ax.set_xlabel('Ratio');ax.set_ylabel('相对上一版延迟降低（%）');ax.legend(ncol=3);fig.tight_layout();save(fig,'CPU_INDEPENDENT_GAINS')
fig,axes=plt.subplots(1,3,figsize=(11,3.5));
for ax,scene in zip(axes,h.scene.unique()):
    for variant,label,color in [('core_only','优化核心＋旧交接',colors[1]),('native_all','优化核心＋原生交接',colors[2])]:
        g=h[(h.scene==scene)&(h.variant==variant)&h.condition.str.startswith('local')].groupby('condition').median(numeric_only=True).reset_index();g['ratio']=g.condition.str[5:].astype(float);g=g.sort_values('ratio')
        if variant=='core_only':ax.plot(g.ratio,g.baseline_ms,'o--',color=colors[0],label='上一版完整路径',ms=3)
        ax.plot(g.ratio,g.ms,'o-',color=color,label=label,ms=3)
    ax.set_title(scene.replace('Replica__','').replace('TUM_RGBD__','TUM/').replace('__','/'),fontsize=8);ax.set_xlabel('Ratio');ax.set_ylabel('准备＋上传＋原渲染（ms）')
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3);fig.tight_layout(rect=[0,0,1,.91]);save(fig,'END_TO_END_NATIVE_HANDOFF')
manifest=dict(real_data=True,dpi=450,outputs=outputs,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    source_csv={n:hashlib.sha256((HERE/'reports'/n).read_bytes()).hexdigest() for n in ['CPU_PAIRED_RESULTS.csv','CPU_AGGREGATE.csv','HANDOFF_PAIRED_RESULTS.csv']},
    notes='真实配对，上一版V4为基线；Ratio线性坐标；CPU不含相机矩阵构造，完整路径含本帧相机/属性H2D。')
(OUT/'DATA_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print('DEEP_FIGURES',outputs)

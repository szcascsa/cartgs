"""真实Student配对速度与质量，450dpi PNG/SVG及数据清单。"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
HERE=Path(__file__).resolve().parent;OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
source_files=['CPU_AGGREGATE.csv','CPU_PAIRED_RESULTS.csv','HANDOFF_PAIRED_RESULTS.csv','QUALITY_PER_CONDITION.csv']
manifest=dict(real_data=True,dpi=450,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sources={n:hashlib.sha256((HERE/'reports'/n).read_bytes()).hexdigest() for n in source_files},outputs=[])
(OUT/'DATA_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':9,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'legend.frameon':False,'savefig.dpi':450,'svg.fonttype':'none'})
colors=['#4E79A7','#59A14F','#F28E2B'];variant=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text())['variant'];a=pd.read_csv(HERE/'reports/CPU_AGGREGATE.csv');a=a[a.variant==variant];s=pd.read_csv(HERE/'reports/CPU_PAIRED_RESULTS.csv');s=s[s.variant==variant];h=pd.read_csv(HERE/'reports/HANDOFF_PAIRED_RESULTS.csv');q=pd.read_csv(HERE/'reports/QUALITY_PER_CONDITION.csv')
primary=['local0.1','local0.2','local0.3','local0.4','local0.5']
def geo(x):return float(np.exp(np.log(x).mean()))
def save(fig,name):
    for ext in ['png','svg']:
        fig.savefig(OUT/(name+'.'+ext),dpi=450,bbox_inches='tight');manifest['outputs'].append(name+'.'+ext)
    plt.close(fig)
def local_data(frame):
    g=frame[frame.condition.str.startswith('local')].groupby('condition').median(numeric_only=True).reset_index();g['ratio']=g.condition.str[5:].astype(float);return g.sort_values('ratio')

fig,ax=plt.subplots(figsize=(7.5,3.6));ratios=np.array([.1,.2,.3,.4,.5])
for field,label,color in [('compensator_speedup','补偿＋解码输出（16组）',colors[0]),('geometric_speedup','全部CPU准备（16组）',colors[1])]:
    ax.plot(ratios,[a[a.condition==c].iloc[0][field] for c in primary],'o-',color=color,label=label)
ax.plot(ratios,[geo(h[h.condition==c].speedup) for c in primary],'o-',color=colors[2],label='准备＋上传＋原渲染（4组）')
ax.axhline(1,color='#888888',lw=.8);ax.set_xlabel('Ratio');ax.set_ylabel('原学生 / 专项优化学生 加速倍数');ax.set_xticks(ratios);ax.legend();fig.tight_layout();save(fig,'STUDENT_FAST_SPEEDUP')

fig,axes=plt.subplots(4,4,figsize=(12.8,10.4),squeeze=False)
for ax,scene in zip(axes.ravel(),s.scene.unique()):
    g=local_data(s[s.scene==scene]);ax.plot(g.ratio,g.baseline_ms,'o--',ms=3,color=colors[0],label='原学生');ax.plot(g.ratio,g.ms,'o-',ms=3,color=colors[1],label='专项优化学生')
    ax.axvspan(.1,.5,color='#BFC5CC',alpha=.18,zorder=0);ax.set_title(scene.replace('Replica/','')+f'\nN={int(g.N.iloc[0]):,}',fontsize=8);ax.set_xlabel('Ratio');ax.set_ylabel('全部CPU准备（ms）');ax.set_xticks([.1,.3,.5,.7])
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2);fig.tight_layout(rect=[0,0,1,.96]);save(fig,'REPLICA_STUDENT_FAST_LATENCY')

fig,axes=plt.subplots(2,2,figsize=(8.5,6.5))
for ax,scene in zip(axes.ravel(),h.scene.unique()):
    g=local_data(h[h.scene==scene]);ax.plot(g.ratio,g.baseline_ms,'o--',ms=3,color=colors[0],label='原学生');ax.plot(g.ratio,g.ms,'o-',ms=3,color=colors[1],label='专项优化学生')
    ax.set_title(scene.replace('Replica__','').replace('__','/'));ax.set_xlabel('Ratio');ax.set_ylabel('准备＋上传＋原渲染（ms）');ax.set_xticks([.1,.2,.3,.4,.5])
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2);fig.tight_layout(rect=[0,0,1,.94]);save(fig,'STUDENT_FAST_END_TO_END')

fig,axes=plt.subplots(4,4,figsize=(12.8,10.4),squeeze=False)
for ax,scene in zip(axes.ravel(),s.scene.unique()):
    g=local_data(q[q.scene==scene]);ax.plot(g.ratio,g.baseline_psnr,'o--',ms=3,color=colors[0],label='原学生');ax.plot(g.ratio,g.new_psnr,'o-',ms=3,color=colors[1],label='专项优化学生')
    ax.axvspan(.1,.5,color='#BFC5CC',alpha=.18,zorder=0);ax.set_title(scene.replace('Replica/',''),fontsize=8);ax.set_xlabel('Ratio');ax.set_ylabel('GT PSNR（dB）');ax.set_xticks([.1,.3,.5,.7])
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2);fig.tight_layout(rect=[0,0,1,.96]);save(fig,'STUDENT_FAST_QUALITY')
manifest['notes']='同一个学生专项优化前后：CPU16组双轮，完整路径4场景双轮，质量3保存相机。Ratio线性坐标，灰区0.1–0.5。'
(OUT/'DATA_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print('STUDENT_FIGURES',manifest['outputs'])

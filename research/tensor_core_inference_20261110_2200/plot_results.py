"""真实双轮计时及质量，线性Ratio轴、450DPI、PNG/SVG和哈希清单。"""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
HERE=Path(__file__).resolve().parent;R=HERE/'reports';OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.is_file():fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':9,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'savefig.dpi':450,'svg.fonttype':'none','legend.frameon':False})
names=['PAIRED_RESULTS.csv','COMPONENT_RESULTS.csv','QUALITY_PER_CONDITION.csv','AGGREGATE.csv'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=dict(real_data=True,dpi=450,script_sha256=sha(Path(__file__)),sources={n:sha(R/n) for n in names},outputs=[],notes='相同学生，CUDA v31对Tensor Core。16组双轮；质量3轨迹视角；组件仅3组开发场景。')
(OUT/'DATA_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
d=pd.read_csv(R/'PAIRED_RESULTS.csv');q=pd.read_csv(R/'QUALITY_PER_CONDITION.csv');co=pd.read_csv(R/'COMPONENT_RESULTS.csv');f=json.loads((R/'FROZEN_CANDIDATE.json').read_text());ratios=[.1,.2,.3,.4,.5];colors=['#4E79A7','#59A14F','#F28E2B','#B07AA1']
geo=lambda x:float(np.exp(np.log(x).mean()))
def save(fig,name):
    fig.tight_layout(rect=[0,0,1,.955] if fig.legends else None)
    for ext in ['png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=450,bbox_inches='tight');manifest['outputs'].append(name+'.'+ext)
    plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,3.7))
ax=axes[1]
for i,(scope,label) in enumerate([('prepare','完整准备（16组）'),('pipeline','准备＋渲染（16组）')]):
    ax.plot(ratios,[geo(d[(d.variant=='both')&(d.scope==scope)&(d.condition==f'local{r:g}')].speedup) for r in ratios],'o-',color=colors[i],label=label)
ax.axhline(1,color='#888888',lw=.8);ax.set_title('每帧重新准备：16组双轮');ax.set_xticks(ratios);ax.set_xlabel('Ratio');ax.set_ylabel('原CUDA / Tensor Core 加速倍数');ax.legend(fontsize=8)
ax=axes[0]
for i,stage in enumerate(['selector','compensator']):
    variant=f'm{f[stage]}_w{f["warps"] if stage=="selector" else f["comp_warps"]}'
    ax.plot(ratios,[geo(co[(co.variant==variant)&(co.stage==stage)&(co.ratio==r)].speedup) for r in ratios],'s--',color=colors[i+2],label=stage+'组件（3组）')
ax.axhline(1,color='#888888',lw=.8);ax.set_title('独立组件：3组开发场景');ax.set_xticks(ratios);ax.set_xlabel('Ratio');ax.set_ylabel('原CUDA / Tensor Core 加速倍数');ax.legend(fontsize=8);save(fig,'TENSOR_CORE_SPEEDUP')
for scope,title in [('prepare','完整准备（ms）'),('pipeline','准备＋渲染（ms）')]:
    fig,axes=plt.subplots(4,4,figsize=(12.6,10.3))
    data=d[(d.variant=='both')&(d.scope==scope)&d.condition.isin([f'local{r:g}' for r in ratios])]
    for ax,(scene,g) in zip(axes.ravel(),data.groupby('scene',sort=True)):
        x=g.groupby('condition').median(numeric_only=True).reset_index();x['ratio']=x.condition.str[5:].astype(float);x=x.sort_values('ratio')
        ax.plot(x.ratio,x.baseline_ms,'o--',ms=3,color=colors[0],label='原CUDA');ax.plot(x.ratio,x.ms,'o-',ms=3,color=colors[1],label='Tensor Core')
        ax.set_title(scene.replace('Replica/','')+f'  N={int(g.N.iloc[0]):,}',fontsize=8);ax.set_xticks([.1,.3,.5]);ax.set_xlabel('Ratio');ax.set_ylabel(title)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2);fig.tight_layout(rect=[0,0,1,.96]);save(fig,'TENSOR_CORE_'+scope.upper())
fig,axes=plt.subplots(4,4,figsize=(12.6,10.3));data=q[(q.variant=='both')&q.condition.isin([f'local{r:g}' for r in ratios])]
for ax,(scene,g) in zip(axes.ravel(),data.groupby('scene',sort=True)):
    x=g.copy();x['ratio']=x.condition.str[5:].astype(float);x=x.sort_values('ratio')
    ax.plot(x.ratio,x.psnr_loss,'o-',ms=3,color=colors[2]);ax.axhline(0,color='#888888',lw=.8);ax.set_title(scene.replace('Replica/',''),fontsize=8);ax.set_xlabel('Ratio');ax.set_ylabel('额外PSNR损失（dB）');ax.set_xticks([.1,.3,.5])
save(fig,'TENSOR_CORE_QUALITY_LOSS');(OUT/'DATA_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print('TENSOR_FIGURES',manifest['outputs'])

"""原始计时与质量数据生成科研图；中文、450DPI、线性Ratio、PNG/SVG。"""
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
plt.rcParams.update({'font.size':9,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.18,'savefig.dpi':450,'svg.fonttype':'none','legend.frameon':False})
colors={'A':'#4E79A7','B':'#B07AA1','C':'#F28E2B','D':'#59A14F'}
labels={'A':'原FP32融合CUDA','B':'初始Tensor Core V1','C':'已优化Tensor Core网络','D':'网络＋准备/调度'}
files=['TIMINGS_ALL_VARIANTS.csv','PAIRED_ALL_VARIANTS.csv','PRIMARY_SUMMARY.csv','COMPONENT_RESULTS.csv','QUALITY_PER_CONDITION.csv','QUALITY_ALL_FRAMES.csv']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=dict(real_data=True,dpi=450,script_sha256=sha(Path(__file__)),sources={name:sha(R/name) for name in files},outputs=[],
    protocol='16组双轮，移动/跳转视角，每帧重做准备；计时外图像质量；同一学生，固定原Renderer。')
(OUT/'DATA_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
timings=pd.read_csv(R/files[0]);paired=pd.read_csv(R/files[1]);summary=pd.read_csv(R/files[2]);co=pd.read_csv(R/files[3]);q=pd.read_csv(R/files[4]);ratios=[.1,.2,.3,.4,.5];conds=[f'local{x:g}' for x in ratios]
geo=lambda x:float(np.exp(np.log(x).mean()))


def save(fig,name):
    fig.tight_layout(rect=[0,0,1,.95] if fig.legends else None)
    for extension in ['png','svg']:
        output=OUT/(name+'.'+extension);fig.savefig(output,dpi=450,bbox_inches='tight');manifest['outputs'].append(output.name)
    plt.close(fig)


fig,axes=plt.subplots(1,3,figsize=(11.6,3.6))
for ax,scope,title in zip(axes,['engine_prepare','prepare','pipeline'],['网络与地图准备','相机＋完整准备','准备＋渲染']):
    for ref in ['A','C']:
        y=[geo(paired[(paired.reference==ref)&(paired.variant=='D')&(paired.scope==scope)&(paired.condition==condition)].speedup) for condition in conds]
        ax.plot(ratios,y,'o-',color=colors[ref],label='D / '+labels[ref])
    ax.set_title(title);ax.axhline(1,color='#888888',lw=.8);ax.set_xlabel('Ratio');ax.set_ylabel('加速倍数（对照延迟/D延迟）');ax.set_xticks(ratios);ax.legend(fontsize=7)
save(fig,'JOINT_SPEEDUP')

for scope,title in [('prepare','相机＋完整准备（ms）'),('pipeline','准备＋渲染（ms）')]:
    fig,axes=plt.subplots(4,4,figsize=(12.8,10.4))
    data=timings[(timings.scope==scope)&timings.condition.isin(conds)&timings.variant.isin(['A','C','D'])]
    for ax,(scene,g) in zip(axes.ravel(),data.groupby('scene')):
        for variant in ['A','C','D']:
            h=g[g.variant==variant].copy();h['ratio']=h.condition.str[5:].astype(float);h=h.groupby('ratio').p50.median().sort_index()
            ax.plot(h.index,h.values,'o-' if variant=='D' else '--',ms=2.7,color=colors[variant],label=labels[variant],lw=1.2)
        ax.set_title(scene.replace('Replica/','')+f'  N={int(g.N.iloc[0]):,}',fontsize=8);ax.set_xlabel('Ratio');ax.set_ylabel(title);ax.set_xticks([.1,.3,.5])
    handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='upper center',ncol=3,fontsize=8);save(fig,'JOINT_'+scope.upper()+'_ALL_SCENES')

fig,axes=plt.subplots(1,2,figsize=(9.2,3.7));names=['D_noF','D_noP','D_noS','D_noH'];nice=['视锥融合','线性预投影','GPU动态调度','C++相机']
for ax,scope in zip(axes,['prepare','pipeline']):
    data=summary[(summary.reference=='D')&(summary.scope==scope)].set_index('variant')
    values=[100*(1/float(data.loc[name,'speedup'])-1) for name in names]
    ax.bar(nice,values,color=['#4E79A7','#F28E2B','#B07AA1','#59A14F']);ax.axhline(0,color='#888888',lw=.8)
    ax.set_title('完整准备' if scope=='prepare' else '准备＋渲染');ax.set_ylabel('关闭该项后的配对延迟变化（%）');ax.tick_params(axis='x',labelrotation=15)
save(fig,'JOINT_ABLATION')

fig,axes=plt.subplots(1,2,figsize=(9.2,3.7))
for ax,stage in zip(axes,['selector','compensator']):
    for variant in ['A','C','D']:
        h=co[(co.stage==stage)&(co.variant==variant)].groupby('ratio').p50.median()
        ax.plot(h.index,h.values*1000,'o-',color=colors[variant],label=variant,ms=4)
    ax.set_xlabel('Ratio');ax.set_ylabel('CUDA Event中位耗时（μs）');ax.set_title(stage);ax.legend();ax.set_xticks(ratios)
save(fig,'JOINT_NETWORK_COMPONENTS')

fig,axes=plt.subplots(4,4,figsize=(12.8,10.4));data=q[(q.variant=='D')&q.condition.isin(conds)]
for ax,(scene,g) in zip(axes.ravel(),data.groupby('scene')):
    g=g.copy();g['ratio']=g.condition.str[5:].astype(float);g=g.sort_values('ratio')
    ax.plot(g.ratio,g.psnr_loss,'o-',color='#F28E2B',ms=3);ax.axhline(0,color='#888888',lw=.8)
    ax.set_title(scene.replace('Replica/',''),fontsize=8);ax.set_xlabel('Ratio');ax.set_ylabel('D相对A额外PSNR损失（dB）');ax.set_xticks([.1,.3,.5])
save(fig,'JOINT_QUALITY_ALL_SCENES')

fig,ax=plt.subplots(figsize=(6.8,3.9));rawq=pd.read_csv(R/'QUALITY_ALL_FRAMES.csv');rawq=rawq[rawq.condition.isin(conds)]
for variant in ['A','B','C','D']:
    runtime=timings[(timings.variant==variant)&(timings.scope=='pipeline')&timings.condition.isin(conds)].p50.median()
    if variant=='A':loss=0.
    elif variant=='C':
        h=rawq[rawq.variant=='D'];loss=(h.baseline_psnr-h.C_psnr).mean()
    else:loss=rawq[rawq.variant==variant].psnr_loss.mean()
    ax.scatter([runtime],[loss],s=70,color=colors[variant]);ax.annotate(variant,(runtime,loss),xytext=(6,6),textcoords='offset points',color=colors[variant])
ax.set_xlabel('准备＋渲染 p50（ms，越小越快）');ax.set_ylabel('相对A平均额外PSNR损失（dB，越小越好）')
ax.set_title('Ratio 0.1–0.5，同一学生；16组、每组3个质量视角',fontsize=9);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));save(fig,'JOINT_SPEED_QUALITY_PARETO')
(OUT/'DATA_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('JOINT_FIGURES',manifest['outputs'])

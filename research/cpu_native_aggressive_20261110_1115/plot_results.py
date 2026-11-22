"""真实CPU性能和GPU原路径质量，450dpi PNG/SVG与数据清单。"""
from pathlib import Path
import json,hashlib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
HERE=Path(__file__).resolve().parent;OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.is_file():fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':9,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.22,'legend.frameon':False,'savefig.dpi':450,'svg.fonttype':'none'})
s=pd.read_csv(HERE/'reports/CPU_PAIRED_RESULTS.csv');q=pd.read_csv(HERE/'reports/CPU_QUALITY_ALL_FRAMES.csv')
colors=['#4E79A7','#F28E2B','#59A14F'];outputs=[]
def save(fig,name):
    for extension in ['png','svg']:
        path=OUT/(name+'.'+extension);fig.savefig(path,dpi=450,bbox_inches='tight');outputs.append(path.name)
    plt.close(fig)
for dataset,rows,cols,name in [('Replica',4,4,'REPLICA_CPU_LATENCY'),('other',4,2,'TUM_VECTOR_CPU_LATENCY')]:
    scenes=[scene for scene in s.scene.unique() if scene.startswith('Replica/')==(dataset=='Replica')]
    fig,axes=plt.subplots(rows,cols,figsize=(cols*3.25,rows*2.65),squeeze=False)
    for ax,scene in zip(axes.ravel(),scenes):
        g=s[(s.scene==scene)&s.condition.isin(['local0.01','local0.03','local0.05','local0.1','local0.2','local0.3','local0.7'])].groupby('condition').median(numeric_only=True).reset_index()
        g['ratio']=g.condition.str[5:].astype(float);g=g.sort_values('ratio')
        ax.plot(g.ratio,g.pytorch_ms,'o-',color=colors[0],label='原CPU PyTorch',markersize=3)
        ax.plot(g.ratio,g.median_ms,'o-',color=colors[2],label='C++/AVX2激进CPU',markersize=3)
        ax.set_title(scene.replace('TUM_RGBD','TUM').replace('freiburg','fr'),fontsize=8);ax.set_xlabel('Ratio');ax.set_ylabel('CPU准备中位延迟（ms）')
        ax.set_xticks([.01,.1,.3,.7]);ax.tick_params(labelsize=7)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2)
    fig.tight_layout(rect=[0,0,1,.96]);save(fig,name)

fig,axes=plt.subplots(1,2,figsize=(10,3.6))
for dataset,color in zip(['Replica','TUM_RGBD','VECtor'],colors):
    g=s[s.scene.str.startswith(dataset+'/')&s.condition.isin(['local0.01','local0.05','local0.1'])]
    axes[0].scatter(g.M,g.median_ms,s=17,alpha=.6,color=color,label=dataset)
    g=q[q.scene.str.startswith(dataset+'/')]
    axes[1].scatter(g.K_cpu,g.psnr_loss,s=13,alpha=.5,color=color,label=dataset)
axes[0].set_xlabel('实际候选 Gaussian 数 M');axes[0].set_ylabel('低Ratio CPU准备中位（ms）');axes[0].legend(fontsize=8)
axes[1].axhline(0,color='#555555',linestyle='--',linewidth=1);axes[1].set_xlabel('实际选中 Gaussian 数 K');axes[1].set_ylabel('原GPU PSNR − 新CPU PSNR（dB）')
axes[1].set_title('同固定键；正值为质量下降');fig.tight_layout();save(fig,'SPEED_AND_QUALITY')
manifest=dict(real_data=True,source_csv={n:hashlib.sha256((HERE/'reports'/n).read_bytes()).hexdigest() for n in ['CPU_PAIRED_RESULTS.csv','CPU_QUALITY_ALL_FRAMES.csv']},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),outputs=outputs,dpi=450,numeric_ratio_axis=True,
    notes='CPU延迟含整条准备，不含相机矩阵构造/H2D/渲染。质量同GPU原算法固定噪声键。')
(OUT/'DATA_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print('CPU_FIGURES_COMPLETE',outputs)

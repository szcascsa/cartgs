"""空闲GPU正式复测曲线，横轴使用实际入渲染点数。"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/reports/idle_retest_20261110'
with (OUT/'RASTERIZER_IDLE_RESULTS.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
with (OUT/'PIPELINE_IDLE_RESULTS.csv').open(encoding='utf-8-sig') as f:pipeline=list(csv.DictReader(f))
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(4,2,figsize=(14,14),layout='constrained')
conditions=['local1pct','local5pct','local10pct','local70pct']
for i,scene in enumerate(sorted({v['scene'] for v in rows})):
    ax=axes[i,0]
    group=[next(v for v in rows if v['scene']==scene and v['condition']==c) for c in conditions+['full']]
    xs=[float(v['K_median']) for v in group]
    ax.set_xscale('log')
    for name,color,label in [('original','#222222','原Rasterizer'),('warp_expand','#2878B5','Warp展开'),('bitset','#D95319','深度排序＋Bitset')]:
        ax.plot(xs,[float(v[name]) for v in group],'o-',color=color,label=label)
    ax.set_xticks(xs,[f'{label}\nK≈{float(v["K_median"]):,.0f}' for label,v in zip(['1%','5%','10%','70%','Full'],group)])
    ax.set_ylabel('纯渲染 / ms');ax.set_title(scene+' · 纯渲染')
    ax=axes[i,1]
    group=[next(v for v in pipeline if v['scene']==scene and v['condition']==c) for c in conditions]
    xs=[float(v['K_median']) for v in group]
    ax.set_xscale('log')
    for name,color,label in [('original_total_ms','#222222','原Rasterizer'),('warp_total_ms','#2878B5','Warp展开'),('bitset_total_ms','#D95319','深度排序＋Bitset')]:
        ax.plot(xs,[float(v[name]) for v in group],'o-',color=color,label=label)
    ax.set_xticks(xs,[f'{label}\nK≈{float(v["K_median"]):,.0f}' for label,v in zip(['1%','5%','10%','70%'],group)])
    ax.set_ylabel('准备＋渲染 / ms');ax.set_title(scene+' · 每帧重新准备')
for ax in axes.flat:
    ax.set_xlabel('实际入渲染点数K（对数轴）');ax.grid(axis='y',alpha=.25)
axes[0,0].legend();axes[0,1].legend()
fig.suptitle('RTX 3090空闲复测 · 相同选点和补偿属性 · FP32渲染输出逐位一致\n左：纯渲染两轮中位数；右：每次重新准备当前视角')
fig.savefig(OUT/'SPARSE_RASTERIZER_IDLE_BENCHMARK.png',dpi=170)

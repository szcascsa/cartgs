"""并发诊断图；标题明确不作为空闲GPU正式速度。"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE/'results'
rows=list(csv.DictReader((ROOT/'reports/SPARSE_RASTERIZER_RESULTS.csv').open(encoding='utf-8-sig')))
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
colors={'original':'#222222','warp_expand':'#2878B5','bitset':'#D95319'}
names={'original':'原Rasterizer','warp_expand':'Warp展开','bitset':'深度排序＋Bitset'}
for ax,scene in zip(axes.flat,sorted({v['scene'] for v in rows})):
    group=[next(v for v in rows if v['scene']==scene and v['condition']==c) for c in ['local1pct','local5pct','local10pct','local70pct','full']]
    xs=[float(v['K_median']) for v in group]
    for key,color in colors.items():ax.plot(xs,[float(v[key]) for v in group],'o-',color=color,label=names[key])
    ax.set_xscale('log')
    ax.set_xticks(xs,[f'{label}\nK≈{float(v["K_median"]):,.0f}' for label,v in zip(['1%','5%','10%','70%','Full'],group)])
    ax.set_title(scene);ax.set_ylabel('纯渲染墙钟 / ms');ax.set_xlabel('实际入渲染点数K（对数轴）');ax.grid(axis='y',alpha=.25)
axes.flat[0].legend()
fig.suptitle('低点数Rasterizer：并发负载下的诊断，正式空闲GPU测速待进行')
fig.savefig(ROOT/'reports/SPARSE_RASTERIZER_DIAGNOSTIC.png',dpi=170)

from pathlib import Path
import csv
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

root=Path(__file__).resolve().parents[1]
def read(path):return list(csv.DictReader(path.open(encoding='utf-8-sig')))
new=read(root/'evidence/cpu_optimization/APK_CPU_SUMMARY.csv')
old=read(root/'evidence/left_eye_shared/CPU_APK_BENCHMARK_SUMMARY.csv')
old=[r for r in old if r['run']=='cpu_inference_1794318528968' and not int(r['fixed_k'])]
pairs=read(root/'evidence/cpu_optimization/PAIRED_RESULTS.csv')
pairs=[r for r in pairs if r['source'].startswith('cpu_opt_final_') and not int(r['fixed_k'])]
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
x=[.1,.2,.3,.4,.5]
axes[0].plot(x,[float(r['cpu_total_ms_median']) for r in old],marker='o',linestyle='--',color='#667085',label='旧前台 N=14,294')
for run,color in zip(sorted({r['run'] for r in new}),['#1870C5','#E37735']):
    rows=[r for r in new if r['run']==run and not int(r['fixed_k'])]
    axes[0].plot(x,[float(r['cpu_total_ms_median']) for r in rows],marker='o',color=color,label=f"新前台 第{1+sorted({r['run'] for r in new}).index(run)}遍 N={int(rows[0]['N']):,}")
axes[0].set_title('Quest 3前台：历史与本次两遍（非严格配对）',fontproperties=font)
for variant,color,label in [('original','#667085','原生原版2线程'),('optimized','#1870C5','原生优化4线程')]:
    values=[statistics.median(float(r[variant+'_cpu_total_ms']) for r in pairs if abs(float(r['ratio'])-ratio)<1e-6) for ratio in x]
    axes[1].plot(x,values,marker='o',color=color,label=label)
axes[1].set_title('原生配对：12姿态、逐帧变化、两轮',fontproperties=font)
for axis in axes:
    axis.set_xticks(x);axis.set_xlabel('Ratio');axis.set_ylabel('完整CPU中位耗时 / ms',fontproperties=font)
    axis.set_ylim(bottom=0);axis.grid(alpha=.2);axis.legend(prop=font,frameon=False)
fig.suptitle('Replica office2 RGB-D · 原Selector + Student32 · 不含GPU上传／渲染／显示',fontproperties=font,fontsize=13)
fig.savefig(root/'report/QUEST_CPU_LATENCY.png',dpi=180)
fig.savefig(root/'report/QUEST_CPU_LATENCY.svg')
print('WROTE QUEST_CPU_LATENCY.png/.svg')

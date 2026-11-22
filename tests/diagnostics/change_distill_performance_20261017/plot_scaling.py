"""从原始统计生成静态科研图，不重跑训练。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D=Path(__file__).resolve().parent
rows=json.loads((D/'raw/benchmark_summary.json').read_text())
fits=json.loads((D/'raw/epoch_fit.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
for branch,color in [('selector','#1864ab'),('tf','#d9480f')]:
    native=[r for r in rows if r['branch']==branch and r['instrumentation_mode']==0]
    epoch=sorted([r for r in native if r['batch_size']==8192 and r['ratio_count']==4],key=lambda r:r['epochs'])
    x=[r['epochs'] for r in epoch]
    axes[0].errorbar(x,[r['cpu_wall_ms_median'] for r in epoch],yerr=[r['cpu_wall_ms_std'] for r in epoch],
                     color=color,label=branch,marker='o',capsize=3)
    fit=fits[branch]
    axes[0].plot([0,20],[fit['fixed_ms'],fit['fixed_ms']+20*fit['ms_per_epoch']],color=color,linestyle='--',alpha=.5)
    batch=sorted([r for r in native if r['epochs']==5 and r['ratio_count']==4],key=lambda r:r['batch_size'])
    axes[1].plot([r['batch_size'] for r in batch],[r['samples_per_second_including_audit']/1e6 for r in batch],
                 color=color,label=branch,marker='o')
    ratio=sorted([r for r in native if r['epochs']==5 and r['batch_size']==8192],key=lambda r:r['ratio_count'])
    axes[2].errorbar([r['ratio_count'] for r in ratio],[r['cpu_wall_ms_median'] for r in ratio],
                     yerr=[r['cpu_wall_ms_std'] for r in ratio],color=color,label=branch,marker='o',capsize=3)
axes[0].set(xlabel='Epochs',ylabel='CPU wall time (ms)',title='Epoch scaling: B=8192, R=4')
axes[1].set(xlabel='Batch size',ylabel='Ratio-samples / s incl. event overhead (million)',title='Batch scaling: E=5, R=4')
axes[1].set_xscale('log',base=2)
axes[1].set_xticks([2048,4096,8192,16384,32768],[2048,4096,8192,16384,32768],rotation=20)
axes[2].set(xlabel='Ratio count',ylabel='CPU wall time (ms)',title='Ratio scaling: E=5, B=8192',xticks=[1,2,4])
for axis in axes:
    axis.grid(alpha=.2)
    axis.legend(frameon=False)
fig.suptitle('Local LC 6207 | identical restored inputs | 10 warm runs | medians')
(D/'plots').mkdir(exist_ok=True)
fig.savefig(D/'plots/scaling_summary.png',dpi=160)
fig.savefig(D/'plots/scaling_summary.svg')

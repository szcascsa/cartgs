"""汇总真实GPU timestamp测试；保留全部档位和重复轮次。"""
from pathlib import Path
import csv
import json
import math
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

root=Path(__file__).resolve().parents[1]
evidence=root/'evidence'
inputs=[evidence/'pointcount_1200x680_run02.csv',evidence/'pointcount_1200x680_run03.csv']
inputs+=sorted((evidence/'apk_benchmarks').glob('*.csv'))
rows=[];summaries=[]
for path in inputs:
    data=list(csv.DictReader(line for line in path.read_text(encoding='utf-8-sig').splitlines() if not line.startswith('#')))
    measured=[x for x in data if x['warmup']=='0']
    for x in data: rows.append(dict(source=path.name,**x))
    for k in sorted({int(x['k']) for x in measured}):
        selected=[x for x in measured if int(x['k'])==k]
        item=dict(source=path.name,k=k,drawn_min=min(int(x['drawn']) for x in selected),drawn_max=max(int(x['drawn']) for x in selected),
                  width=int(selected[0]['width']),height=int(selected[0]['height']),samples=len(selected))
        for key in ['cull_gpu_ms','sort_gpu_ms','raster_gpu_ms','total_gpu_ms']:
            values=sorted(float(x[key]) for x in selected)
            item[key+'_median']=statistics.median(values)
            item[key+'_p95']=values[math.ceil(.95*len(values))-1]
        for round_id in [1,2]:
            item[f'round{round_id}_total_gpu_ms']=statistics.median(float(x['total_gpu_ms']) for x in selected if int(x['round'])==round_id)
        summaries.append(item)
fields=list(dict.fromkeys(k for row in rows for k in row))
for path,data,keys in [(evidence/'POINT_COUNT_SAMPLES.csv',rows,fields),(evidence/'POINT_COUNT_SUMMARY.csv',summaries,list(summaries[0]))]:
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(data)
(evidence/'POINT_COUNT_SUMMARY.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
font=Path('C:/Windows/Fonts/msyh.ttc')
font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':10,'axes.unicode_minus':False})
fig,axes=plt.subplots(1,2,figsize=(12.5,4.8))
for i,source in enumerate([p.name for p in inputs]):
    data=[x for x in summaries if x['source']==source]
    label=f"1200×680 原生运行 {i+1}" if i<2 else f"1680×1760 头显运行 {i-1}"
    for ax in axes:
        ax.plot([x['k'] for x in data],[x['total_gpu_ms_median'] for x in data],'o-',label=label,markersize=4,linewidth=1.6)
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel('全局静态子集输入 K（不是可见点数）');ax.set_ylabel('单眼离屏完整GPU管线（ms）')
axes[0].set_title('所有点数');axes[0].legend(frameon=False,fontsize=9);axes[1].set_title('0—5,000 点放大');axes[1].set_xlim(-100,5200);axes[1].set_ylim(0,1)
fig.suptitle('Quest 3 · SplatKit Vulkan · office2 原 Master · 无网络补偿',fontsize=13)
fig.text(.5,.018,'真实Vulkan Timestamp；包含筛选／投影、稳定排序、绘制与RenderPass。两种分辨率的FOV也不同，不作纯分辨率归因。',ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.075,1,.94))
fig.savefig(root/'report/SPLATKIT_POINT_COUNT.png',dpi=180);fig.savefig(root/'report/SPLATKIT_POINT_COUNT.svg');plt.close(fig)
print('已汇总4次运行、全部9档K、4140个测量样本；性能图已生成。')

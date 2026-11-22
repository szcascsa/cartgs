"""真实完整轨迹的原始请求统计；曲线平滑只用于展示。"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

ROOT=Path(__file__).resolve().parent
METHODS=['v2_8192','v2_expanded','v2_fast','predictive','all_candidates']
LABELS=['V2：上限8,192','V2：解除固定上限','快速PI：相同限速','视角预测控制','全部候选（参照）']
COLORS=['#0072B2','#009E73','#CC79A7','#D55E00','#777777']


def read(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def write(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def numbers(rows,key):return np.array([float(r[key]) for r in rows])


def main():
    path=ROOT/'formal/REQUESTS_WITH_QUALITY.csv';rows=read(path)
    assert len(rows)==30000 and len({(r['repeat'],r['method'],r['frame']) for r in rows})==30000
    result=[];per_repeat=[]
    for repeat in [None,0,1,2]:
        for method in METHODS:
            a=[r for r in rows if r['method']==method and (repeat is None or int(r['repeat'])==repeat)]
            t,k,q=(numbers(a,x) for x in ['render_ms','K','psnr'])
            assert len(a)==(6000 if repeat is None else 2000)
            pool=k>=numbers(a,'eligible_N')
            s=dict(method=method,repeat='all' if repeat is None else repeat,n=len(a),mean_ms=float(t.mean()),
                median_ms=float(np.median(t)),p95_ms=float(np.percentile(t,95)),p99_ms=float(np.percentile(t,99)),
                max_ms=float(t.max()),band_percent=float(100*np.mean((t>=.19)&(t<=.21))),
                over_02_percent=float(100*np.mean(t>.2)),over_021_percent=float(100*np.mean(t>.21)),
                under_019_percent=float(100*np.mean(t<.19)),MAE_ms=float(np.mean(np.abs(t-.2))),
                mean_K=float(k.mean()),min_K=int(k.min()),max_K=int(k.max()),mean_psnr=float(q.mean()),
                candidate_saturated_percent=float(100*np.mean(pool)),
                saturated_under019_percent=float(100*np.mean(pool&(t<.19))),
                control_host_p50_us=float(np.median(numbers(a,'controller_host_us'))),
                control_host_p95_us=float(np.percentile(numbers(a,'controller_host_us'),95)),
                telemetry_host_p50_us=float(np.median(numbers(a,'telemetry_host_us'))),
                prepare_host_mean_ms=float(numbers(a,'prepare_host_ms').mean()),
                render_host_mean_us=float(numbers(a,'render_host_us').mean()),
                summary_p50_us=float(np.median(numbers(a,'view_summary_us'))),
                summary_p95_us=float(np.percentile(numbers(a,'view_summary_us'),95)))
            (result if repeat is None else per_repeat).append(s)
    write(ROOT/'reports/SUMMARY.csv',result);write(ROOT/'reports/PER_REPEAT.csv',per_repeat)
    pred=[r for r in rows if r['method']=='predictive']
    nerror=numbers(pred,'predicted_N')-numbers(pred,'eligible_N')
    diagnostic=dict(candidate_count_exact_percent=float(100*np.mean(nerror==0)),
        candidate_count_max_abs=int(np.max(np.abs(nerror))),
        prediction_MAE_ms=float(np.mean(np.abs(numbers(pred,'predicted_ms')-numbers(pred,'render_ms')))),
        setup=json.loads((ROOT/'formal/SUMMARY_SETUP.json').read_text()))
    # “全部候选”只是当前网络ratio=1的实测端点，不假设延迟对K严格单调。
    endpoint={};paired=[]
    for r in rows:
        if r['method']=='all_candidates':endpoint[int(r['repeat']),int(r['frame'])]=float(r['render_ms'])
    for method in METHODS[:-1]:
        a=[r for r in rows if r['method']==method]
        high=np.array([endpoint[int(r['repeat']),int(r['frame'])] for r in a])
        t=numbers(a,'render_ms');mask=high>=.19
        paired.append(dict(method=method,endpoint_ge019_samples=int(mask.sum()),
            band_in_endpoint_ge019_percent=float(100*np.mean((t[mask]>=.19)&(t[mask]<=.21))),
            endpoint_under019_samples=int((~mask).sum()),
            mean_abs_error_all_ms=float(np.abs(t-.2).mean())))
    write(ROOT/'reports/ENDPOINT_CONTEXT.csv',paired)
    diagnostic['endpoint_below019_percent']=float(100*np.mean([v<.19 for v in endpoint.values()]))
    (ROOT/'reports/DIAGNOSTIC.json').write_text(json.dumps(diagnostic,indent=2))
    monitor=[json.loads(line) for line in (ROOT/'logs/formal05_MONITOR.jsonl').read_text().splitlines()]
    lo,hi=np.min(numbers(rows,'frame_start_unix')),np.max(numbers(rows,'frame_start_unix'))
    during=[m for m in monitor if lo<=m['time']<=hi]
    load=dict(samples=len(during),external_GPU_active=sum(bool(m['foreign_active']) for m in during),
        external_CPU_active=sum(bool(m['foreign_cpu_active']) for m in during))
    (ROOT/'reports/LOAD.json').write_text(json.dumps(load,indent=2))
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})
    plot_data=[]
    fig,axes=plt.subplots(3,1,figsize=(12,8.6),sharex=True)
    for method,label,color in zip([METHODS[i] for i in [0,2,3,4]],[LABELS[i] for i in [0,2,3,4]],[COLORS[i] for i in [0,2,3,4]]):
        a=sorted([r for r in rows if r['method']==method and r['repeat']=='0'],key=lambda r:int(r['frame']))
        x,k,q,t=(numbers(a,z) for z in ['frame','K','psnr','render_ms'])
        assert np.array_equal(x,np.arange(2000))
        ts,qs=gaussian_filter1d(t,10,mode='reflect'),gaussian_filter1d(q,35,mode='reflect')
        style=dict(color=color,lw=1.1,ls='--' if method=='all_candidates' else '-')
        axes[0].plot(x,k,label=label,**style);axes[1].plot(x,qs,**style);axes[2].plot(x,ts,**style)
        plot_data.extend(dict(method=method,frame=int(x[j]),K=int(k[j]),render_ms=t[j],render_smoothed_ms=ts[j],
            psnr=q[j],psnr_smoothed=qs[j]) for j in range(2000))
    axes[0].set(ylabel='实际高斯点数 K',title='(a) 每帧实际选择数量')
    axes[1].set(ylabel='PSNR (dB)',title='(b) 渲染质量（展示平滑σ=35帧）')
    axes[2].set(ylabel='原生渲染时间 (ms)',title='(c) 相同原生计时边界（展示平滑σ=10帧）',xlabel='RGB帧编号')
    axes[2].axhspan(.19,.21,color='#999999',alpha=.15);axes[2].axhline(.2,color='#444444',ls=':',lw=1.)
    for ax in axes:ax.set_xlim(0,1999);ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.51,.967),frameon=False,ncol=2)
    fig.suptitle('Replica office2 RGB-D｜1200×680｜Mode40｜0.2 ms目标',fontsize=13,y=.997)
    fig.tight_layout(rect=(0,.04,1,.915))
    fig.text(.5,.01,'第一轮完整2000帧；平滑仅用于展示。汇总指标使用三轮30,000次原始请求。',ha='center',fontsize=9)
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/PREDICTIVE_CONTROL.{ext}',dpi=450,bbox_inches='tight')
    plt.close(fig);write(ROOT/'figures/PLOT_DATA.csv',plot_data)
    fig,axes=plt.subplots(1,2,figsize=(10.3,3.6))
    for method,label,color in zip([METHODS[i] for i in [0,2,3]],[LABELS[i] for i in [0,2,3]],[COLORS[i] for i in [0,2,3]]):
        a=[r for r in rows if r['method']==method];t=np.sort(numbers(a,'render_ms'))
        axes[0].plot(t,np.arange(1,len(t)+1)/len(t),label=label,color=color)
        s=next(s for s in result if s['method']==method)
        axes[1].bar(label,s['band_percent'],color=color,width=.65)
        axes[1].text(label,s['band_percent']+1,f"{s['band_percent']:.1f}%",ha='center')
    axes[0].set(xlabel='原始单次延迟 (ms)',ylabel='累计概率',xlim=(.07,.3))
    axes[0].axvspan(.19,.21,color='#888888',alpha=.15);axes[0].legend(frameon=False,fontsize=8)
    axes[1].set(ylabel='0.19–0.21 ms原始命中率 (%)',ylim=(0,100))
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/RAW_DISTRIBUTION.{ext}',dpi=450,bbox_inches='tight')
    plt.close(fig)
    (ROOT/'figures/MANIFEST.json').write_text(json.dumps(dict(real_data=True,source=str(path.relative_to(ROOT)),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),repeat=0,frames=2000,
        raw_requests=30000,render_sigma_frames=10,psnr_sigma_frames=35,statistics_unfiltered=True),indent=2))
    (ROOT/'figures/data-manifest.md').write_text('# 数据清单\n\n'
        '|图|真实数据源|脚本|处理|\n|---|---|---|---|\n'
        '|PREDICTIVE_CONTROL PNG/SVG|formal/REQUESTS_WITH_QUALITY.csv；figures/PLOT_DATA.csv|analyze.py|第一轮，σ10耗时、σ35 PSNR，仅展示|\n'
        '|RAW_DISTRIBUTION PNG/SVG|formal/REQUESTS_WITH_QUALITY.csv|analyze.py|三轮原始请求，无筛选和平滑|\n',encoding='utf-8')
    print(json.dumps(dict(summary=result,diagnostic=diagnostic,load=load),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

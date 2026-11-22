"""只使用真实单次请求汇总；平滑仅用于曲线展示。"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT=Path(__file__).resolve().parent
METHODS=['fixed_3000','fixed_256','legacy_outer','legacy_inner','v2_inner']
LABELS={'fixed_3000':'固定3,000点','fixed_256':'固定256点',
    'legacy_outer':'旧PI＋旧外层计时','legacy_inner':'旧PI＋原生计时','v2_inner':'改进PI＋原生计时'}
COLORS={'fixed_3000':'#4D4D4D','fixed_256':'#56B4E9','legacy_outer':'#CC79A7',
    'legacy_inner':'#0072B2','v2_inner':'#D55E00'}


def read(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def write(path,rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        w=csv.DictWriter(stream,list(rows[0]));w.writeheader();w.writerows(rows)


def numbers(rows,key):return np.asarray([float(r[key]) for r in rows])


def main():
    path=ROOT/'formal/REQUESTS_WITH_QUALITY.csv'
    rows=read(path);summary=[];per_repeat=[]
    assert len(rows)==30000 and len({(r['repeat'],r['method'],r['frame']) for r in rows})==30000
    for repeat in [None,0,1,2]:
        for method in METHODS:
            data=[r for r in rows if r['method']==method and (repeat is None or int(r['repeat'])==repeat)]
            t,k,q=numbers(data,'render_ms'),numbers(data,'K'),numbers(data,'psnr')
            assert len(data)==(6000 if repeat is None else 2000)
            s=dict(method=method,repeat='all' if repeat is None else repeat,requests=len(data),
                mean_ms=float(t.mean()),median_ms=float(np.median(t)),p95_ms=float(np.percentile(t,95)),
                p99_ms=float(np.percentile(t,99)),max_ms=float(t.max()),
                band_percent=float(100*np.mean((t>=.19)&(t<=.21))),
                over_021_percent=float(100*np.mean(t>.21)),under_019_percent=float(100*np.mean(t<.19)),
                MAE_ms=float(np.abs(t-.2).mean()),mean_K=float(k.mean()),min_K=int(k.min()),max_K=int(k.max()),
                at_min_percent=float(100*np.mean(k==256)),
                effective_upper_percent=float(100*np.mean(k>=np.minimum(8192,numbers(data,'eligible_N')))),
                mean_psnr=float(q.mean()),
                controller_host_p50_us=float(np.median(numbers(data,'controller_host_us'))),
                controller_host_p95_us=float(np.percentile(numbers(data,'controller_host_us'),95)),
                prepare_host_mean_ms=float(numbers(data,'prepare_host_ms').mean()),
                render_host_mean_us=float(numbers(data,'render_host_us').mean()))
            (summary if repeat is None else per_repeat).append(s)
    write(ROOT/'reports/SUMMARY.csv',summary);write(ROOT/'reports/PER_REPEAT.csv',per_repeat)
    states=json.loads((ROOT/'formal/CONTROLLER_SUMMARY.json').read_text())
    assert len(states)==15 and all(s['updates']==2000 and s['pending_samples']==0 and
        s['dropped_samples']==0 and s['stale_samples']==0 for s in states)
    validation=json.loads((ROOT/'formal/VALIDATION.json').read_text());assert validation['passed']
    monitor=[json.loads(line) for line in (ROOT/'logs/formal_MONITOR.jsonl').read_text().splitlines()]
    first,last=min(numbers(rows,'frame_start_unix')),max(numbers(rows,'frame_start_unix'))
    during=[r for r in monitor if first<=r['time']<=last]
    load=dict(samples=len(during),foreign_gpu_samples=sum(bool(r['foreign_active']) for r in during),
        foreign_cpu_samples=sum(bool(r['foreign_cpu_active']) for r in during))
    (ROOT/'reports/LOAD.json').write_text(json.dumps(load,indent=2))
    # 第一轮完整轨迹，防止三次中位数掩盖实际波动；统计仍覆盖全部原始请求。
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})
    display=['fixed_3000','legacy_inner','v2_inner'];plot_rows=[]
    fig,axes=plt.subplots(3,1,figsize=(11.5,8.7),sharex=True)
    for method in display:
        rr=sorted([r for r in rows if r['method']==method and r['repeat']=='0'],key=lambda r:int(r['frame']))
        assert [int(r['frame']) for r in rr]==list(range(2000))
        x,k,t,q=(numbers(rr,key) for key in ['frame','K','render_ms','psnr'])
        ts,qs=gaussian_filter1d(t,10,mode='reflect'),gaussian_filter1d(q,35,mode='reflect')
        axes[0].plot(x,k,color=COLORS[method],lw=1.,label=LABELS[method])
        axes[1].plot(x,qs,color=COLORS[method],lw=1.3)
        axes[2].plot(x,ts,color=COLORS[method],lw=1.3)
        plot_rows.extend(dict(method=method,frame=int(x[i]),K=int(k[i]),render_ms=t[i],
            render_ms_smoothed=ts[i],psnr=q[i],psnr_smoothed=qs[i]) for i in range(2000))
    axes[0].set(ylabel='高斯点数 K',title='(a) 每帧实际选点数')
    axes[1].set(ylabel='PSNR (dB)',title='(b) 每视角重建质量（展示平滑σ=35帧）')
    axes[2].set(ylabel='原生渲染时间 (ms)',xlabel='RGB帧编号',title='(c) 相同原生计时边界（展示平滑σ=10帧）')
    axes[2].axhspan(.19,.21,color='#999999',alpha=.15)
    axes[2].axhline(.2,color='#555555',ls='--',lw=.9)
    for ax in axes:ax.set_xlim(0,1999);ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.52,.969))
    fig.suptitle('Replica office2 RGB-D｜1200×680｜Mode40｜目标0.2 ms',y=.995,fontsize=13)
    fig.tight_layout(rect=(0,.042,1,.94))
    fig.text(.5,.012,'第一轮完整2000帧；曲线仅作展示平滑。命中率、尾延迟和表格均来自三轮未平滑的单次请求。',ha='center',fontsize=9)
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/PID_V2_CONTROL.{ext}',dpi=450,bbox_inches='tight')
    plt.close(fig);write(ROOT/'figures/PLOT_DATA.csv',plot_rows)
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.7))
    for method in ['legacy_inner','v2_inner']:
        data=[r for r in rows if r['method']==method];t=np.sort(numbers(data,'render_ms'))
        axes[0].plot(t,np.arange(1,len(t)+1)/len(t),color=COLORS[method],label=LABELS[method])
        s=next(s for s in summary if s['method']==method)
        axes[1].bar(LABELS[method],s['band_percent'],color=COLORS[method],width=.55)
        axes[1].text(LABELS[method],s['band_percent']+.5,f"{s['band_percent']:.2f}%",ha='center')
    axes[0].axvspan(.19,.21,color='#999999',alpha=.15)
    axes[0].set(xlabel='原始单次渲染延迟 (ms)',ylabel='累计概率',xlim=(0,.45));axes[0].legend(frameon=False)
    axes[1].set(ylabel='±5%目标区间命中率 (%)',ylim=(0,max(5,max(s['band_percent'] for s in summary)) *1.2))
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/PID_V2_RAW_DISTRIBUTION.{ext}',dpi=450,bbox_inches='tight')
    plt.close(fig)
    manifest=dict(source=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        requests=len(rows),figure_repeat=0,render_sigma_frames=10,psnr_sigma_frames=35,
        real_data=True,raw_metrics=True,quality_not_used_by_controller=True)
    (ROOT/'figures/MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    (ROOT/'figures/data-manifest.md').write_text('# 数据清单\n\n所有数据来自真实office2全轨迹请求。\n\n'
        '|图|数据|处理|\n|---|---|---|\n'
        '|PID_V2_CONTROL|formal/REQUESTS_WITH_QUALITY.csv；figures/PLOT_DATA.csv|第一轮，耗时σ10帧、PSNRσ35帧，仅展示|\n'
        '|PID_V2_RAW_DISTRIBUTION|formal/REQUESTS_WITH_QUALITY.csv|三轮原始单次请求，无平滑|\n',encoding='utf-8')
    print(json.dumps(dict(summary=summary,load=load,validation=validation),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

"""汇总全部轮次，保留外部GPU负载差异，不按耗时筛选样本。"""
import csv
import json
from pathlib import Path
from collections import defaultdict
import sys
import numpy as np

ROOT=Path(sys.argv[1]).resolve()
OUT=ROOT/'reports'
METHODS=['cartgs_full','cartgs_full_cached','default_ratio50','default_500','default_full']


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False))


def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))


def stats(values):
    v=np.asarray(values,dtype=float);assert np.isfinite(v).all()
    return dict(mean=float(v.mean()),median=float(np.median(v)),p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9)),
                minimum=float(v.min()),maximum=float(v.max()))


def main():
    assert json.loads((ROOT/'STATUS.json').read_text())['state']=='complete'
    rows=read('TIMINGS.csv');assert len(rows)==5*3*792
    groups=defaultdict(list)
    for r in rows:
        assert abs(float(r['total_ms'])-float(r['inference_ms'])-float(r['render_ms']))<1e-8
        groups[r['method'],r['repeat']].append(r)
    summaries=[]
    for method in METHODS:
        for repeat in ['all','0','1','2']:
            values=[r for r in rows if r['method']==method and (repeat=='all' or r['repeat']==repeat)]
            assert len(values)==(2376 if repeat=='all' else 792)
            item=dict(method=method,repeat=repeat,samples=len(values))
            for k in ['inference_ms','selector_ms','compensation_ms','other_prepare_ms','render_ms','total_ms','candidate_N','eligible_N','K']:
                item[k]=stats([r[k] for r in values])
            summaries.append(item)
    save('SUMMARY.json',summaries)
    with (OUT/'SUMMARY.csv').open('w',newline='') as f:
        values=[dict(method=v['method'],repeat=v['repeat'],samples=v['samples'],**{k+'_'+n:x for k,t in v.items() if isinstance(t,dict) for n,x in t.items()}) for v in summaries]
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    pairs=[]
    lookup={(r['method'],r['repeat'],r['frame']):r for r in rows};assert len(lookup)==len(rows)
    for ref,method in [('cartgs_full','default_500'),('cartgs_full_cached','default_500'),('default_500','default_ratio50'),
                       ('default_full','default_500'),('cartgs_full','default_ratio50'),('cartgs_full_cached','default_ratio50'),('default_full','default_ratio50')]:
        for repeat in ['all','0','1','2']:
            chosen=[r for r in rows if r['method']==method and (repeat=='all' or r['repeat']==repeat)]
            a=np.asarray([float(r['total_ms']) for r in chosen]);b=np.asarray([float(lookup[ref,r['repeat'],r['frame']]['total_ms']) for r in chosen])
            pairs.append(dict(reference=ref,method=method,repeat=repeat,ratio_of_mean=float(a.mean()/b.mean()),
                mean_difference_ms=float((a-b).mean()),paired_ratio=stats(a/b),faster_fraction=float((a<b).mean())))
    save('PAIRED_COMPARISONS.json',pairs)
    q=read('QUALITY_COUNTS.csv');assert len(q)==4*792
    quality=[]
    for method in [m for m in METHODS if m!='cartgs_full_cached']:
        selected=[r for r in q if r['method']==method];assert len(selected)==792
        item=dict(method=method,frames=792)
        for k in ['psnr','K','projected_K','candidate_N']:item[k]=stats([r[k] for r in selected])
        if method in ['default_ratio50','default_500']:
            item['network_ratio']=stats([r['network_ratio'] for r in selected])
            item['budget_shortfall_frames']=sum(int(r['budget_shortfall'])>0 for r in selected)
            if method=='default_500':assert all(int(r['K'])==500 for r in selected)
            else:assert all(int(r['requested_K'])==int(.5*int(r['candidate_N'])) and int(r['K'])==min(int(r['requested_K']),int(r['eligible_N'])) for r in selected)
        quality.append(item)
    save('QUALITY_SUMMARY.json',quality)
    events=read('CUDA_EVENTS.csv');assert len(events)==480
    event_summary=[]
    for method in METHODS:
        v=[r for r in events if r['method']==method]
        event_summary.append(dict(method=method,samples=len(v),**{k:stats([r[k] for r in v]) for k in ['inference_stream_ms','render_stream_ms','total_stream_ms','total_ms']}))
    save('CUDA_EVENT_SUMMARY.json',event_summary)
    trace=json.loads((OUT/'PROFILE.json').read_text())['traceEvents']
    scopes=[v for v in trace if v.get('cat')=='user_annotation' and v.get('name','').startswith('METHOD/')]
    kernels=[v for v in trace if v.get('cat')=='kernel'];assert len(scopes)==20 and kernels
    profile=[];seen=set();top=defaultdict(lambda:dict(count=0,microseconds=0.))
    for scope in scopes:
        selected=[(i,v) for i,v in enumerate(kernels) if scope['ts']<=v['ts'] and v['ts']+v['dur']<=scope['ts']+scope['dur']+10]
        runtime=[v for v in trace if v.get('cat')=='cuda_runtime' and scope['ts']<=v['ts'] and v['ts']+v['dur']<=scope['ts']+scope['dur']+10]
        assert not seen.intersection(i for i,_ in selected);seen.update(i for i,_ in selected)
        profile.append(dict(method=scope['name'][7:],kernel_launches=len(selected),kernel_ms=sum(v['dur'] for _,v in selected)/1000,
            observed_scope_ms=scope['dur']/1000,device_syncs=sum('DeviceSynchronize' in v['name'] for v in runtime),
            stream_syncs=sum('StreamSynchronize' in v['name'] for v in runtime),
            runtime_sync_ms=sum(v['dur'] for v in runtime if 'Synchronize' in v['name'])/1000))
        for _,v in selected:top[v['name']]['count']+=1;top[v['name']]['microseconds']+=v['dur']
    assert len(seen)==len(kernels),(len(seen),len(kernels))
    psummary=[]
    for method in METHODS:
        v=[r for r in profile if r['method']==method];assert len(v)==4
        psummary.append(dict(method=method,samples=4,**{k:stats([r[k] for r in v]) for k in ['kernel_launches','kernel_ms','device_syncs','stream_syncs','runtime_sync_ms']}))
    save('PROFILE_SUMMARY.json',dict(per_call=profile,summary=psummary,total_kernels=len(kernels),all_kernels_attributed=True,
        top_kernels=[dict(name=k,**v) for k,v in sorted(top.items(),key=lambda x:x[1]['microseconds'],reverse=True)[:30]],
        note='独立Profiler运行，只有kernel类别dur被累加，不含gpu_user_annotation或嵌套CPU范围，避免重复统计。该时段未逐调用采样外部进程状态，Kernel时间只作归因，不与轻负载主测混比。'))
    print(json.dumps([dict(method=v['method'],repeat=v['repeat'],inference_ms=v['inference_ms']['mean'],render_ms=v['render_ms']['mean'],total_ms=v['total_ms']['mean']) for v in summaries],ensure_ascii=False),flush=True)


if __name__=='__main__':main()

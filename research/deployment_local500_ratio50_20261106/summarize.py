"""全部轮次性能及配对质量汇总，保留独立组件计时的范围。"""
from common import *
from collections import defaultdict
import numpy as np


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def stats(values,key,out):
    a=np.asarray([float(v[key]) for v in values]);assert np.isfinite(a).all(),key
    for label,value in [('mean',a.mean()),('median',np.median(a)),('p10',np.percentile(a,10)),('p90',np.percentile(a,90)),('p99',np.percentile(a,99)),('min',a.min()),('max',a.max())]:out[key+'_'+label]=float(value)


def main():
    assert json.loads((ROOT/'STATUS.json').read_text())['state']=='complete'
    r=ROOT/'reports';rows=read(r/'TIMINGS.csv');assert len(rows)==16632
    buckets=defaultdict(list)
    for v in rows:
        buckets[v['backend'],v['case'],'all'].append(v);buckets[v['backend'],v['case'],v['repeat']].append(v)
    summary=[]
    for (backend,case,repeat),values in buckets.items():
        item=dict(backend=backend,case=case,repeat=repeat,samples=len(values))
        for key in ['inference_ms','render_ms','total_ms','K','candidate_N','network_ratio']:
            if key=='network_ratio' and backend=='full':continue
            stats(values,key,item)
        item['fps_from_mean']=1000/item['total_ms_mean'];summary.append(item)
    write(r/'SUMMARY.csv',summary);save(r/'SUMMARY.json',summary)
    index={(v['backend'],v['case'],v['repeat'],v['frame']):v for v in rows};paired=[]
    for case in ['points500','ratio50']:
        for reference in ['pytorch','cuda']:
            for repeat in ['all','0','1','2']:
                values=[v for v in rows if v['backend']=='deployment' and v['case']==case and (repeat=='all' or v['repeat']==repeat)]
                item=dict(case=case,reference=reference,backend='deployment',repeat=repeat,samples=len(values))
                for key in ['inference_ms','render_ms','total_ms']:
                    a=np.asarray([float(index[reference,case,v['repeat'],v['frame']][key]) for v in values]);b=np.asarray([float(v[key]) for v in values])
                    item[key+'_reference_mean']=float(a.mean());item[key+'_deployment_mean']=float(b.mean());item[key+'_mean_reduction_pct']=float(100*(1-b.mean()/a.mean()))
                    item[key+'_median_reduction_pct']=float(100*(1-np.median(b)/np.median(a)));item[key+'_speedup']=float(a.mean()/b.mean())
                item['same_K_fraction']=float(np.mean([v['K']==index[reference,case,v['repeat'],v['frame']]['K'] for v in values]))
                item['same_candidate_N_fraction']=float(np.mean([v['candidate_N']==index[reference,case,v['repeat'],v['frame']]['candidate_N'] for v in values]))
                assert item['same_K_fraction']==item['same_candidate_N_fraction']==1.;paired.append(item)
    write(r/'PAIRED_SPEED.csv',paired);save(r/'PAIRED_SPEED.json',paired)
    events=read(r/'CUDA_EVENTS.csv');assert len(events)==672
    result=[];by=defaultdict(list)
    for v in events:by[v['backend'],v['case']].append(v)
    for (backend,case),values in by.items():
        item=dict(backend=backend,case=case,samples=len(values))
        for key in ['inference_stream_ms','render_stream_ms','total_stream_ms']:stats(values,key,item)
        result.append(item)
    save(r/'CUDA_EVENT_SUMMARY.json',result);write(r/'CUDA_EVENT_SUMMARY.csv',result)
    comp=read(r/'COMPONENTS.csv');assert len(comp)==1600
    by=defaultdict(list);result=[]
    for v in comp:by[v['case'],v['stage'],v['backend']].append(v)
    for (case,stage,backend),values in by.items():
        item=dict(case=case,stage=stage,backend=backend,samples=len(values),N=int(values[0]['N']),K=int(values[0]['K']),ratio=float(values[0]['network_ratio']))
        for key in ['wall_ms','stream_ms']:stats(values,key,item)
        result.append(item)
    save(r/'COMPONENT_SUMMARY.json',result);write(r/'COMPONENT_SUMMARY.csv',result)
    quality=read(ROOT/'evaluation/PER_FRAME.csv');assert len(quality)==1584
    perceptual=read(ROOT/'evaluation/PERCEPTUAL.csv');assert len(perceptual)==132
    qs=[]
    for case in ['points500','ratio50']:
        values=[v for v in quality if v['case']==case];item=dict(case=case,frames=len(values))
        for key in ['base_psnr','deployment_psnr','delta_psnr','K','candidate_N','ratio','image_max_abs','image_mean_abs','image_mse']:
            stats(values,key,item)
        for key in ['means3D_max_abs','scales_max_abs','rotations_max_abs','opacities_max_abs','dc_max_abs']:
            selected=[v for v in values if v[key]]
            if selected:stats(selected,key,item)
        item['ids_same_fraction']=float(np.mean([v['ids_same']=='True' for v in values]));item['radii_same_fraction']=float(np.mean([v['radii_same']=='True' for v in values]))
        item['minimum_id_retention']=min(float(v['id_retention']) for v in values)
        for key in ['delta_ssim','delta_lpips']:stats([v for v in perceptual if v['case']==case],key,item)
        qs.append(item)
    save(r/'QUALITY_SUMMARY.json',qs);write(r/'QUALITY_SUMMARY.csv',qs)
    profile=json.loads((r/'PROFILE.json').read_text())['traceEvents'];scopes=[v for v in profile if v.get('cat')=='user_annotation' and v.get('name','').startswith('METHOD/')]
    assert len(scopes)==28
    ps=[]
    for scope in scopes:
        values=[v for v in profile if 'dur' in v and v['ts']>=scope['ts'] and v['ts']+v['dur']<=scope['ts']+scope['dur']]
        kernels=[v for v in values if v.get('cat')=='kernel'];sync=[v for v in values if v.get('cat')=='cuda_runtime' and 'Synchronize' in v.get('name','')]
        ps.append(dict(scope=scope['name'],kernel_launches=len(kernels),kernel_us=sum(v['dur'] for v in kernels),host_scope_us=scope['dur'],synchronizations=len(sync),sync_us=sum(v['dur'] for v in sync)))
    save(r/'PROFILE_SUMMARY.json',ps)
    print(json.dumps(dict(summary_rows=len(summary),quality=qs,paired_speed=[v for v in paired if v['repeat']=='all']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()

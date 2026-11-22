"""全帧质量、实际K及三轮延迟，均保留完整样本。"""
from common import *
from collections import defaultdict
import numpy as np


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def stats(rows,key,out):
    a=np.asarray([float(v[key]) for v in rows]);assert np.isfinite(a).all(),key
    for name,value in [('mean',a.mean()),('median',np.median(a)),('p10',np.percentile(a,10)),('p90',np.percentile(a,90)),('p99',np.percentile(a,99)),('min',a.min()),('max',a.max())]:out[key+'_'+name]=float(value)


def main():
    assert json.loads((ROOT/'STATUS.json').read_text())['state']=='complete'
    r=ROOT/'reports';times=read(r/'TIMINGS.csv');assert len(times)==11880
    groups=defaultdict(list)
    for v in times:groups[v['case'],'all'].append(v);groups[v['case'],v['repeat']].append(v)
    summary=[]
    for (case,repeat),values in groups.items():
        item=dict(case=case,repeat=repeat,samples=len(values))
        for key in ['prepare_ms','render_ms','total_ms','K','candidate_N','eligible_N']:stats(values,key,item)
        item['budget_shortfall_samples']=sum(int(v['shortfall'])>0 for v in values);item['fps_from_mean']=1000/item['total_ms_mean'];summary.append(item)
    save(r/'SPEED_SUMMARY.json',summary);write(r/'SPEED_SUMMARY.csv',summary)
    events=read(r/'CUDA_EVENTS.csv');assert len(events)==480
    summary=[]
    for case in sorted({v['case'] for v in events}):
        item=dict(case=case,samples=96)
        for key in ['prepare_stream_ms','render_stream_ms','total_stream_ms']:stats([v for v in events if v['case']==case],key,item)
        summary.append(item)
    save(r/'EVENT_SUMMARY.json',summary);write(r/'EVENT_SUMMARY.csv',summary)
    rows=read(ROOT/'evaluation/PER_FRAME.csv');assert len(rows)==7128
    lookup={(v['backend'],v['case'],int(v['frame'])):v for v in rows};assert len(lookup)==len(rows)
    cases=defaultdict(list)
    for v in rows:cases[v['backend'],v['case']].append(v)
    quality=[]
    for (backend,case),values in cases.items():
        for partition in ['all','bank','nonbank']:
            selected=[v for v in values if partition=='all' or (v['bank']=='True')==(partition=='bank')]
            assert len(selected)==dict(all=792,bank=72,nonbank=720)[partition]
            item=dict(backend=backend,case=case,partition=partition,frames=len(selected),ratio=values[0]['ratio'])
            for key in ['psnr','ssim','lpips','photo_l1','candidate_N','eligible_N','requested_K','K','projected_K','shortfall']:stats(selected,key,item)
            fractions=np.asarray([int(v['K'])/max(1,int(v['candidate_N'])) for v in selected])
            item.update(actual_fraction_mean=float(fractions.mean()),actual_fraction_min=float(fractions.min()),actual_fraction_max=float(fractions.max()))
            item['budget_shortfall_frames']=sum(int(v['shortfall'])>0 for v in selected);quality.append(item)
    save(r/'QUALITY_SUMMARY.json',quality);write(r/'QUALITY_SUMMARY.csv',quality)
    differences=read(ROOT/'evaluation/OUTPUT_DIFFERENCES.csv');assert len(differences)==3168
    paired=[]
    for v in differences:
        a,b=lookup['cuda',v['case'],int(v['frame'])],lookup['deployment',v['case'],int(v['frame'])]
        assert a['K']==b['K'] and a['candidate_N']==b['candidate_N'];p=dict(v,bank=a['bank'])
        for key in ['psnr','ssim','lpips','photo_l1']:p['delta_'+key]=float(b[key])-float(a[key])
        paired.append(p)
    write(ROOT/'evaluation/PAIRED_PER_FRAME.csv',paired)
    summaries=[]
    for case in sorted({v['case'] for v in paired}):
        values=[v for v in paired if v['case']==case]
        for partition in ['all','bank','nonbank']:
            selected=[v for v in values if partition=='all' or (v['bank']=='True')==(partition=='bank')]
            item=dict(case=case,partition=partition,frames=len(selected))
            for key in ['delta_psnr','delta_ssim','delta_lpips','delta_photo_l1','image_max_abs','image_mean_abs','image_mse']:stats(selected,key,item)
            for key in ['means3D_max_abs','scales_max_abs','rotations_max_abs','opacities_max_abs','dc_max_abs']:
                chosen=[v for v in selected if v[key]]
                if chosen:stats(chosen,key,item)
            item['ids_same_fraction']=float(np.mean([v['ids_same']=='True' for v in selected]));item['radii_same_fraction']=float(np.mean([v['radii_same']=='True' for v in selected]))
            item['minimum_id_retention']=min(float(v['id_retention']) for v in selected);summaries.append(item)
    save(r/'PAIRED_QUALITY_SUMMARY.json',summaries);write(r/'PAIRED_QUALITY_SUMMARY.csv',summaries)
    print('SUMMARY_COMPLETE',len(times),len(rows),len(paired),flush=True)


if __name__=='__main__':main()

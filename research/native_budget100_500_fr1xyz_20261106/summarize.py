"""按共同RGB帧合并真贡献点数、PSNR与三轮原始延迟。"""
from common import *
from collections import defaultdict
import numpy as np


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def stats(values,key,out):
    a=np.asarray([float(v[key]) for v in values]);assert np.isfinite(a).all()
    for name,value in [('mean',a.mean()),('median',np.median(a)),('p10',np.percentile(a,10)),('p90',np.percentile(a,90)),('p99',np.percentile(a,99)),('min',a.min()),('max',a.max())]:out[key+'_'+name]=float(value)


def main():
    assert json.loads((ROOT/'STATUS.json').read_text())['state']=='complete'
    quality=read(ROOT/'evaluation/OURS_PER_VIEW.csv');times=read(ROOT/'reports/TIMINGS.csv');cartgs=read(ROOT/'evaluation/CARTGS_PER_VIEW.csv')
    assert len(quality)==3960 and len(times)==11880 and len(cartgs)==792
    hist=read(ROOT/'reference/CARTGS_TIMINGS_ONLY.csv');assert len(hist)==2376
    timing=defaultdict(list)
    for v in times:timing[int(v['budget']),int(v['frame'])].append(v)
    q={(int(v['budget']),int(v['frame'])):v for v in quality};assert len(q)==3960
    combined=[];summary=[];repeats=[]
    for budget in [100,200,300,400,500]:
        for frame in range(792):
            v=q[budget,frame];ts=timing[budget,frame];ref=cartgs[frame]
            assert len(ts)==3 and {int(x['repeat']) for x in ts}=={0,1,2} and abs(float(v['timestamp'])-float(ref['timestamp']))<1e-6
            row=dict(method='ours_'+str(budget),budget=budget,frame=frame,rgb_index=int(ref['rgb_index']),timestamp=v['timestamp'],map_N=13298,
                K=int(v['K']),projected_K=int(v['projected_K']),contributing_K=int(v['contributing_K']),psnr=float(v['psnr']),
                candidate_N=int(v['candidate_N']),eligible_N=int(v['eligible_N']),network_ratio=float(v['network_ratio']),
                shortfall=int(v['shortfall']),below_training_min=v['below_training_min']=='True',source='fresh_native_v4')
            for k in ['prepare_ms','render_ms','total_ms']:row[k]=float(np.mean([float(x[k]) for x in ts]))
            combined.append(row)
        values=[v for v in combined if v['budget']==budget];raw=[v for v in times if int(v['budget'])==budget]
        item=dict(method='ours_'+str(budget),budget=budget,frames=792,timing_samples=len(raw),source='fresh_native_v4')
        for k in ['K','projected_K','contributing_K','psnr','candidate_N','network_ratio']:stats(values,k,item)
        for k in ['prepare_ms','render_ms','total_ms']:stats(raw,k,item)
        item['shortfall_frames']=sum(v['shortfall']>0 for v in values);item['below_training_min_frames']=sum(v['below_training_min'] for v in values);summary.append(item)
        for repeat in range(3):
            r=dict(method=item['method'],budget=budget,repeat=repeat)
            for k in ['prepare_ms','render_ms','total_ms']:stats([v for v in raw if int(v['repeat'])==repeat],k,r)
            repeats.append(r)
    for v in cartgs:
        combined.append(dict(method='cartgs_full',budget='',frame=int(v['frame']),rgb_index=int(v['rgb_index']),timestamp=v['timestamp'],map_N=35301,
            K=int(v['K']),projected_K=int(v['projected_K']),contributing_K=int(v['contributing_K']),psnr=float(v['psnr']),
            candidate_N=35301,eligible_N=35301,network_ratio='',shortfall=0,below_training_min=False,source='historical_cartgs',
            prepare_ms=0.,render_ms=float(v['render_ms']),total_ms=float(v['total_ms'])))
    item=dict(method='cartgs_full',budget='',frames=792,timing_samples=2376,source='historical_cartgs')
    for k in ['K','projected_K','contributing_K','psnr']:stats(cartgs,k,item)
    for k in ['prepare_ms','render_ms','total_ms']:
        if k=='prepare_ms':
            for v in hist:v['prepare_ms']=0.
        stats(hist,k,item)
    summary.append(item);write(ROOT/'reports/PER_VIEW_COMPARISON.csv',combined);save(ROOT/'reports/SUMMARY.json',summary);write(ROOT/'reports/SUMMARY.csv',summary)
    save(ROOT/'reports/ROUND_SUMMARY.json',repeats);write(ROOT/'reports/ROUND_SUMMARY.csv',repeats)
    assert len(combined)==4752
    save(ROOT/'reports/PLOT_PROTOCOL.json',dict(frames=792,columns=3,count='real_alpha_compositing_contributors',count_smoothing=False,
        psnr=dict(method='gaussian_filter1d',sigma=35,truncate=3,mode='reflect',window=211),
        timing=dict(per_frame_reduction='mean of all three repeats',method='gaussian_filter1d',sigma=12,truncate=3,mode='reflect',window=73),
        averages='all labels and tables use raw values',cartgs='historical, different GPU load',render_images_saved=0))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

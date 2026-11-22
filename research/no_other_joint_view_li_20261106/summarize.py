"""两模型同帧质量、几何、曝光和同轮性能统计。"""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from common import ROOT,save

OUT=ROOT/'evaluation/reports'
CONTROL=Path('/home/prmi/zwj/experiments/view_conditioned_li_20261106')
TRAIN=[.01,.05,.1,.3,.5,.7]
METRICS=['psnr','ssim','lpips','photo_l1','K','contributing_K','projected_K','candidate_N','requested_K']


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def write(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def stats(values,key,out,prefix=None):
    a=np.asarray([float(v[key]) for v in values]);assert np.isfinite(a).all(),key
    for label,v in [('mean',a.mean()),('median',np.median(a)),('p10',np.percentile(a,10)),('p90',np.percentile(a,90))]:
        out[(prefix or key)+'_'+label]=float(v)


def partitions(values):
    for name in ['all','bank','nonbank']:
        selected=[v for v in values if name=='all' or (str(v['bank'])=='True')==(name=='bank')]
        assert len(selected)==dict(all=792,bank=72,nonbank=720)[name]
        yield name,selected


def main():
    assert json.loads((ROOT/'evaluation/STATUS.json').read_text())['state']=='complete'
    rows=read(OUT/'PER_VIEW.csv');assert len(rows)==55440
    lookup={(v['group'],v['case'],int(v['frame'])):v for v in rows};assert len(lookup)==len(rows)
    cases=defaultdict(list)
    for v in rows:cases[v['group'],v['case']].append(v)
    summaries=[];geometry=[]
    for (group,case),values in cases.items():
        for partition,selected in partitions(values):
            item={k:values[0][k] for k in ['group','case','scope','ratio','ratio_split']};item.update(partition=partition,frames=len(selected))
            for key in METRICS:stats(selected,key,item)
            item['budget_shortfall_frames']=sum(int(v['budget_shortfall'])>0 for v in selected);summaries.append(item)
        if case!='full':
            item=dict(group=group,case=case,frames=792)
            for key in ['xyz_p50','xyz_p90','xyz_p99','xyz_max','scale_p50','scale_p90','scale_p99','scale_max','rotation_p99_deg',
                        'opacity_mean','xyz_meters_p50','xyz_meters_p90','xyz_meters_p99','xyz_meters_max',
                        'xyz_over10_fraction','xyz_over10_alpha_fraction','xyz_over10_radii_zero_fraction']:
                if values[0].get(key):stats(values,key,item)
            item['xyz_global_max']=max(float(v['xyz_max']) for v in values)
            item['scale_global_max']=max(float(v['scale_max']) for v in values);geometry.append(item)
    index={(v['group'],v['case'],v['partition']):v for v in summaries}
    prior=[v for v in json.loads((CONTROL/'evaluation/reports/SUMMARY.json').read_text()) if v['group']=='both_li']
    for old in prior:
        new=index['four_head',old['case'],old['partition']]
        assert all(new[k]==value for k,value in old.items() if k!='group'),old['case']
    save(OUT/'HISTORICAL_METRIC_PARITY.json',dict(passed=True,cases=len(prior),rerenders=0))
    write(OUT/'SUMMARY.csv',summaries);save(OUT/'SUMMARY.json',summaries)
    write(OUT/'GEOMETRY_SUMMARY.csv',geometry);save(OUT/'GEOMETRY_SUMMARY.json',geometry)
    pairs=[];by_case=defaultdict(list)
    for r in rows:
        if r['group']!='no_other':continue
        old=lookup['four_head',r['case'],int(r['frame'])]
        assert old['timestamp']==r['timestamp'] and old['image']==r['image'] and old['bank']==r['bank']
        item={k:r[k] for k in ['case','scope','ratio','ratio_split','frame','bank']}
        for key in METRICS:item['delta_'+key]=float(r[key])-float(old[key])
        item['same_selection']=r['selection_sha256']==old['selection_sha256'];pairs.append(item);by_case[r['case']].append(item)
    write(OUT/'PAIRED_PER_VIEW.csv',pairs);paired=[]
    for case,values in by_case.items():
        for partition,selected in partitions(values):
            item={k:values[0][k] for k in ['case','scope','ratio','ratio_split']};item.update(partition=partition,frames=len(selected))
            for key in METRICS:stats(selected,'delta_'+key,item)
            item['psnr_improved_fraction']=float(np.mean([v['delta_psnr']>0 for v in selected]))
            item['same_K_fraction']=float(np.mean([v['delta_K']==0 for v in selected]));paired.append(item)
    write(OUT/'PAIRED_SUMMARY.csv',paired);save(OUT/'PAIRED_SUMMARY.json',paired)
    pindex={v['case']:v for v in paired if v['partition']=='all'}
    delta=lambda r:pindex[f'global_{r:g}']['delta_psnr_mean']
    gate=dict(global_train_mean_delta=float(np.mean([delta(r) for r in TRAIN])),global_train_min_delta=min(delta(r) for r in TRAIN),
              full_delta=pindex['full']['delta_psnr_mean'])
    gate['quality_guard_passed']=gate['global_train_mean_delta']>=-.1 and gate['global_train_min_delta']>=-.2 and gate['full_delta']>=-.1
    save(OUT/'QUALITY_GATES.json',gate)
    exposure=[];timing=[]
    for group,folder in [('four_head',CONTROL/'runs/full_both_li'),('no_other',ROOT/'runs/full_no_other')]:
        steps=read(folder/'real_lod_steps.csv');local=list(map(json.loads,(folder/'view_budget_steps.jsonl').read_text().splitlines()))
        assert len(steps)==len(local)==11580
        for ratio in TRAIN:
            a=[v for v in steps if abs(float(v['ratio'])-ratio)<1e-6];b=[v for v in local if abs(v['ratio']-ratio)<1e-6]
            assert len(a)==len(b)==1930
            exposure.append(dict(group=group,ratio=ratio,steps=len(a),full_N_sum=13298*len(a),global_K_sum=sum(int(v['K']) for v in a),
                local_K_sum=sum(v['K'] for v in b),local_candidate_N_sum=sum(v['candidate_N'] for v in b),
                li_supervised=sum(v['li_supervised'] for v in b),li_loss_mean=float(np.mean([v['li_loss'] for v in b]))))
        item=dict(group=group,steps=len(steps));stats(steps,'ms',item,'mapping_ms');timing.append(item)
    write(OUT/'TRAINING_EXPOSURE.csv',exposure);save(OUT/'TRAINING_TIMING.json',timing)
    speed_summary()
    print(json.dumps(dict(rows=len(rows),cases=len(cases),quality_gates=gate)),flush=True)


def speed_summary():
    folder=ROOT/'speed';state=json.loads((folder/'STATUS.json').read_text());assert state['state']=='complete'
    rows=read(folder/'TIMINGS.csv');assert len(rows)==14256
    by=defaultdict(list)
    for r in rows:
        by[r['group'],r['budget'],'all'].append(r);by[r['group'],r['budget'],r['repeat']].append(r)
    summary=[]
    for (group,budget,repeat),values in by.items():
        item=dict(group=group,budget=budget,repeat=repeat,samples=len(values))
        for key in ['inference_ms','selector_ms','compensation_ms','other_prepare_ms','render_ms','total_ms','K','candidate_N','eligible_N']:stats(values,key,item)
        summary.append(item)
    write(folder/'SUMMARY.csv',summary);save(folder/'SUMMARY.json',summary)
    index={(r['group'],r['budget'],r['repeat'],r['frame']):r for r in rows};pairs=[]
    for budget in ['points500','ratio50','full']:
        for repeat in ['all','0','1','2']:
            values=[r for r in rows if r['group']=='no_other' and r['budget']==budget and (repeat=='all' or r['repeat']==repeat)]
            item=dict(budget=budget,repeat=repeat,samples=len(values))
            for key in ['inference_ms','selector_ms','compensation_ms','other_prepare_ms','render_ms','total_ms']:
                old=np.asarray([float(index['four_head',budget,r['repeat'],r['frame']][key]) for r in values]);new=np.asarray([float(r[key]) for r in values])
                item[key+'_old_mean']=float(old.mean());item[key+'_new_mean']=float(new.mean());item[key+'_saved_mean']=float((old-new).mean())
                item[key+'_reduction_pct']=float(100*(old.mean()-new.mean())/old.mean()) if old.mean() else 0.
            item['same_K_fraction']=float(np.mean([r['K']==index['four_head',budget,r['repeat'],r['frame']]['K'] for r in values]));pairs.append(item)
    write(folder/'PAIRED_SUMMARY.csv',pairs);save(folder/'PAIRED_SUMMARY.json',pairs)
    for name,expected in [('COMPENSATION_SAME_INPUT',1600),('COMPENSATION_FORWARD_BACKWARD',640)]:
        values=read(folder/(name+'.csv'));assert len(values)==expected
        by=defaultdict(list)
        for r in values:by[r['K'],r['ratio'],r['group']].append(r)
        result=[]
        for (k,ratio,group),chosen in by.items():
            item=dict(K=int(k),ratio=float(ratio),group=group,samples=len(chosen))
            for key in ['wall_ms','cuda_stream_ms']:stats(chosen,key,item)
            result.append(item)
        save(folder/(name+'_SUMMARY.json'),result);write(folder/(name+'_SUMMARY.csv'),result)
    trace=json.loads((folder/'PROFILE.json').read_text())['traceEvents'];kernels=[v for v in trace if v.get('cat')=='kernel'];profile=[]
    for scope in [v for v in trace if v.get('cat')=='user_annotation' and v.get('name','').startswith('METHOD/') and 'dur' in v]:
        chosen=[v for v in kernels if v['ts']>=scope['ts'] and v['ts']+v['dur']<=scope['ts']+scope['dur']]
        profile.append(dict(scope=scope['name'],kernels=len(chosen),kernel_us=sum(v['dur'] for v in chosen),cpu_scope_us=scope['dur']))
    assert len(profile)==24
    save(folder/'PROFILE_SUMMARY.json',profile)


if __name__=='__main__':main()

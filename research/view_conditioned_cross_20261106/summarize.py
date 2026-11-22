"""同帧配对统计、实际曝光和预声明质量门槛。"""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from common import ROOT,BASE,PREVIOUS,GROUPS,save

OUT=ROOT/'evaluation/reports'
TRAIN=[.01,.05,.1,.3,.5,.7]
UNSEEN=[.03,.08,.2,.4,.6]


def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def partitions(values):
    for name in ['all','bank','nonbank']:
        chosen=[r for r in values if name=='all' or (str(r['bank'])=='True')==(name=='bank')]
        assert len(chosen)==dict(all=792,bank=72,nonbank=720)[name]
        yield name,chosen


def stats(values,key,out,prefix=None):
    a=np.array([float(v[key]) for v in values]);assert np.isfinite(a).all()
    for name,value in [('mean',a.mean()),('median',np.median(a)),('p10',np.percentile(a,10)),('p90',np.percentile(a,90))]:
        out[(prefix or key)+'_'+name]=float(value)


def main():
    assert json.loads((ROOT/'evaluation/STATUS.json').read_text())['state']=='complete'
    with (OUT/'PER_VIEW.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==134640
    lookup={(r['group'],r['case'],int(r['frame'])):r for r in rows};assert len(lookup)==len(rows)
    cases=defaultdict(list)
    for r in rows:cases[r['group'],r['case']].append(r)
    summaries=[]
    for (group,case),values in cases.items():
        for partition,chosen in partitions(values):
            item={k:values[0][k] for k in ['group','case','scope','ratio','ratio_split']};item.update(partition=partition,frames=len(chosen))
            for key in ['psnr','ssim','lpips','photo_l1','K','contributing_K','projected_K','candidate_N','requested_K']:stats(chosen,key,item)
            item['budget_shortfall_frames']=sum(int(r['budget_shortfall'])>0 for r in chosen);summaries.append(item)
    legacy=json.loads((PREVIOUS/'evaluation/reports/SUMMARY.json').read_text())
    current={(r['group'],r['case'],r['partition']):r for r in summaries}
    for old in legacy:
        new=current[old['group'],old['case'],old['partition']]
        assert all(new[k]==v for k,v in old.items()),(old['group'],old['case'],old['partition'])
    save(OUT/'HISTORICAL_METRIC_PARITY.json',dict(passed=True,cases=len(legacy),rerenders=0))
    write('SUMMARY.csv',summaries);save(OUT/'SUMMARY.json',summaries)
    pairs=[]
    from itertools import combinations
    comparisons=list(combinations(['baseline','mode','view']+GROUPS,2))
    for reference,group in comparisons:
        for r in rows:
            if r['group']!=group:continue
            ref=lookup[reference,r['case'],int(r['frame'])]
            assert r['image']==ref['image'] and r['timestamp']==ref['timestamp']
            item={k:r[k] for k in ['case','scope','ratio','ratio_split','frame','bank']};item.update(reference=reference,group=group)
            for key in ['psnr','ssim','lpips','photo_l1','K','contributing_K','projected_K']:item['delta_'+key]=float(r[key])-float(ref[key])
            item['same_selection']=r['selection_sha256']==ref['selection_sha256'];pairs.append(item)
    write('PAIRED_PER_VIEW.csv',pairs)
    by_case=defaultdict(list)
    for r in pairs:by_case[r['reference'],r['group'],r['case']].append(r)
    paired=[]
    for _,values in by_case.items():
        for partition,chosen in partitions(values):
            item={k:values[0][k] for k in ['reference','group','case','scope','ratio','ratio_split']};item.update(partition=partition,frames=len(chosen))
            for key in ['psnr','ssim','lpips','photo_l1','K','contributing_K','projected_K']:stats(chosen,'delta_'+key,item)
            item['psnr_improved_fraction']=float(np.mean([r['delta_psnr']>0 for r in chosen]))
            item['same_K_fraction']=float(np.mean([r['delta_K']==0 for r in chosen]));paired.append(item)
    write('PAIRED_SUMMARY.csv',paired);save(OUT/'PAIRED_SUMMARY.json',paired)
    pindex={(r['reference'],r['group'],r['case']):r for r in paired if r['partition']=='all'}
    gates=[]
    for reference,group in comparisons:
        d=lambda scope,r:pindex[reference,group,f'{scope}_{r:g}']['delta_psnr_mean']
        v=dict(reference=reference,group=group,global_train_mean_delta=float(np.mean([d('global',r) for r in TRAIN])),
            global_train_min_delta=min(d('global',r) for r in TRAIN),full_delta=pindex[reference,group,'full']['delta_psnr_mean'],
            view_train_mean_delta=float(np.mean([d('view',r) for r in TRAIN])),view_unseen_mean_delta=float(np.mean([d('view',r) for r in UNSEEN])),
            view_1pct_delta=d('view',.01),view_5pct_delta=d('view',.05))
        v['quality_guard_passed']=v['global_train_mean_delta']>=-.1 and v['global_train_min_delta']>=-.2 and v['full_delta']>=-.1;gates.append(v)
    save(OUT/'QUALITY_GATES.json',gates)
    geometry=[]
    for (group,case),values in cases.items():
        if group=='baseline' or case=='full':continue
        item=dict(group=group,case=case,frames=792,statistics='各视角统计量的平均值；Scale为每点最大轴相对Master倍率')
        for key in ['xyz_p50','xyz_p90','xyz_p99','xyz_max','scale_p50','scale_p90','scale_p99','scale_max','rotation_p99_deg','opacity_mean','xyz_meters_p50','xyz_meters_p90','xyz_meters_p99','xyz_meters_max','xyz_over10_fraction','xyz_over10_alpha_fraction','xyz_over10_radii_zero_fraction']:
            if values[0].get(key):stats(values,key,item)
        item['xyz_global_max']=max(float(v['xyz_max']) for v in values)
        item['scale_global_max']=max(float(v['scale_max']) for v in values)
        geometry.append(item)
    write('GEOMETRY_SUMMARY.csv',geometry);save(OUT/'GEOMETRY_SUMMARY.json',geometry)
    exposure=[];timings=[]
    for group in ['baseline','mode','view']+GROUPS:
        folder=BASE/'runs/full_no_li' if group=='baseline' else (PREVIOUS if group in ['mode','view'] else ROOT)/'runs'/f'full_{group}'
        with (folder/'real_lod_steps.csv').open() as f:steps=list(csv.DictReader(f))
        local=list(map(json.loads,(folder/'view_budget_steps.jsonl').read_text().splitlines()))
        assert len(steps)==len(local)==11580
        for ratio in TRAIN:
            a=[r for r in steps if abs(float(r['ratio'])-ratio)<1e-6]
            b=[r for r in local if abs(float(r['ratio'])-ratio)<1e-6]
            assert len(a)==len(b)==1930
            exposure.append(dict(group=group,ratio=ratio,steps=len(a),full_N_sum=13298*len(a),
                global_K_sum=sum(int(r['K']) for r in a),global_K_mean=float(np.mean([int(r['K']) for r in a])),
                local_candidate_N_sum=sum(r['candidate_N'] for r in b),local_K_sum=sum(r['K'] for r in b),
                local_K_mean=float(np.mean([r['K'] for r in b]))))
        ms=np.array([float(r['ms']) for r in steps]);timings.append(dict(group=group,steps=len(steps),mapping_ms_mean=float(ms.mean()),
            mapping_ms_median=float(np.median(ms)),mapping_ms_p90=float(np.percentile(ms,90)),
            timing_scope='含Full/global/local的记录时间；历史运行/共享GPU负载不同，不能直接归因为网络速度'))
    write('TRAINING_EXPOSURE.csv',exposure);save(OUT/'TRAINING_TIMING.json',timings)
    factorial(lookup,cases)
    print(json.dumps(dict(rows=len(rows),summaries=len(summaries),paired_cases=len(paired),quality_gates=gates)),flush=True)


def factorial(lookup,cases):
    results=[]
    metrics=['psnr','ssim','lpips','photo_l1','K','contributing_K','xyz_p99','xyz_max','scale_p99']
    for case in sorted({c for g,c in cases if g=='mode'}):
        for metric in metrics:
            if metric.startswith(('xyz_','scale_')) and case=='full':continue
            for effect,weights in [('selector_at_mode',{'selector_view':1,'mode':-1}),
                ('compensator_at_mode',{'compensator_view':1,'mode':-1}),
                ('selector_at_compensator_view',{'view':1,'compensator_view':-1}),
                ('compensator_at_selector_view',{'view':1,'selector_view':-1}),
                ('interaction',{'view':1,'selector_view':-1,'compensator_view':-1,'mode':1})]:
                values=[sum(w*float(lookup[g,case,f][metric]) for g,w in weights.items()) for f in range(792)]
                a=np.asarray(values);item=dict(case=case,metric=metric,effect=effect,frames=792,
                    mean=float(a.mean()),median=float(np.median(a)),p10=float(np.percentile(a,10)),p90=float(np.percentile(a,90)))
                results.append(item)
    write('FACTORIAL_EFFECTS.csv',results);save(OUT/'FACTORIAL_EFFECTS.json',results)


if __name__=='__main__':main()

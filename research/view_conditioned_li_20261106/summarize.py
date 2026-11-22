"""同帧配对统计、实际曝光和预声明质量门槛。"""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from common import ROOT,BASE,PREVIOUS,GROUPS,CONTROLS,HISTORY_GROUPS,history_run,save

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
    assert len(rows)==215424
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
    comparisons=[(CONTROLS[g],g) for g in GROUPS]+[('baseline',g) for g in GROUPS]
    comparisons += [('selector_li','compensator_li'),('selector_li','both_li'),('compensator_li','both_li')]
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
    for group in HISTORY_GROUPS+GROUPS:
        folder=ROOT/'runs'/f'full_{group}' if group in GROUPS else history_run(group)
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
    totals=[]
    for group in HISTORY_GROUPS+GROUPS:
        a=[v for v in exposure if v['group']==group]
        totals.append(dict(group=group,global_K_exposure_sum=sum(v['global_K_sum'] for v in a),
            local_K_exposure_sum=sum(v['local_K_sum'] for v in a),candidate_N_exposure_sum=sum(v['local_candidate_N_sum'] for v in a)))
    save(OUT/'EXPOSURE_TOTALS.json',totals)
    paired_geometry(lookup,cases)
    li_diagnostics()
    print(json.dumps(dict(rows=len(rows),summaries=len(summaries),paired_cases=len(paired),quality_gates=gates)),flush=True)


def paired_geometry(lookup,cases):
    result=[]
    for group in GROUPS:
        reference=CONTROLS[group]
        for case in sorted({c for g,c in cases if g==group and c!='full'}):
            values=[]
            for frame in range(792):
                a,b=lookup[group,case,frame],lookup[reference,case,frame]
                row=dict(group=group,reference=reference,case=case,frame=frame,bank=a['bank'])
                for k in ['xyz_p50','xyz_p90','xyz_p99','xyz_max','scale_p99','scale_max','opacity_mean']:
                    row['delta_'+k]=float(a[k])-float(b[k])
                values.append(row)
            for partition,chosen in partitions(values):
                item=dict(group=group,reference=reference,case=case,partition=partition,frames=len(chosen))
                for k in ['xyz_p50','xyz_p90','xyz_p99','xyz_max','scale_p99','scale_max','opacity_mean']:stats(chosen,'delta_'+k,item)
                result.append(item)
    write('PAIRED_GEOMETRY.csv',result);save(OUT/'PAIRED_GEOMETRY.json',result)


def li_diagnostics():
    result=[];totals=[]
    for group in GROUPS:
        local=[json.loads(v) for v in (ROOT/'runs'/f'full_{group}'/'view_budget_steps.jsonl').read_text().splitlines()]
        assert len(local)==11580
        for ratio in TRAIN:
            values=[r for r in local if abs(r['ratio']-ratio)<1e-6]
            labeled=[r for r in values if r['li_supervised']]
            item=dict(group=group,ratio=ratio,steps=len(values),supervised=len(labeled),skipped=len(values)-len(labeled))
            for key in ['li_loss','li_teacher_K','K','budget_loss','photo_loss']:stats(labeled,key,item)
            diagnostics=[r for r in labeled if 'li_teacher_mass_fraction' in r]
            item['diagnostic_samples']=len(diagnostics)
            for key in ['li_teacher_zero_contributors','li_teacher_mass_fraction','li_selected_mass_fraction','li_positive_contributors',
                'li_sample_teacher_overlap_fraction','li_selector_grad_norm','pre_local_selector_grad_norm','li_vs_pre_local_gradient_cosine']:
                if diagnostics:stats(diagnostics,key,item)
            if diagnostics:
                item['teacher_zero_fraction_mean']=float(np.mean([r['li_teacher_zero_contributors']/max(1,r['li_teacher_K']) for r in diagnostics]))
            result.append(item)
        totals.append(dict(group=group,supervised=sum(r['li_supervised'] for r in local),
            diagnostic_samples=sum('li_teacher_mass_fraction' in r for r in local)))
    write('LI_DIAGNOSTICS.csv',result);save(OUT/'LI_DIAGNOSTICS.json',dict(rows=result,totals=totals,
        diagnostic_scope='每100步及首6步；100与6有公因数，部分ratio仅首步样本，不将诊断均值视为无偏全训练统计；梯度余弦对Full/global累积，不等于纯Photo冲突'))


if __name__=='__main__':main()

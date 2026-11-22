"""诊断复跑统计；所有表均显式标记 replay，禁止作为原训练的精确历史。"""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import numpy as np
import torch
from plyfile import PlyData

root=Path('/home/prmi/zwj/octree-explicit-diagnostics-20261021')
out=root/'evidence'
def write(name,rows):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

summary=[];lifetimes=[];source_rows=[];keys_rows=[];survivor_rows=[];hist=[];window_rows=[];cohorts=[];recreations=[]
correctness={}
for seed in (2026,2027):
    run=root/f'diagnostic_seed{seed}'
    births={};deaths={};prunes=defaultdict(list);key_creations=defaultdict(list);live=set();active_keys={};concurrent_duplicates=0
    for row in csv.reader((run/'lifecycle_trace.csv').open()):
        typ=row[0];step=int(row[1]);uid=int(row[2])
        if typ=='B':
            assert uid not in births
            level,kx,ky,kz=map(int,row[3:7]);key=(level,kx,ky,kz)
            if key in active_keys:concurrent_duplicates+=1
            active_keys[key]=uid
            birth=dict(seed=seed,uid=uid,birth_mapping=step,level=level,key_x=kx,key_y=ky,key_z=kz,
                       source=row[7],source_rows=int(row[8]),distinct_source_xyz=int(row[9]))
            birth['real_observations_at_birth']=int(row[9]) if row[7].endswith('observation') else None
            birth['minimum_supplemented_slots']=max(0,10-int(row[9]))
            birth['perturb_all_slots']=int(row[9])<10
            births[uid]=birth;key_creations[key].append(uid);live.add(uid)
        elif typ=='P':
            assert uid in live
            prunes[step].append((uid,int(row[3]),list(map(float,row[4:14]))))
        elif typ=='D':
            assert uid in live and uid not in deaths
            deaths[uid]=step;live.remove(uid)
            b=births[uid];key=tuple(b[k] for k in ['level','key_x','key_y','key_z'])
            active_keys.pop(key,None)
        else:raise ValueError(typ)
    checkpoint=run/'10000_shutdown/ply'
    model=torch.jit.load(str(checkpoint/'selector.pt'),map_location='cpu')
    buffers=dict(model.named_buffers())
    uids=buffers['anchors.block_uids'].numpy()
    assert live==set(uids.tolist())
    assert set(births)==set(range(len(births)))
    text=(run/'training.log').read_text()
    geometry_events=len(re.findall(r'\[CD BEGIN\].*type=(?:LOCAL|GLOBAL)_GEOMETRY',text))
    assert geometry_events==0,'Birth/current key comparison requires geometry trace'
    d=PlyData.read(checkpoint/'point_cloud/iteration_10000/point_cloud.ply')['vertex'].data
    opacity=(1/(1+np.exp(-np.clip(d['opacity'].astype(float),-700,700)))).reshape(-1,10)
    assert len(opacity)==len(uids)
    for j,uid in enumerate(uids):
        b=births[int(uid)]
        survivor_rows.append(dict(b,age_at_end=10000-b['birth_mapping'],failed_members=int((opacity[j]<.02).sum()),
            opacity_mean=float(opacity[j].mean()),opacity_min=float(opacity[j].min()),opacity_max=float(opacity[j].max())))
    for uid,b in births.items():
        if uid in deaths:
            lifetime=deaths[uid]-b['birth_mapping'];assert lifetime>=0
            lifetimes.append(dict(b,death_mapping=deaths[uid],lifetime_mapping=lifetime))
    for key,ids in key_creations.items():
        dead=sum(uid in deaths for uid in ids)
        keys_rows.append(dict(seed=seed,level=key[0],key_x=key[1],key_y=key[2],key_z=key[3],
            creation_count=len(ids),recreation_count=len(ids)-1,deleted_instances=dead,final_instances=len(ids)-dead,
            first_mapping=births[ids[0]]['birth_mapping'],last_mapping=births[ids[-1]]['birth_mapping'],
            uids=';'.join(map(str,ids)),sources=';'.join(births[uid]['source'] for uid in ids)))
        for old,new in zip(ids[:-1],ids[1:]):
            assert old in deaths and deaths[old]<=births[new]['birth_mapping']
            recreations.append(dict(seed=seed,level=key[0],key_x=key[1],key_y=key[2],key_z=key[3],
                previous_uid=old,new_uid=new,previous_death_mapping=deaths[old],new_birth_mapping=births[new]['birth_mapping'],
                vacancy_mapping=births[new]['birth_mapping']-deaths[old],new_source=births[new]['source']))
    for step,rows in sorted(prunes.items()):
        counts=np.array([n for uid,n,op in rows]);retained=counts<10
        window_rows.append(dict(seed=seed,mapping=step,anchors_before=len(rows),deleted_anchors=int((~retained).sum()),
            retained_anchors=int(retained.sum()),retained_failed_members=int(counts[retained].sum()),
            retained_with_failed=int(((counts>0)&retained).sum()),retained_with_ge5_failed=int(((counts>=5)&retained).sum())))
        for n in range(11):hist.append(dict(seed=seed,mapping=step,failed_members=n,anchors=int((counts==n).sum())))
    for population,ids in [('all_created',list(births)),('final_surviving',list(map(int,uids)))]:
        for source in sorted(set(b['source'] for b in births.values())):
            selected=[births[i] for i in ids if births[i]['source']==source]
            if not selected:continue
            counts=np.array([b['distinct_source_xyz'] for b in selected])
            for support in sorted(set(counts)):
                source_rows.append(dict(seed=seed,population=population,source=source,distinct_source_xyz=int(support),
                    real_observation_source=source.endswith('observation'),anchors=int((counts==support).sum())))
    for source in ['all']+sorted(set(b['source'] for b in births.values())):
        bs=[b for b in births.values() if source=='all' or b['source']==source]
        ls=np.array([deaths[b['uid']]-b['birth_mapping'] for b in bs if b['uid'] in deaths])
        row=dict(seed=seed,source=source,created=len(bs),deleted=len(ls),surviving=len(bs)-len(ls),
            deleted_fraction=len(ls)/len(bs),lifetime_mean=float(ls.mean()) if len(ls) else None,
            lifetime_median=float(np.median(ls)) if len(ls) else None,
            lifetime_p10=float(np.quantile(ls,.1)) if len(ls) else None,
            lifetime_p90=float(np.quantile(ls,.9)) if len(ls) else None)
        for t in [1,10,50,100,250,500,1000]:row[f'dead_within_{t}_fraction']=float((ls<=t).mean()) if len(ls) else None
        eligible=[b for b in bs if b['birth_mapping']<=9500]
        early=sum(b['uid'] in deaths and deaths[b['uid']]-b['birth_mapping']<=500 for b in eligible)
        row['followup500_eligible_births']=len(eligible)
        row['followup500_early_deaths']=early
        row['followup500_early_death_fraction']=early/len(eligible) if eligible else None
        cohorts.append(row)
    surviving=[births[int(uid)] for uid in uids]
    obs=[b for b in surviving if b['real_observations_at_birth'] is not None]
    sparse=[b for b in obs if b['real_observations_at_birth']<=3]
    counts=np.array([len(ids) for ids in key_creations.values()])
    lif=np.array([r['lifetime_mapping'] for r in lifetimes if r['seed']==seed])
    summary.append(dict(seed=seed,diagnostic_replay=True,created=len(births),deleted=len(deaths),final=len(uids),
        final_observation_born=len(obs),final_growth_born=len(surviving)-len(obs),
        final_real_obs_1_3=len(sparse),real_obs_1_3_fraction_all_survivors=len(sparse)/len(surviving),
        real_obs_1_3_fraction_observation_survivors=len(sparse)/len(obs),
        final_real_obs_1_3_by_level={str(l):sum(b['level']==l for b in sparse) for l in range(3)},
        final_real_obs_1_3_by_count={str(k):sum(b['real_observations_at_birth']==k for b in sparse) for k in [1,2,3]},
        unique_spatial_keys=len(key_creations),recreated_keys=int((counts>1).sum()),
        recreation_events=int((counts-1).sum()),maximum_creation_count=int(counts.max()),
        creation_count_histogram={str(k):int((counts==k).sum()) for k in sorted(set(counts))},
        concurrent_duplicate_creation=concurrent_duplicates,deleted_lifetime_mean=float(lif.mean()),
        deleted_lifetime_median=float(np.median(lif)),deleted_within500_fraction=float((lif<=500).mean()),
        geometry_events=geometry_events))
    correctness[str(seed)]=dict(all_births_unique=True,final_uid_reconciled=True,created_minus_dead_equals_final=True,
        no_concurrent_duplicate_creation=concurrent_duplicates==0,geometry_events=geometry_events,
        gaussian_count_ten_per_anchor=len(d)==10*len(uids),all_lifetimes_nonnegative=True,
        log_total_created=sum(map(int,re.findall(r'Explicit Anchor Insert\] added=(\d+)',text)))==len(births),
        log_total_deleted=sum(map(int,re.findall(r'Explicit Anchor Prune\] removed=(\d+)',text)))==len(deaths))
write('replay_surviving_anchor_births.csv',survivor_rows)
write('replay_deleted_anchor_lifetimes.csv',lifetimes)
write('replay_spatial_key_creation_counts.csv',keys_rows)
write('replay_source_support_histogram.csv',source_rows)
write('replay_prune_windows.csv',window_rows)
write('replay_prune_member_histogram.csv',hist)
write('replay_lifetime_summary.csv',cohorts)
write('replay_recreation_events.csv',recreations)
(out/'replay_summary.json').write_text(json.dumps(summary,indent=2))
(out/'replay_correctness.json').write_text(json.dumps(correctness,indent=2))
print(json.dumps(summary,indent=2))

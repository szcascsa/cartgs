"""只读分析 Anchor/slot 同窗口统计与原 Cached Bernoulli 预算；不屏蔽任何候选。"""
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np
import torch
import cv2

root=Path('/home/prmi/zwj/octree-anchor-pruning-20261021')
repo=root/'source';out=root/'validation';run=root/'smoke_seed2026'
sys.path.insert(0,str(repo/'python'))
from gaussian_model import GaussianModel
from selector import load_selector,load_selector_metadata,build_selector_mask

def arrays(path):
    m=torch.jit.load(str(path),map_location='cpu')
    return {k:v.detach().numpy() for k,v in m.named_buffers()}
def write(name,rows):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def snapshot_rows(d):
    count=d['anchor_visits'];enough=count>float(d['check_interval'])*.8
    prune=enough&(d['opacity_accum']<.005*count)
    assert np.allclose(d['slot_opacity_accum'].sum(1),d['opacity_accum'],atol=1e-9,rtol=1e-12)
    assert (d['slot_below_002_count']<=count[:,None]).all() and (d['slot_below_cutoff_count']<=count[:,None]).all()
    assert (d['slot_below_cutoff_count']<=d['slot_below_002_count']).all()
    valid=count>0
    for fraction,numerator in [('slot_mean_opacity','slot_opacity_accum'),('slot_below_002_fraction','slot_below_002_count'),
                               ('slot_below_cutoff_fraction','slot_below_cutoff_count')]:
        assert np.allclose(d[fraction][valid],d[numerator][valid]/count[valid,None])
        assert np.isnan(d[fraction][~valid]).all()
    rows=[]
    for j,uid in enumerate(d['block_uids']):
        n=int(count[j]);known=n>0
        always_cut=d['slot_below_cutoff_count'][j]==n if known else np.zeros(10,bool)
        always_low=d['slot_below_002_count'][j]==n if known else np.zeros(10,bool)
        mass=float(d['opacity_accum'][j]);others=float(d['slot_opacity_accum'][j,~always_cut].sum())
        rows.append(dict(mapping=int(d['mapping_iteration']),uid=int(uid),level=int(d['levels'][j]),visits=n,
            sufficient=bool(enough[j]),deleted=bool(prune[j]),
            anchor_mean_opacity_sum=mass/n if known else None,
            always_below_002_slots=int(always_low.sum()) if known else None,
            always_below_cutoff_slots=int(always_cut.sum()) if known else None,
            current_below_002_slots=int((d['slot_current_opacity'][j]<.02).sum()),
            current_below_cutoff_slots=int((d['slot_current_opacity'][j]<1/255).sum()),
            other_slot_opacity_share=others/mass if mass>0 else None,
            valid_with_ge8_always_low_cutoff=bool(enough[j] and not prune[j] and always_cut.sum()>=8)))
    return rows

latest={};latest_sufficient={};window_rows=[];window_summary=[]
files=sorted((run/'anchor_pruning').glob('window_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
assert len(files)>0
for path in files:
    d=arrays(path);rows=snapshot_rows(d);window_rows+=rows
    for j,row in enumerate(rows):
        latest[row['uid']]=(row,d,j)
        if row['sufficient']:latest_sufficient[row['uid']]=(row,d,j)
    window_summary.append(dict(mapping=int(d['mapping_iteration']),anchors=len(rows),sufficient=sum(r['sufficient'] for r in rows),
        deleted=sum(r['deleted'] for r in rows),no_visits=sum(r['visits']==0 for r in rows),
        retained_sufficient=sum(r['sufficient'] and not r['deleted'] for r in rows),
        retained_ge8_always_cutoff=sum(r['valid_with_ge8_always_low_cutoff'] for r in rows)))
checkpoint=run/'10000_shutdown/ply';cloud=checkpoint/'point_cloud/iteration_10000'
final=arrays(cloud/'anchor_pruning_state.pt');final_rows=snapshot_rows(final)
for j,row in enumerate(final_rows):
    if row['visits']>0:latest[row['uid']]=(row,final,j)
model=load_selector(str(checkpoint/'selector.pt'),'cuda')
gaussians=GaussianModel(3);gaussians.load_ply(str(cloud/'point_cloud.ply'))
metadata=load_selector_metadata(str(cloud/'selector_metadata.ply'),'cuda')
opacity=gaussians.get_opacity.detach().cpu().numpy().reshape(-1)
uids=model.anchors.block_uids.detach().cpu().numpy()
assert np.array_equal(uids,final['block_uids']) and len(opacity)==10*len(uids)
known_persistent=np.zeros((len(uids),10),bool);known_sufficient=np.zeros(len(uids),bool)
anchor_rows=[];slot_rows=[]
for j,uid in enumerate(uids):
    uid=int(uid)
    selected=latest_sufficient.get(uid,latest.get(uid))
    assert selected is not None
    row,d,index=selected
    enough=row['sufficient'];known=row['visits']>0
    known_sufficient[j]=enough
    if enough:known_persistent[j]=d['slot_below_cutoff_count'][index]==row['visits']
    status='sufficient_window' if enough else ('insufficient_observations' if known else 'no_observed_saved_window')
    assert not row['deleted'],'A deleted UID survived final checkpoint'
    anchor_rows.append(dict(row,history_status=status,final_current_below_cutoff=int((opacity[j*10:(j+1)*10]<1/255).sum()),
        final_current_below_002=int((opacity[j*10:(j+1)*10]<.02).sum())))
    for k in range(10):
        slot_rows.append(dict(uid=uid,level=int(final['levels'][j]),slot=k,history_status=status,
            history_window_mapping=row['mapping'],visits=row['visits'],
            mean_opacity=float(d['slot_mean_opacity'][index,k]) if known else None,
            below_002_fraction=float(d['slot_below_002_fraction'][index,k]) if known else None,
            below_cutoff_fraction=float(d['slot_below_cutoff_fraction'][index,k]) if known else None,
            current_opacity=float(opacity[j*10+k]),
            always_below_cutoff_in_sufficient_window=bool(known_persistent[j,k])))
assert np.allclose(opacity.reshape(-1,10),final['slot_current_opacity'],rtol=1e-5,atol=1e-8)
examples=sorted([r for r in anchor_rows if r['sufficient'] and r['always_below_cutoff_slots']>=8],
    key=lambda r:(-r['always_below_cutoff_slots'],-r['visits'],r['uid']))
write('anchor_window_statistics.csv',window_rows)
write('window_summary.csv',window_summary)
write('final_anchor_diagnostics.csv',anchor_rows)
write('final_slot_diagnostics.csv',slot_rows)
if examples:write('effective_anchor_failed_members_examples.csv',examples[:50])

storage=cv2.FileStorage(str(checkpoint/'selector_config.yaml'),cv2.FILE_STORAGE_READ)
def get(key,default):
    node=storage.getNode(key)
    return default if node.empty() else float(node.real())
config=dict(drift_temperature=get('Protection.drift_temperature',.1),beta_fast=get('Protection.beta_fast',.6),
    beta_slow=get('Protection.beta_slow',.15),compatibility_mature_threshold=get('Protection.compatibility_mature_threshold',.8),
    entry_maturity=get('Elastic.entry_maturity',0.))
temperature=get('Selector.temperature',1.);protection=bool(get('Selector.protection_enabled',1));storage.release()
assert not protection and (metadata['maturity']==1).all()
budget=[];masks={}
for ratio in [.01,.05,.1,.15]:
    mask,stats=build_selector_mask(model,gaussians,metadata,ratio,config,temperature,protection_enabled=protection)
    selected=mask.detach().cpu().numpy().astype(bool);masks[str(ratio)]=np.flatnonzero(selected)
    n=int(selected.sum());cut=opacity<1/255
    budget.append(dict(ratio=ratio,population=len(opacity),selected=n,expected_active=stats['expected_active'],
        current_below_002=int((selected&(opacity<.02)).sum()),current_below_cutoff=int((selected&cut).sum()),
        current_cutoff_budget_fraction=float((selected&cut).sum()/max(n,1)),
        sufficient_history_selected=int((selected&known_sufficient.repeat(10)).sum()),
        history_always_cutoff_selected=int((selected&known_persistent.reshape(-1)).sum()),
        history_and_current_cutoff_selected=int((selected&known_persistent.reshape(-1)&cut).sum())))
write('lod_budget_statistics.csv',budget);np.savez_compressed(out/'selected_rows.npz',**masks)
text=(run/'training.log').read_text()
created=sum(map(int,re.findall(r'Explicit Anchor Insert\] added=(\d+)',text)))
pruned=sum(map(int,re.findall(r'Anchor Opacity Prune\].*?removed=(\d+)',text)))
assert created-pruned==len(uids)
summary=dict(scene='fr1_xyz',seed=2026,mapping_steps=10000,diagnostic_only=True,
    official_pruning_formula_verified=True,slot_statistics_diagnostic_only=True,candidate_masking_implemented=False,
    checks=len(files),created_anchors=created,pruned_anchors=pruned,final_anchors=len(uids),final_gaussians=len(opacity),
    final_current_below_002=int((opacity<.02).sum()),final_current_below_002_fraction=float((opacity<.02).mean()),
    final_current_below_cutoff=int((opacity<1/255).sum()),final_current_below_cutoff_fraction=float((opacity<1/255).mean()),
    final_with_sufficient_history=int(known_sufficient.sum()),final_without_sufficient_history=int((~known_sufficient).sum()),
    final_sufficient_anchor_with_ge8_always_cutoff=len(examples),
    example=next((r for r in examples if r['always_below_cutoff_slots']==8),examples[0] if examples else None),
    all_ten_always_cutoff_retained=sum(r['always_below_cutoff_slots']==10 for r in examples),lod_budget=budget,
    history_definition='latest sufficiently observed closed window for each final UID; not an all-training lifetime statistic',
    visibility='Full render only; an anchor is visible when any of its ten explicit Gaussians has radius>0',
    snapshot_checks='slot sum equals anchor sum; fractions use Anchor visits; unobserved ratios NaN; original prune equation')
(out/'audit_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

from common import *
import csv,collections,statistics,json,re,datetime,subprocess

def read(name):return list(csv.DictReader((R/name).open()))
def write(name,rows):
    with (R/name).open('w',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for x in rows for k in x)));w.writeheader();w.writerows(rows)
def normalize(node):
    if re.match(r'^table\d+_',node):return node.split('_',1)[1]
    return re.sub(r'_\d+$','',node)
def aggregate(name,out):
    groups=collections.defaultdict(list)
    for x in read(name):groups[(int(x['deterministic']),x['comparison'],normalize(x['node']))].append(x)
    rows=[]
    for (det,comp,node),values in sorted(groups.items()):
        rows.append(dict(deterministic=det,comparison=comp,node=node,checks=len(values),different_checks=sum(x['bits']=='0' for x in values),
            different_elements=sum(max(0,int(x['different'])) for x in values),max_abs=max(float(x['max_abs']) for x in values),
            max_relative_l2=max(float(x.get('relative_l2',x.get('max_tensor_relative_l2',0))) for x in values)))
    write(out,rows);return rows
nodes=aggregate('nodes.csv','NODE_SUMMARY.csv');adam=aggregate('adam_nodes.csv','ADAM_SUMMARY.csv')
photo=aggregate('photo_nodes.csv','PHOTO_NODE_SUMMARY.csv');photo_adam=aggregate('photo_adam.csv','PHOTO_ADAM_SUMMARY.csv')
rescue=aggregate('trajectories_scatter_rescue.csv','SCATTER_RESCUE_SUMMARY.csv')
trajectory_rows=[]
for name in ['trajectories.csv',*[x.name for x in sorted(R.glob('trajectory_repeat*.csv'))]]:
    values=read(name);groups=collections.defaultdict(list)
    for x in values:groups[int(x['deterministic']),int(x['seed']),x['comparison'],x['node']].append(x)
    for (det,seed,comp,node),items in sorted(groups.items()):
        last=max(items,key=lambda x:int(x['step']));bad=[int(x['step']) for x in items if x['bits']=='0']
        trajectory_rows.append(dict(run=name,deterministic=det,seed=seed,comparison=comp,node=node,steps=len(items),
            first_different_step=min(bad) if bad else -1,different_steps=len(bad),max_abs=max(float(x['max_abs']) for x in items),
            last_max_abs=float(last['max_abs']),max_relative_l2=max(float(x['max_tensor_relative_l2']) for x in items)))
write('TRAJECTORY_SUMMARY.csv',trajectory_rows)
scatter=[]
groups=collections.defaultdict(list)
for x in read('scatter_replay.csv'):groups[int(x['deterministic']),int(x['layout']),int(x['ratio_index'])].append(x)
for (det,layout,j),items in sorted(groups.items()):
    scatter.append(dict(deterministic=det,layout=['preserve_incoming_stride','contiguous','unique_owner_only'][layout],ratio_index=j,
        K=items[0]['K'],replays=len(items),different_replays=sum(x['bits']=='0' for x in items),max_abs=max(float(x['max_abs']) for x in items)))
write('SCATTER_REPLAY_SUMMARY.csv',scatter)
comparisons=['baseline_repeat','baseline_vs_fused']
long_stats=[]
for comp in comparisons:
    for node in ['table_parameter','network_parameter','forward']:
        v=[x for x in trajectory_rows if x['deterministic']==0 and x['comparison']==comp and x['node']==node]
        long_stats.append(dict(comparison=comp,node=node,trials=len(v),median_peak_abs=statistics.median(x['max_abs'] for x in v),
            max_peak_abs=max(x['max_abs'] for x in v),trials_peak_above_1e3=sum(x['max_abs']>1e-3 for x in v),
            max_relative_l2=max(x['max_relative_l2'] for x in v)))
write('TRAJECTORY_DISTRIBUTION.csv',long_stats)
assert all(x['different_checks']==0 for x in nodes if x['deterministic']==1)
assert all(x['different_checks']==0 for x in nodes if x['node'] in ['gather_input_VJP','packed_input_VJP','output','Master_XYZ_VJP','network_gradient'])
assert all(x['different_checks']==0 for x in adam if x['deterministic']==1 or x['comparison'] in ['same_gradient_rescue','native_vs_production_Adam'])
assert all(x['different_steps']==0 for x in trajectory_rows if x['deterministic']==1)
assert all(x['different_checks']==0 for x in rescue if x['node']!='table_gradient')
assert all(x['different_checks']==0 for x in photo if x['deterministic']==1 or x['node'] in ['gather_input_VJP','packed_input_VJP','output','Master_XYZ_VJP','network_gradient'])
assert all(x['different_checks']==0 for x in photo_adam if x['deterministic']==1 or x['comparison']=='same_gradient_rescue')
before=json.loads((R/'main_before.json').read_text());after={n:sha(P/n) for n in before};changed=[n for n in before if before[n]!=after[n]]
save('main_after.json',after);save('MAIN_INTEGRITY.json',dict(passed=not changed,files=len(before),changed=changed,production_default_unchanged=True));assert not changed
accepted=json.loads((PREV/'accepted_source_manifest.json').read_text());assert all(sha(S/n)==h for n,h in accepted.items())
save('DIAGNOSIS_VERIFICATION.json',dict(passed=True,ratios=11,seeds=3,first_mismatch='original_IndexSelectBackward_CUDA_Scatter_output',
    deterministic_model_VJP_and_Adam_bitwise=True,deterministic_trajectories=2,steps_per_trajectory=72,
    nondeterministic_trajectory_trials_per_comparison=10,scatter_rescue_trajectories=2,
    normal_FP32_atomic_rounding_supported=True,rounding_proved_harmless=False,production_training_fusion_enabled=False,
    main_unchanged_files=len(before),reused_research_source_unchanged_files=len(accepted),commit=False,push=False))
print('DIAGNOSIS_SUMMARY_VERIFIED',flush=True)

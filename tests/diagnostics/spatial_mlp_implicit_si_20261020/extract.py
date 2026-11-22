"""固定MLP hard排名、共享传播与空间诊断；不反馈训练。"""
from pathlib import Path
import json,csv,hashlib,shutil
import numpy as np,torch
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,rows):
    with open(ROOT/n,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(a):
    a=np.asarray(a,dtype=float)
    if not len(a):return dict(N=0,mean=0.,median=0.,std=0.,p1=0.,p10=0.,p90=0.,p99=0.,max_abs=0.)
    return dict(N=len(a),mean=float(a.mean()),median=float(np.median(a)),std=float(a.std()),p1=float(np.quantile(a,.01)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),p99=float(np.quantile(a,.99)),max_abs=float(abs(a).max()))
def corr(x,y):return float(np.corrcoef(x,y)[0,1])if np.std(x)>0 and np.std(y)>0 else 0.
assert read(ROOT/'training_complete.json')['all_passed'];reg=read(ROOT/'preregistration.json');sets={};rankings={};curves=[];hard=[];hardviews=[];outputs=[];indirect=[];groups=[];roots=[];smooth=[];smoothcorr=[];overlap=[];promotion=[];promotionrows=[];statechecks={};row_records=[]
legacysets=np.load(ROOT/'legacy_selected_sets.npz')
for name in reg['scenes']:
    here=ROOT/name;x=np.load(here/'ranking_inputs.npz');b=np.load(here/'rankings.npz');norm=np.load(here/'normalized_xyz.npz');ids=x['row_ids'];N=len(ids);base=np.log(x['raw_I'].astype(float)+1e-12);rawrank=np.empty(N,np.int64);rawrank[b['A_order']]=np.arange(N)+1
    rng=np.random.default_rng(reg['smoothness']['seed']);pair=rng.integers(0,N,size=(reg['smoothness']['pairs'],2));distance=np.linalg.norm(norm['normalized_xyz'][pair[:,0]]-norm['normalized_xyz'][pair[:,1]],axis=1);worlddist=np.linalg.norm(x['xyz'][pair[:,0]]-x['xyz'][pair[:,1]],axis=1);bins=np.quantile(distance,[0,.25,.5,.75,1]);np.savez_compressed(here/'spatial_pair_fixture.npz',pair=pair,normalized_distance=distance,world_distance=worlddist,bins=bins)
    for ratio in reg['ratios']:
        K=int(ratio*N+.5)
        for arm,key in [('Raw','A_order'),('Fixed','B_order'),('C','C0_order')]:sets[f'{name}_{arm}_{ratio}']=np.sort(ids[b[key][:K]])
        for seed in reg['seeds']:
            case=f'MLP_{int(100*ratio)}_seed_{seed}';d=here/'models'/case;shutil.copy2(here/'normalization.json',d/'normalization.json');meta=dict(scene=name,ratio=ratio,seed=seed);records=[json.loads(l[3:])for l in(d/'training.log').read_text().splitlines()if l.startswith('PG ')]
            for z in records:
                kind=z.pop('kind')
                if kind=='training':
                    layer=z.pop('layer_grad_norm')
                    for j,v in enumerate(layer):z[f'layer{j+1}_grad_norm']=v
                    for k in ['output_changed_fraction','selected_changed_fraction','unselected_changed_fraction','mean_abs_output_update']:z.setdefault(k,'')
                    curves.append(dict(**meta,**z))
                elif kind=='hard_summary':
                    s=np.sort(np.asarray(z.pop('selected_source_rows'),dtype=np.int64));sets[f'{name}_{case}_step_{z["step"]}']=s;hard.append(dict(**meta,**z,K=K))
                elif kind=='hard_view':hardviews.append(dict(**meta,**z,K=K))
            for step in reg['tensor_checkpoints']:
                t={k:v.detach().cpu().numpy()for k,v in torch.jit.load(str(d/f'step_{step}.pt'),map_location='cpu').named_buffers()};a=t['correction'];score=base+a;order=np.lexsort((ids,-score));s=np.sort(ids[order[:K]]);key=f'{name}_{case}_step_{step}';np.testing.assert_array_equal(s,sets[key]);np.testing.assert_array_equal(t['normalized_xyz'],norm['normalized_xyz']);rankings[key]=ids[order]
                assert np.isfinite(score).all()and len(s)==len(np.unique(s))==K
                if step==0:assert np.all(a==0);np.testing.assert_array_equal(order,b['A_order'])
                np.savez_compressed(d/f'step_{step}.npz',row_ids=ids,correction=a,score=score,order=order,selected=s,direct_selected_count=t['direct_selected_count'])
                with np.errstate(over='ignore',invalid='ignore'):factor=np.exp(a)
                for quantity,v in [('correction',a),('exp_correction',factor)]:
                    with np.errstate(over='ignore',invalid='ignore'):st=stats(v)
                    st={k:('NONFINITE_DERIVED_FACTOR'if isinstance(value,float)and not np.isfinite(value)else value)for k,value in st.items()}
                    outputs.append(dict(**meta,step=step,quantity=quantity,derived_factor_nonfinite_count=int((~np.isfinite(v)).sum()),**st))
                count=t['direct_selected_count'];never=count==0;changed=abs(a)>reg['output_numeric_epsilon'];indirect.append(dict(**meta,step=step,diagnostic='cumulative_never_selected',N=N,selected_N=int((count>0).sum()),unselected_N=int(never.sum()),all_changed_fraction=float(changed.mean()),selected_changed_fraction=float(changed[count>0].mean())if(count>0).any()else 0.,unselected_changed_fraction=float(changed[never].mean())if never.any()else 0.,unselected_update_abs_mean=float(abs(a[never]).mean())if never.any()else 0.))
                if step!=2000:continue
                finalrank=np.empty(N,np.int64);finalrank[order]=np.arange(N)+1;shift=rawrank-finalrank;selected=np.isin(ids,s)
                for group,mask in [('frequently',count>1000),('rarely',(count>0)&(count<=1000)),('never',count==0)]:
                    for quantity,value in [('correction',a),('abs_correction',abs(a)),('score_shift',a),('Raw_rank_promotion',shift)]:groups.append(dict(**meta,group=group,quantity=quantity,**stats(value[mask])))
                oldt=dict(torch.jit.load(str(here/'legacy_models'/f'PG_{int(100*ratio)}_seed_{seed}'/'step_2000.pt'),map_location='cpu').named_buffers());olda=oldt['coefficient'].numpy();oldsel=legacysets[f'{name}_PG_{int(100*ratio)}_seed_{seed}_step_2000'];sets[f'{name}_PG-ST_{ratio}_{seed}']=oldsel;sets[f'{name}_MLP-ST_{ratio}_{seed}']=s
                for arm,field in [('MLP-ST',a),('PG-ST',olda)]:
                    diff=abs(field[pair[:,0]]-field[pair[:,1]])
                    smoothcorr.append(dict(**meta,arm=arm,pairs=len(pair),corr_normalized_distance_abs_difference=corr(distance,diff),corr_world_distance_abs_difference=corr(worlddist,diff)))
                    for i,label in enumerate(['very_near','near','medium','far']):
                        m=(distance>=bins[i])&(distance<=bins[i+1]if i==3 else distance<bins[i+1]);smooth.append(dict(**meta,arm=arm,distance_bin=label,bin_low=float(bins[i]),bin_high=float(bins[i+1]),mean_distance=float(distance[m].mean()),**stats(diff[m])))
                root=b['old_root']
                for r in np.unique(root):
                    m=root==r;roots.append(dict(**meta,root=int(r),N=int(m.sum()),correction_mean=float(a[m].mean()),correction_std=float(a[m].std()),correction_range=float(np.ptp(a[m])),selected_count=int(selected[m].sum()),direct_selected_count_sum=int(count[m].sum()),mean_Raw_SI=float(x['raw_I'][m].mean()),final_selected_ratio=float(selected[m].mean())))
                for arm in ['PG-ST','Raw','Fixed','C']:
                    ref=oldsel if arm=='PG-ST'else sets[f'{name}_{arm}_{ratio}'];inter=np.intersect1d(s,ref);added=np.setdiff1d(s,ref);removed=np.setdiff1d(ref,s);overlap.append(dict(**meta,reference=arm,K=K,intersection=len(inter),Jaccard=len(inter)/(2*K-len(inter)),added=len(added),removed=len(removed)));sets[f'{name}_{case}_added_vs_{arm}']=added;sets[f'{name}_{case}_removed_vs_{arm}']=removed
                for arm,s in [('MLP-ST',s),('PG-ST',oldsel)]:
                    added=np.setdiff1d(s,sets[f'{name}_Raw_{ratio}']);idx=np.searchsorted(ids,added);rr=rawrank[idx];pct=(rr-1)/(N-1);promotion.append(dict(**meta,arm=arm,K=K,**stats(pct),max_Raw_rank=int(rr.max())if len(rr)else 0,count_Raw_rank_gt_2K=int((rr>2*K).sum())))
                    for row,r,pctvalue in zip(added,rr,pct):promotionrows.append(dict(**meta,arm=arm,source_row=int(row),Raw_rank=int(r),Raw_rank_percentile=float(pctvalue)))
                np.savez_compressed(d/'final_field.npz',row_ids=ids,normalized_xyz=norm['normalized_xyz'],correction=a,score=score,order=order,ranking=ids[order],direct_selected_count=count,Raw_rank=rawrank,final_rank=finalrank)
                statechecks[key]=dict(all_passed=True,N=N,K=K,never_selected=int(never.sum()),never_selected_nonzero_correction=int((never&changed).sum()),never_selected_exact_nonzero_correction=int((never&(a!=0)).sum()),correction_std=float(a.std()),full_initial_Raw_exact=True,normalized_SHA=sha(here/'normalized_xyz.npz'))
            for step in reg['output_update_steps']:
                t={k:v.detach().cpu().numpy()for k,v in torch.jit.load(str(d/f'output_update_{step}.pt'),map_location='cpu').named_buffers()};delta=abs(t['correction_after']-t['correction_before']);m=t['selected_mask'];changed=delta>reg['output_numeric_epsilon'];indirect.append(dict(**meta,step=step,diagnostic='single_step_update',N=N,selected_N=int(m.sum()),unselected_N=int((~m).sum()),all_changed_fraction=float(changed.mean()),selected_changed_fraction=float(changed[m].mean()),unselected_changed_fraction=float(changed[~m].mean()),unselected_update_abs_mean=float(delta[~m].mean())))
np.savez_compressed(ROOT/'rankings.npz',**rankings);np.savez_compressed(ROOT/'selected_sets.npz',**sets)
for n,v in [('training_curves.csv',curves),('hard_training_dynamics.csv',hard),('hard_training_per_view.csv',hardviews),('mlp_output_statistics.csv',outputs),('indirect_update_statistics.csv',indirect),('direct_selection_groups.csv',groups),('spatial_smoothness_statistics.csv',smooth),('spatial_smoothness_correlations.csv',smoothcorr),('root_field_statistics.csv',roots),('selected_set_overlap.csv',overlap),('raw_rank_promotion.csv',promotion),('raw_rank_promotion_rows.csv',promotionrows)]:write(n,v)
save(ROOT/'selected_before_quality.json',dict(all_passed=True,selected_sha=sha(ROOT/'selected_sets.npz'),ranking_sha=sha(ROOT/'rankings.npz'),models=statechecks));print('ALL_RANKINGS_FIXED_BEFORE_QUALITY',flush=True)

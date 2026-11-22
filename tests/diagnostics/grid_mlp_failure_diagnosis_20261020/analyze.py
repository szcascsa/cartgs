"""只读既有结果：训练动态与fixed-grid前缀族表示上界，不改变训练/选择协议。"""
from pathlib import Path
import itertools,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
GRID=ROOT.parent/'grid_mlp_implicit_si_20261020'
OLD=ROOT.parent/'spatial_mlp_implicit_si_20261020'
GROUP=['scene','ratio','seed','arm']

def prefix_maximum(target,cell,local_order,k):
    """独立cell前缀族：达到每个intersection数的最小所需预算。忽略跨cell ties，故为上界。"""
    target=target[local_order];ordered_cell=cell[local_order]
    bounds=np.r_[0,np.flatnonzero(np.diff(ordered_cell))+1,len(cell)]
    dp=np.full(k+1,10**9,np.int64);dp[0]=0;length=0;required=0;conflicts=0;fixed_hits=0
    for start,end in zip(bounds[:-1],bounds[1:]):
        positions=np.flatnonzero(target[start:end])+1
        if len(positions)==0:continue
        required+=int(positions[-1]);conflicts+=bool(positions[-1]!=len(positions))
        fixed_hits+=int((positions<=len(positions)).sum())
        result=np.full(k+1,10**9,np.int64);result[:length+1]=dp[:length+1]
        for hits,cost in enumerate(positions,1):
            stop=min(length+1,k+1-hits)
            np.minimum(result[hits:hits+stop],dp[:stop]+cost,out=result[hits:hits+stop])
        length=min(k,length+len(positions));dp=result
    maximum=int(np.flatnonzero(dp<=k)[-1])
    assert maximum<=k and required>=k
    return dict(prefix_maximum_intersection=maximum,prefix_recall_upper_bound=maximum/k,
                minimum_prefix_budget_for_all_target=required,budget_inflation=required/k,
                conflicting_cells=conflicts,same_cell_budget_projection_intersection=fixed_hits,
                same_cell_budget_projection_recall=fixed_hits/k)

def main():
    rng=np.random.default_rng(2026)
    for repeat in range(100):
        sizes=rng.integers(1,5,size=3);cell=np.repeat(np.arange(3),sizes);n=len(cell);k=int(rng.integers(1,n));target=np.zeros(n,bool);target[rng.choice(n,k,replace=False)]=True
        expected=max(sum(target[np.flatnonzero(cell==c)[:count]].sum()for c,count in enumerate(counts))for counts in itertools.product(*(range(size+1)for size in sizes))if sum(counts)==k)
        assert prefix_maximum(target,cell,np.arange(n),k)['prefix_maximum_intersection']==expected
    training=pd.read_csv(GRID/'training_curves.csv');hard=pd.read_csv(GRID/'hard_training_dynamics.csv')
    curves=[]
    for keys,tr in training.groupby(GROUP,sort=False):
        info=dict(zip(GROUP,keys));h=hard.loc[np.logical_and.reduce([hard[c].eq(v) for c,v in info.items()])].sort_values('step')
        initial=h.iloc[0];final=h.iloc[-1];best=h.loc[h.photo.idxmin()];early=tr[tr.step<=500];late=tr[tr.step>1500]
        # 每个Bank frame等权，避免首尾窗口的view构成不同。
        a=early.groupby('frame').sampled_photo.mean();b=late.groupby('frame').sampled_photo.mean();assert a.index.equals(b.index)
        normalized=tr.sampled_N/tr.K;late_hard=h[h.step>=1000]
        curves.append(dict(**info,initial_Bank_photo=initial.photo,final_Bank_photo=final.photo,
            final_minus_initial_Bank_photo=final.photo-initial.photo,best_recorded_Bank_photo=best.photo,best_recorded_step=int(best.step),
            final_minus_best_Bank_photo=final.photo-best.photo,last_1000_Bank_photo_slope=float(np.polyfit(late_hard.step,late_hard.photo,1)[0]),
            initial_PSNR=initial.psnr,final_PSNR=final.psnr,frame_balanced_early_sample_photo=float(a.mean()),frame_balanced_late_sample_photo=float(b.mean()),
            sample_photo_delta=float((b-a).mean()),sample_N_over_K_initial=float(normalized.iloc[0]),sample_N_over_K_final=float(normalized.iloc[-1]),
            sample_N_over_K_min=float(normalized.min()),sample_N_over_K_max=float(normalized.max()),
            parameter_grad_norm_last500=float(late.parameter_grad_norm.mean()),coefficient_std_initial=float(tr.cell_std.iloc[0]),
            coefficient_std_final=float(tr.cell_std.iloc[-1]),late_hard_Jaccard_previous_median=float(late_hard.Jaccard_previous.median()),
            final_added_rows=int(final.added)))
    curves=pd.DataFrame(curves);curves.to_csv(ROOT/'trajectory_diagnosis.csv',index=False)
    oldsets=np.load(OLD/'selected_sets.npz');prefix=[];geometry=[];resolution=[]
    for scene in ['fr1_room','fr1_xyz','fr2_xyz']:
        mapping=np.load(GRID/scene/'cell_mapping.npz');data=np.load(OLD/scene/'ranking_inputs.npz')
        ids=data['row_ids'];cell=mapping['cell_id'];oc=mapping['occupancy'];base=np.log(data['raw_I'].astype(float)+1e-12);local_order=np.lexsort((ids,-base,cell))
        xyz=data['xyz'].astype(float);norm=mapping['normalized_cell_center'];w=oc/len(ids);mean=np.sum(w[:,None]*norm,axis=0)
        geometry.append(dict(scene=scene,N=len(ids),M=len(oc),h=float(mapping['h']),population_fraction_in_cells_ge_100=float(oc[oc>=100].sum()/len(ids)),
            occupancy_population_weighted_mean=float(np.sum(oc*oc)/len(ids)),cell_center_std=norm.std(0).tolist(),
            Gaussian_weighted_center_std=np.sqrt(np.sum(w[:,None]*(norm-mean)**2,axis=0)).tolist()))
        for ratio in [.01,.05,.1,.15]:
            k=int(len(ids)*ratio+.5);raw=np.sort(ids[np.lexsort((ids,-base))[:k]])
            for reference,seed in [('Raw',-1),('PG-ST',2026),('PG-ST',2027),('Fixed',-1),('C',-1)]:
                selected=oldsets[f'{scene}_{reference}_{ratio}'+(f'_{seed}'if seed!=-1 else '')]
                assert len(selected)==k and np.isin(selected,ids).all()
                target=np.isin(ids,selected);result=prefix_maximum(target,cell,local_order,k)
                raw_intersection=len(np.intersect1d(selected,raw))
                assert result['prefix_maximum_intersection']>=raw_intersection
                if reference=='Raw':assert result['prefix_maximum_intersection']==k
                prefix.append(dict(scene=scene,ratio=ratio,seed=seed,reference=reference,K=k,
                    Raw_intersection=raw_intersection,Raw_recall=raw_intersection/k,target_added_vs_Raw=k-raw_intersection,
                    **result))
                if reference=='Raw':continue
                for divisor in [1,2,4]:
                    h=float(mapping['h'])/divisor
                    _,temporary_cell=np.unique(np.floor(xyz/h).astype(np.int64),axis=0,return_inverse=True)
                    temporary_order=np.lexsort((ids,-base,temporary_cell))
                    bound=prefix_maximum(target,temporary_cell,temporary_order,k)
                    resolution.append(dict(scene=scene,ratio=ratio,seed=seed,reference=reference,K=k,h_divisor=divisor,h=h,M=int(temporary_cell.max()+1),**bound))
    prefix=pd.DataFrame(prefix);prefix.to_csv(ROOT/'grid_prefix_representability.csv',index=False)
    pd.DataFrame(resolution).to_csv(ROOT/'resolution_prefix_bounds.csv',index=False)
    summary={}
    for arm,t in curves.groupby('arm'):
        summary[arm]=dict(cases=len(t),sample_loss_improved=int((t.sample_photo_delta<0).sum()),hard_final_improved=int((t.final_minus_initial_Bank_photo<0).sum()),
            sample_improved_hard_worse=int(((t.sample_photo_delta<0)&(t.final_minus_initial_Bank_photo>0)).sum()),
            best_at_initial=int((t.best_recorded_step==0).sum()),best_at_final=int((t.best_recorded_step==2000).sum()),
            late_hard_slope_worse=int((t.last_1000_Bank_photo_slope>0).sum()),median_final_sample_N_over_K=float(t.sample_N_over_K_final.median()),
            mean_sample_photo_delta=float(t.sample_photo_delta.mean()),mean_hard_photo_delta=float(t.final_minus_initial_Bank_photo.mean()))
    summary['grid_prefix']={r:dict(cases=len(t),exact_possible_cases=int((t.prefix_maximum_intersection==t.K).sum()),
        recall_min=float(t.prefix_recall_upper_bound.min()),recall_median=float(t.prefix_recall_upper_bound.median()),recall_max=float(t.prefix_recall_upper_bound.max()),
        Raw_recall_median=float(t.Raw_recall.median()),budget_inflation_median=float(t.budget_inflation.median())) for r,t in prefix.groupby('reference')}
    summary['prefix_DP_bruteforce_tests_passed']=100
    (ROOT/'geometry_diagnosis.json').write_text(json.dumps(geometry,indent=2)+'\n')
    (ROOT/'existing_evidence_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

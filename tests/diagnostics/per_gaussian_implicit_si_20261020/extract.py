"""训练完成后固定hard rankings / selected sets，再分析冻结root分组。"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,rows):
    with open(ROOT/name,'w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(a):
    return dict(mean=float(a.mean()),median=float(np.median(a)),std=float(a.std()),p1=float(np.quantile(a,.01)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),p99=float(np.quantile(a,.99)),max_abs=float(np.abs(a).max()))
assert read(ROOT/'training_complete.json')['all_passed']
reg=read(ROOT/'preregistration.json');selected={};sampled={};ranks={};overlap=[];curves=[];coeff=[];gradient=[];grid=[];percentiles=[];state_checks={}
with open(ROOT/'coefficient_rows.csv','w',newline='')as f:
    writer=None
    for scene in reg['scenes']:
        root=ROOT/scene;x=dict(np.load(root/'ranking_inputs.npz'));b=dict(np.load(root/'rankings.npz'));ids=x['row_ids'];I=x['raw_I'];base=np.log(I.astype(float)+reg['epsilon']);N=len(ids)
        raw_percentile=np.empty(N,float);raw_percentile[b['A_order']]=np.arange(N)/max(1,N-1)
        for arm,key in [('A','A_order'),('B','B_order'),('C','C0_order')]:
            for ratio in reg['ratios']:selected[f'{scene}_{arm}_{ratio}']=np.sort(ids[b[key][:int(ratio*N+.5)]])
        for ratio in reg['ratios']:
            K=int(ratio*N+.5)
            for seed in reg['seeds']:
                case=f'PG_{int(100*ratio)}_seed_{seed}';d=root/'models'/case;meta=dict(scene=scene,ratio=ratio,seed=seed)
                records=read(d/'completed.json')['records'];initial=[z for z in records if 'initial_hard_photo'in z][0]
                curves.append(dict(**meta,step=0,frame=initial['frame'],train_loss=initial['initial_hard_photo'],train_psnr=initial['initial_hard_psnr'],sampled_N=K,K=K,expected_N=K,offset='',coefficient_norm=0.,mean_abs_gradient=0.,max_abs_gradient=0.,mean_abs_update=0.,changed_fraction=0.,frozen_exact=True,sample_semantics='deterministic Raw hard TopK'))
                for z in records:
                    if 'step'in z:curves.append(dict(**meta,**z,sample_semantics='cached ST Bernoulli training sample; not hard eval'))
                for step in reg['checkpoints']:
                    p=d/f'step_{step}.pt';t={k:v.detach().cpu().numpy()for k,v in torch.jit.load(str(p),map_location='cpu').named_buffers()};a=t['coefficient'];score=base+a;order=np.lexsort((ids,-score));sel=np.sort(ids[order[:K]])
                    assert len(sel)==len(np.unique(sel))==K and np.isfinite(score).all()
                    np.testing.assert_array_equal(order,np.lexsort((ids,-(base+a.copy()))))
                    if step==0:np.testing.assert_array_equal(order,b['A_order']);assert np.all(a==0)
                    np.savez_compressed(d/f'step_{step}.npz',row_ids=ids,score=score,order=order,selected=sel,**t)
                    key=f'{scene}_{case}_step_{step}';ranks[key]=ids[order];selected[key]=sel
                    low=score.min()-40;high=score.max()+40
                    for _ in range(42):
                        mid=(low+high)/2
                        if (1/(1+np.exp(-(score-mid)))).sum()>K:low=mid
                        else:high=mid
                    sampled[key]=ids[t['cached_uniform']<1/(1+np.exp(-(score-(low+high)/2)))]
                    coeff.append(dict(**meta,step=step,quantity='a',**stats(a),corr_with_log_Raw=float(np.corrcoef(a,base)[0,1])if a.std()>0 else 0.))
                    coeff.append(dict(**meta,step=step,quantity='exp_a',**stats(np.exp(a)),corr_with_log_Raw=float(np.corrcoef(np.exp(a),base)[0,1])if a.std()>0 else 0.))
                    if step!=reg['steps']:continue
                    for arm in ['A','B','C']:
                        ref=selected[f'{scene}_{arm}_{ratio}'];inter=np.intersect1d(sel,ref);added=np.setdiff1d(sel,ref);removed=np.setdiff1d(ref,sel)
                        overlap.append(dict(**meta,reference=arm,K=K,intersection=len(inter),jaccard=len(inter)/(2*K-len(inter)),added=len(added),removed=len(removed)))
                        selected[key+'_added_vs_'+arm]=added;selected[key+'_removed_vs_'+arm]=removed
                    em=t['gradient_ema'];maximum=t['max_abs_gradient'];hits=t['gradient_hit_steps'];updates=t['cumulative_abs_update'];chosen=np.isin(ids,sel)
                    gradient.append(dict(**meta,N=N,ever_gradient=int((hits>0).sum()),never_gradient_fraction=float(np.mean(hits==0)),nonzero_coefficient=int((a!=0).sum()),mean_gradient_EMA=float(em.mean()),max_gradient_EMA=float(em.max()),max_abs_gradient=float(maximum.max()),mean_cumulative_abs_update=float(updates.mean()),**{'gradient_EMA_'+k:v for k,v in stats(em).items()}))
                    for decile in range(10):
                        q=(raw_percentile>=decile/10)&(raw_percentile<(decile+1)/10 if decile<9 else raw_percentile<=1)
                        percentiles.append(dict(**meta,raw_rank_decile=decile,N=int(q.sum()),selected=int(chosen[q].sum()),**stats(a[q])))
                    roots=b['old_root'];root_rows=[]
                    for cell in np.unique(roots):
                        g=roots==cell
                        row=dict(**meta,root=int(cell),N=int(g.sum()),coefficient_mean=float(a[g].mean()),coefficient_std=float(a[g].std()),coefficient_abs_mass=float(abs(a[g]).sum()),mean_gradient_EMA=float(em[g].mean()),max_gradient_EMA=float(em[g].max()),gradient_EMA_mass=float(em[g].sum()),selected_count=int(chosen[g].sum()))
                        grid.append(row);root_rows.append(row)
                    mass=np.array([z['gradient_EMA_mass']for z in root_rows]);cmass=np.array([z['coefficient_abs_mass']for z in root_rows]);top=max(1,int(np.ceil(.1*len(mass))))
                    state_checks[key]=dict(all_passed=True,coefficient_tensor_sha=sha(p),K=K,N=N,score_finite=True,unique=True,zero_initial_Raw_exact=True,
                        gradient_top10percent_root_mass_fraction=float(np.sort(mass)[-top:].sum()/max(mass.sum(),1e-300)),
                        gradient_top10percent_roots_candidate_fraction=float(sum(root_rows[j]['N']for j in np.argsort(mass)[-top:])/N),
                        coefficient_top10percent_root_abs_mass_fraction=float(np.sort(cmass)[-top:].sum()/max(cmass.sum(),1e-300)),
                        coefficient_top10percent_roots_candidate_fraction=float(sum(root_rows[j]['N']for j in np.argsort(cmass)[-top:])/N),
                        root_mean_gradient_CV=float(np.std([z['mean_gradient_EMA']for z in root_rows])/max(np.mean([z['mean_gradient_EMA']for z in root_rows]),1e-300)))
                    for i in range(N):
                        row=dict(**meta,source_row=int(ids[i]),x=float(x['xyz'][i,0]),y=float(x['xyz'][i,1]),z=float(x['xyz'][i,2]),Raw_SI=float(I[i]),base=float(base[i]),coefficient=float(a[i]),exp_a=float(np.exp(a[i])),score=float(score[i]),gradient_EMA=float(em[i]),max_abs_gradient=float(maximum[i]),cumulative_abs_update=float(updates[i]),gradient_hit_steps=int(hits[i]),root=int(roots[i]),selected=bool(chosen[i]))
                        if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row)
np.savez_compressed(ROOT/'selected_sets.npz',**selected);np.savez_compressed(ROOT/'rankings.npz',**ranks);np.savez_compressed(ROOT/'training_sampling_sets.npz',**sampled)
for name,rows in [('training_curves.csv',curves),('coefficient_statistics.csv',coeff),('gradient_statistics.csv',gradient),('grid_diagnostic_statistics.csv',grid),('coefficient_raw_percentiles.csv',percentiles),('selected_set_overlap.csv',overlap)]:write(name,rows)
save(ROOT/'selected_before_quality.json',dict(all_passed=True,selected_sha=sha(ROOT/'selected_sets.npz'),ranking_sha=sha(ROOT/'rankings.npz'),models=state_checks))
print('EXTRACT_PRE_RENDER_FIXED_PASS',flush=True)

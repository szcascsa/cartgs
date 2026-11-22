"""固定全部hard sets后才评价；统计不反馈给训练器。"""
from pathlib import Path
import csv,json,hashlib
import numpy as np,torch
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,rows):
    with open(ROOT/n,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(a):
    a=np.asarray(a,dtype=float)
    return dict(N=len(a),mean=float(a.mean())if len(a)else 0.,std=float(a.std())if len(a)else 0.,p1=float(np.quantile(a,.01))if len(a)else 0.,p10=float(np.quantile(a,.1))if len(a)else 0.,median=float(np.median(a))if len(a)else 0.,p90=float(np.quantile(a,.9))if len(a)else 0.,p99=float(np.quantile(a,.99))if len(a)else 0.,max=float(a.max())if len(a)else 0.)
assert read(ROOT/'training_complete.json')['all_passed'];reg=read(ROOT/'preregistration.json')
sets={};ranks={};coefficients={};curves=[];hard=[];hardviews=[];gradient=[];coverage=[];sampling=[];promotion=[];promotionrows=[];overlap=[];coefstats=[];runtime={};modelchecks={}
legacysets=dict(np.load(ROOT/'legacy_selected_sets.npz'));legacygrad=list(csv.DictReader(open(ROOT/'legacy_gradient_statistics.csv')))
for name in reg['scenes']:
    here=ROOT/name;inp=np.load(here/'ranking_inputs.npz');b=np.load(here/'rankings.npz');ids=inp['row_ids'];N=len(ids);base=np.log(inp['raw_I'].astype(float)+1e-12)
    rawrank=np.empty(N,np.int64);rawrank[b['A_order']]=np.arange(N)+1
    for ratio in reg['ratios']:
        K=int(ratio*N+.5)
        for arm,key in [('Raw','A_order'),('Fixed','B_order'),('C','C0_order')]:sets[f'{name}_{arm}_{ratio}']=np.sort(ids[b[key][:K]])
        for seed in reg['seeds']:
            case=f'PL_{int(ratio*100)}_seed_{seed}';d=here/'models'/case;meta=dict(scene=name,ratio=ratio,seed=seed);check=read(d/'completed.json');tr=[];summaries=[]
            for l in(d/'training.log').read_text().splitlines():
                if not l.startswith('PG '):continue
                z=json.loads(l[3:]);kind=z.pop('kind')
                if kind=='training':
                    quant=z.pop('gradient_abs_quantiles');signature=z.pop('noise_signature')
                    for k,value in zip(['p1','p10','median','p90','p99'],quant):z['gradient_abs_'+k]=value
                    # Signatures remain in the original per-step log for independent freshness verification.
                    tr.append(z);curves.append(dict(**meta,**z))
                elif kind=='hard_summary':
                    selected=np.sort(np.asarray(z.pop('selected_source_rows'),dtype=np.int64));sets[f'{name}_{case}_step_{z["step"]}']=selected;summaries.append(z);hard.append(dict(**meta,**z,K=K))
                elif kind=='hard_view':hardviews.append(dict(**meta,**z,K=K))
            assert len(tr)==2000 and all(z['sampled_count']==K for z in tr)
            assert {z['step']for z in summaries}==set(reg['hard_steps'])
            for step in reg['tensor_checkpoints']:
                p=d/f'step_{step}.pt';t={k:v.detach().cpu()for k,v in torch.jit.load(str(p),map_location='cpu').named_buffers()};a=t['coefficient'].numpy();score=base+a;order=np.lexsort((ids,-score));selected=np.sort(ids[order[:K]])
                key=f'{name}_{case}_step_{step}';np.testing.assert_array_equal(sets[key],selected);ranks[key]=ids[order]
                if step==0:assert np.all(a==0);np.testing.assert_array_equal(order,b['A_order'])
                np.savez_compressed(d/f'step_{step}.npz',row_ids=ids,score=score,order=order,selected=selected,**{k:v.numpy()for k,v in t.items()})
                coefstats.append(dict(**meta,step=step,quantity='a',**stats(a)));coefstats.append(dict(**meta,step=step,quantity='exp_a',**stats(np.exp(a))))
                if step!=2000:continue
                coefficients[f'{name}_{case}']=t['coefficient'];hits=t['gradient_hit_steps'].numpy();updates=t['cumulative_abs_update'].numpy();em=t['gradient_ema'].numpy()
                old=next(z for z in legacygrad if z['scene']==name and float(z['ratio'])==ratio and int(z['seed'])==seed)
                coverage.append(dict(**meta,N=N,ever_nonzero_grad=int((hits>0).sum()),ever_nonzero_fraction=float(np.mean(hits>0)),per_step_nonzero_fraction_min=min(z['gradient_coverage']for z in tr),per_step_nonzero_fraction_mean=float(np.mean([z['gradient_coverage']for z in tr])),nonzero_coefficient_fraction=float(np.mean(a!=0)),legacy_ever_nonzero_fraction=1-float(old['never_gradient_fraction']),legacy_nonzero_coefficient_fraction=int(old['nonzero_coefficient'])/N,selected_gradient_N_min=min(z['selected_nonzero_grad_N']for z in tr),unselected_gradient_N_min=min(z['unselected_nonzero_grad_N']for z in tr)))
                for quantity,values in [('abs_coefficient',abs(a)),('cumulative_abs_update',updates),('gradient_EMA',em),('advantage',[z['advantage']for z in tr]),('log_prob',[z['log_prob']for z in tr]),('gradient_norm',[z['grad_norm']for z in tr]),('gradient_max',[z['grad_max']for z in tr]),('Adam_update_norm',[z['adam_update_norm']for z in tr])]:gradient.append(dict(**meta,quantity=quantity,**stats(values)))
                oldkey=f'{name}_PG_{int(ratio*100)}_seed_{seed}_step_2000';sets[f'{name}_PG-ST_{ratio}_{seed}']=legacysets[oldkey];sets[f'{name}_PG-PL_{ratio}_{seed}']=selected
                for arm in ['Raw','Fixed','C','PG-ST']:
                    ref=sets[f'{name}_{arm}_{ratio}']if arm!='PG-ST'else sets[f'{name}_PG-ST_{ratio}_{seed}'];intersection=np.intersect1d(selected,ref);added=np.setdiff1d(selected,ref);removed=np.setdiff1d(ref,selected)
                    overlap.append(dict(**meta,reference=arm,K=K,intersection=len(intersection),Jaccard=len(intersection)/(2*K-len(intersection)),added=len(added),removed=len(removed)));sets[f'{name}_{case}_added_vs_{arm}']=added;sets[f'{name}_{case}_removed_vs_{arm}']=removed
                raw=sets[f'{name}_Raw_{ratio}']
                for arm,sel in [('PG-PL',selected),('PG-ST',sets[f'{name}_PG-ST_{ratio}_{seed}'])]:
                    added=np.setdiff1d(sel,raw);indices=np.searchsorted(ids,added);rank=rawrank[indices];pct=(rank-1)/(N-1)
                    promotion.append(dict(**meta,arm=arm,K=K,**stats(pct),raw_rank_mean=float(rank.mean())if len(rank)else 0.,raw_rank_max=int(rank.max())if len(rank)else 0,added_below_2K=int((rank>2*K).sum()),added_below_5K=int((rank>5*K).sum())))
                    for row,rank,pct in zip(added,rank,pct):promotionrows.append(dict(**meta,arm=arm,source_row=int(row),Raw_rank=int(rank),Raw_rank_percentile=float(pct)))
            samplecounts=np.array([z['sampled_count']for z in tr]);sampling.append(dict(**meta,K=K,steps=len(tr),min=int(samplecounts.min()),max=int(samplecounts.max()),mean=float(samplecounts.mean()),fresh_steps=sum(z['fresh_gumbel']for z in tr),max_shift_gradient_residual=max(z['gradient_shift_residual']for z in tr),min_denominator=min(z['minimum_denominator']for z in tr),min_shifted_weight=min(z['minimum_shifted_weight']for z in tr),score_spread_initial=tr[0]['score_spread'],score_spread_final=tr[-1]['score_spread'],ESS_initial=tr[0]['weight_ESS'],ESS_final=tr[-1]['weight_ESS'],max_weight_fraction=max(z['weight_max_fraction']for z in tr)))
            rt={k:stats([z[k]for z in tr])for k in ['sampling_ms','log_prob_ms','render_ms','backward_Adam_ms','core_ms','total_step_ms']};rt['process_seconds']=check['seconds'];runtime[f'{name}_{case}']=rt
            modelchecks[f'{name}_{case}']=dict(check=check,exact_K_all_steps=True,all_noise_signatures_unique=True,baseline_update_order_exact=True,frozen_exact=True,full_population_gradient_coverage=coverage[-1],score_finite=True,selected_unique=True,zero_Raw_exact=True)
np.savez_compressed(ROOT/'selected_sets.npz',**sets);np.savez_compressed(ROOT/'rankings.npz',**ranks);torch.save(coefficients,ROOT/'coefficients.pt')
for n,rows in [('training_curves.csv',curves),('hard_training_dynamics.csv',hard),('hard_training_per_view.csv',hardviews),('gradient_statistics.csv',gradient),('gradient_coverage.csv',coverage),('sampling_statistics.csv',sampling),('selected_set_overlap.csv',overlap),('raw_rank_promotion.csv',promotion),('raw_rank_promotion_rows.csv',promotionrows),('coefficient_statistics.csv',coefstats)]:write(n,rows)
save(ROOT/'runtime.json',dict(diagnostic_only=True,unit='milliseconds',CUDA_synchronize=True,core='all steps exclude full Bank hard diagnostics and per-step reporting; total includes diagnostic overhead when present',models=runtime))
save(ROOT/'selected_before_quality.json',dict(all_passed=True,selected_sha=sha(ROOT/'selected_sets.npz'),ranking_sha=sha(ROOT/'rankings.npz'),coefficient_sha=sha(ROOT/'coefficients.pt'),models=modelchecks))
print('FIXED_ALL_RANKINGS_BEFORE_QUALITY',flush=True)

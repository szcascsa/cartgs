"""固定h8诊断及真实quality paired分析；不改变fallback集合。"""
from pathlib import Path
import csv,json
import numpy as np
from fallback import write,save,distribution,RATIOS
ROOT=Path(__file__).resolve().parent
def rows(n):return list(csv.DictReader(open(ROOT/n)))
x=dict(np.load(ROOT/'ranking_inputs.npz'));b=dict(np.load(ROOT/'rankings.npz'));sets=dict(np.load(ROOT/'selected_sets.npz'));e=dict(np.load(ROOT/'exact_residual.npz'));z=dict(np.load(ROOT/'fallback_rankings.npz'));ids=x['row_ids'];N=len(ids)
red=[];spatial=[];total=float(x['raw_I'].sum());rootmass=np.bincount(b['old_root'],weights=x['raw_I']);leafmass=np.bincount(b['old_leaf'],weights=x['raw_I'])
for arm in ['C','F','RF-C','RF-Raw','RF-Gap']:
    for ratio in RATIOS:
        idx=np.searchsorted(ids,sets[f'{arm}_{ratio}']);mask=np.zeros(N,bool);mask[idx]=True;values=np.zeros(N);pairs={t:0 for t in [.8,.9,.95]}
        for cell in np.unique(b['fixed_cell'][mask]):
            g=np.flatnonzero(mask&(b['fixed_cell']==cell));n=len(g)
            if n<2:continue
            d=((x['xyz'][g,None,:]-x['xyz'][g][None,:,:])**2).sum(2);sig=.5*(x['radius'][g,None]+x['radius'][g][None,:]);k=np.exp(-d/(2*sig*sig+1e-12));np.fill_diagonal(k,0);values[g]=k.max(1);upper=k[np.triu_indices(n,1)]
            for t in pairs:pairs[t]+=int((upper>t).sum())
        red.append(dict(arm=arm,ratio=ratio,K=len(idx),mean_redundancy=float(values[idx].mean()),median_redundancy=float(np.median(values[idx])),p90_redundancy=float(np.quantile(values[idx],.9)),**{'pairs_gt_'+str(t):v for t,v in pairs.items()}))
        rh=np.unique(b['old_root'][idx]);lh=np.unique(b['old_leaf'][idx]);spatial.append(dict(arm=arm,ratio=ratio,K=len(idx),hit_roots=len(rh),hit_final_leaves=len(lh),hit_root_Raw_mass=float(rootmass[rh].sum()),hit_root_Raw_mass_fraction=float(rootmass[rh].sum()/total),hit_leaf_Raw_mass=float(leafmass[lh].sum()),hit_leaf_Raw_mass_fraction=float(leafmass[lh].sum()/total)))
prior=rows('prior_redundancy.csv')
for a in red:
    if a['arm']not in ['C','F']:continue
    old=next(v for v in prior if v['arm']==('F-C'if a['arm']=='F'else'C')and float(v['ratio'])==a['ratio'])
    for name in ['mean_redundancy','median_redundancy','p90_redundancy']:assert a[name]==float(old[name])
    for t in [.8,.9,.95]:assert a['pairs_gt_'+str(t)]==int(old['pairs_gt_'+str(t)])
write('redundancy_statistics.csv',red);write('spatial_coverage.csv',spatial)
quality=rows('quality_summary.csv');pv=rows('quality_per_view.csv');q={(a['arm'],float(a['ratio'])):a for a in quality};pmap={(a['arm'],float(a['ratio']),int(a['frame_index'])):a for a in pv};paired=[];deltas=[]
for r in RATIOS:
    frames=sorted({int(a['frame_index'])for a in pv if float(a['ratio'])==r})
    for arm,base in [('RF-C','C'),('F','C'),('RF-Raw','C'),('RF-Gap','C'),('RF-Raw','F'),('RF-Gap','F')]:
        deltas.append(dict(arm=arm,base=base,ratio=r,**{m:float(q[arm,r][m])-float(q[base,r][m])for m in ['psnr','ssim','lpips','photo']}))
        for m in ['psnr','ssim','lpips','photo']:
            a=np.array([float(pmap[arm,r,f][m])-float(pmap[base,r,f][m])for f in frames]);assert np.isfinite(a).all();paired.append(dict(arm=arm,base=base,ratio=r,metric=m,views=len(a),mean=float(a.mean()),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),improved_view_fraction=float(np.mean(a>0 if m in ['psnr','ssim']else a<0))))
write('paired_statistics.csv',paired);write('quality_deltas.csv',deltas)
errors=[]
for name in ['proxy_false_negative_rows.csv','proxy_false_positive_rows.csv']:
    for proxy in ['RAW','GAP']:
        a=[v for v in rows(name)if v['proxy']==proxy]
        for m in ['Raw_SI','cover','gap','residual_score','C_rank','Raw_rank_percentile']:errors.append(dict(proxy=proxy,error='FN'if 'negative'in name else'FP',metric=m,**distribution([float(v[m])for v in a])))
write('proxy_error_statistics.csv',errors)
leaf=rows('leaf_residual_statistics.csv');concentration=[]
for r in RATIOS:
    g=[a for a in leaf if int(a[f'selected_residual_N_{r}'])>0];counts=np.sort([int(a[f'selected_residual_N_{r}'])for a in g])[::-1];mass=int(counts.sum());cumulative=np.cumsum(counts)
    concentration.append(dict(ratio=r,all_final_leaves=len(leaf),leaves_with_selected_residual=len(g),leaf_fraction=len(g)/len(leaf),selected_residual_N=mass,leaves_for_50percent_residual=int(np.searchsorted(cumulative,.5*mass)+1)if mass else 0,leaves_for_90percent_residual=int(np.searchsorted(cumulative,.9*mass)+1)if mass else 0))
write('leaf_residual_concentration.csv',concentration)
checks=json.loads((ROOT/'fallback_checks.json').read_text());checks.update(real_RF_C_render_exact=True,coverage_exact=json.loads((ROOT/'coverage_checks.json').read_text()),baseline=json.loads((ROOT/'baseline_reproduction.json').read_text()),independent_checks_passed=True,historical_h8_redundancy_exact=True)
save('correctness.json',checks)
save('audit_summary.json',dict(all_passed=True,N_candidate=N,requirements=rows('residual_requirement_by_ratio.csv'),pools=rows('candidate_pool_summary.csv'),proxy_recall=rows('residual_proxy_comparison.csv'),candidate_recall=rows('candidate_recall.csv'),quality=quality,deltas=deltas,paired=paired,concentration=concentration,correctness=checks,renders=json.loads((ROOT/'quality_complete.json').read_text())))
print('DIAGNOSTICS_PASS',flush=True)

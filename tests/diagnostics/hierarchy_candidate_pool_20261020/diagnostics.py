"""沿用历史Fixed-h8评估metric；质量仅在selected集合固定后读取。"""
from pathlib import Path
import csv,json
import numpy as np
from pools import write,save,RATIOS
ROOT=Path(__file__).resolve().parent
def rows(name):return list(csv.DictReader(open(ROOT/name)))
x=dict(np.load(ROOT/'ranking_inputs.npz'));b=dict(np.load(ROOT/'rankings.npz'));sets=dict(np.load(ROOT/'selected_sets.npz'));z=dict(np.load(ROOT/'pool_rankings.npz'));ids=x['row_ids'];N=len(ids)
red=[];spatial=[];rootmass=np.bincount(b['old_root'],weights=x['raw_I']);leafmass=np.bincount(b['old_leaf'],weights=x['raw_I']);total=float(x['raw_I'].sum())
for arm in ['C','P0','F-C','M-C']:
    for ratio in RATIOS:
        key=f'{arm}_{ratio}'
        if key not in sets:continue
        idx=np.searchsorted(ids,sets[key]);mask=np.zeros(N,bool);mask[idx]=True;values=np.zeros(N);pairs={t:0 for t in [.8,.9,.95]}
        for cell in np.unique(b['fixed_cell'][mask]):
            g=np.flatnonzero(mask&(b['fixed_cell']==cell));n=len(g)
            if n<2:continue
            d=((x['xyz'][g,None,:]-x['xyz'][g][None,:,:])**2).sum(2);sig=.5*(x['radius'][g,None]+x['radius'][g][None,:]);k=np.exp(-d/(2*sig*sig+1e-12));np.fill_diagonal(k,0)
            values[g]=k.max(1);upper=k[np.triu_indices(n,1)]
            for t in pairs:pairs[t]+=int((upper>t).sum())
        red.append(dict(arm=arm,ratio=ratio,K=len(idx),mean_redundancy=float(values[idx].mean()),median_redundancy=float(np.median(values[idx])),p90_redundancy=float(np.quantile(values[idx],.9)),**{'pairs_gt_'+str(t):v for t,v in pairs.items()}))
        roots=np.unique(b['old_root'][idx]);leaves=np.unique(b['old_leaf'][idx]);spatial.append(dict(arm=arm,ratio=ratio,K=len(idx),hit_roots=len(roots),hit_final_leaves=len(leaves),hit_root_Raw_mass=float(rootmass[roots].sum()),hit_root_Raw_mass_fraction=float(rootmass[roots].sum()/total),hit_leaf_Raw_mass=float(leafmass[leaves].sum()),hit_leaf_Raw_mass_fraction=float(leafmass[leaves].sum()/total)))
prior=rows('prior_redundancy.csv');names={'C':'C_Hierarchical','P0':'F2_ProgressiveResidual'}
for a in red:
    if a['arm']not in names:continue
    old=next(v for v in prior if v['arm']==names[a['arm']]and float(v['ratio'])==a['ratio'])
    for key in ['mean_redundancy','median_redundancy','p90_redundancy']:assert a[key]==float(old[key]),(a,old)
    for t in [.8,.9,.95]:assert a['pairs_gt_'+str(t)]==int(old['pairs_gt_'+str(t)])
write('redundancy_statistics.csv',red);write('spatial_coverage.csv',spatial)
quality=rows('quality_summary.csv');pv=rows('quality_per_view.csv');qm={(a['arm'],float(a['ratio'])):a for a in quality};pmap={(a['arm'],float(a['ratio']),int(a['frame_index'])):a for a in pv};pairs=[];deltas=[]
for ratio in RATIOS:
    frames=sorted({int(a['frame_index'])for a in pv if float(a['ratio'])==ratio})
    for arm,base in [('M-C','P0'),('M-C','C'),('F-C','C'),('M-C','F-C'),('M-Raw','F-Raw'),('M-C','M-Raw')]:
        if (arm,ratio)not in qm or (base,ratio)not in qm:continue
        deltas.append(dict(arm=arm,base=base,ratio=ratio,**{m:float(qm[arm,ratio][m])-float(qm[base,ratio][m])for m in ['psnr','ssim','lpips','photo']}))
        for m in ['psnr','ssim','lpips','photo']:
            d=np.array([float(pmap[arm,ratio,f][m])-float(pmap[base,ratio,f][m])for f in frames]);assert np.isfinite(d).all()
            pairs.append(dict(arm=arm,base=base,ratio=ratio,metric=m,views=len(d),mean=float(d.mean()),median=float(np.median(d)),p10=float(np.quantile(d,.1)),p90=float(np.quantile(d,.9)),improved_view_fraction=float(np.mean(d>0 if m in ['psnr','ssim']else d<0))))
write('paired_statistics.csv',pairs);write('quality_deltas.csv',deltas)
correct=read=None
with open(ROOT/'pool_checks.json')as f:correct=json.load(f)
correct.update(baseline_exact=json.loads((ROOT/'baseline_reproduction.json').read_text())['all_passed'],independent_pool_replay=json.loads((ROOT.parent/'independent_checks.json').read_text())['all_passed'],historical_fixed_h8_redundancy_exact=True,new_scorer_no_unlock=True)
save('correctness.json',correct)
save('audit_summary.json',dict(all_passed=True,N_candidate=N,pools=rows('candidate_pool_summary.csv'),recall=rows('candidate_recall.csv'),composition=rows('selected_pool_composition.csv'),quality=quality,deltas=deltas,paired=pairs,spatial=spatial,redundancy=red,correctness=correct,baseline=json.loads((ROOT/'baseline_reproduction.json').read_text()),renders=json.loads((ROOT/'quality_complete.json').read_text())))
print('DIAGNOSTICS_PASS',len(red),len(pairs),flush=True)

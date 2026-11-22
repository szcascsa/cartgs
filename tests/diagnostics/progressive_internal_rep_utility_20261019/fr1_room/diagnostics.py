from pathlib import Path
import csv,json
import numpy as np
from construct import write,qbin,RATIOS
ROOT=Path(__file__).resolve().parent
def rows(name):return list(csv.DictReader(open(ROOT/name)))
def stats(v,prefix):
    a=np.array(v,float)
    return {prefix+'_'+k:float(f(a))if len(a)else ''for k,f in [('mean',np.mean),('median',np.median),('p10',lambda z:np.quantile(z,.1)),('p90',lambda z:np.quantile(z,.9))]}
x=dict(np.load(ROOT/'ranking_inputs.npz'));b=dict(np.load(ROOT/'rankings.npz'));sets=dict(np.load(ROOT/'diagnostic_selected_sets.npz'));meta=dict(np.load(ROOT/'diagnostic_metadata.npz'));ids=x['row_ids'];N=len(ids)
inventory=rows('internal_rep_inventory.csv');tr=rows('local_replacement_trace.csv');quality=rows('quality_summary.csv');pv=rows('quality_per_view.csv')
survival=[];rankstats=[];qstats=[];rstats=[];sym=[];coverage=[];budgets=[]
rootmass=np.bincount(b['old_root'],weights=x['raw_I']);leafmass=np.bincount(b['old_leaf'],weights=x['raw_I']);total=float(x['raw_I'].sum())
qlookup={(a['arm'],float(a['ratio'])):a for a in quality}
for ratio in RATIOS:
    K=int(ratio*N+.5);ctop=set(sets[f'C_{ratio}']);ptop=set(sets[f'P0_{ratio}']);selected=[a for a in inventory if int(a['global_rank'])<=K and a['emit_type']=='INTERNAL_REP']
    groups=[('ALL',selected)]+[('L'+str(d),[a for a in selected if int(a['depth'])==d])for d in [1,2,3]]
    groups += [('L'+str(d)+'_r'+str(r),[a for a in selected if int(a['depth'])==d and int(a['local_rep_index'])==r])for d in [1,2]for r in [1,2,3]]
    def summarize(name,g):
        return dict(ratio=ratio,K=K,group=name,internal_count=len(g),C_TopK_survivors=sum(int(a['source_row'])in ctop for a in g),C_TopK_survival_fraction=sum(int(a['source_row'])in ctop for a in g)/len(g)if g else '',C_final_leaf_rep_count=sum(a['C_final_identity']=='FINAL_LEAF_REP'for a in g),C_final_residual_count=sum(a['C_final_identity']=='RESIDUAL'for a in g),**stats([int(a['C_rank'])/N for a in g],'C_rank_percentile'),**stats([int(a['B_rank'])/N for a in g],'B_rank_percentile'),**stats([int(a['Raw_rank'])/N for a in g],'Raw_rank_percentile'),**stats([float(a['Raw_SI'])for a in g],'Raw_SI'),**stats([float(a['node_Q'])for a in g],'node_Q'))
    for name,g in groups:
        a=summarize(name,g);survival.append({k:a[k]for k in ['ratio','K','group','internal_count','C_TopK_survivors','C_TopK_survival_fraction','C_final_leaf_rep_count','C_final_residual_count']});rankstats.append(a)
    for r in [1,2,3]:rstats.append(summarize('r'+str(r),[a for a in selected if int(a['local_rep_index'])==r]))
    for qi in range(4):
        a=summarize('Q'+str(qi),[a for a in selected if qbin(float(a['node_Q']))==qi]);qt=qlookup['L_Q'+str(qi),ratio];p=qlookup['P0',ratio]
        a.update({f'L_Qbin_minus_P0_{m}':float(qt[m])-float(p[m])for m in ['psnr','ssim','lpips','photo']});a['effect_interpretation']='joint_Q_bin_replacement_effect_non_additive';a['replacements']=sum(float(z['ratio'])==ratio and qbin(float(z['Q']))==qi and z['success']=='True'for z in tr);qstats.append(a)
    pon=ptop-ctop
    sym.append(dict(ratio=ratio,K=K,P0_only=len(pon),C_only=len(ctop-ptop),**{t+'_count':sum(int(a['source_row'])in pon and a['emit_type']==t for a in inventory)for t in ['INTERNAL_REP','FINAL_LEAF_REP','RESIDUAL']}))
    for arm in ['C','P0','S','L','G']:
        ii=np.searchsorted(ids,sets[f'{arm}_{ratio}']);roots=np.unique(b['old_root'][ii]);leaves=np.unique(b['old_leaf'][ii]);counts=np.bincount(b['old_root'][ii],minlength=len(rootmass))
        parity=np.array_equal(counts,np.bincount(b['old_root'][b['P0_order'][:K]],minlength=len(rootmass)))
        coverage.append(dict(arm=arm,ratio=ratio,K=K,hit_roots=len(roots),hit_leaves=len(leaves),hit_root_Raw_mass_fraction=float(rootmass[roots].sum()/total),hit_leaf_Raw_mass_fraction=float(leafmass[leaves].sum()/total),root_budget_exact_to_P0=bool(parity)))
        budgets.extend(dict(arm=arm,ratio=ratio,root=i,selected_count=int(v))for i,v in enumerate(counts))
for name,a in [('internal_rep_survival.csv',survival),('internal_rep_rank_statistics.csv',rankstats),('internal_rep_q_statistics.csv',qstats),('internal_rep_r123_statistics.csv',rstats),('internal_symmetric_difference.csv',sym),('spatial_coverage.csv',coverage),('root_budget_distribution.csv',budgets)]:write(name,a)
pairs=[];pmap={(a['arm'],float(a['ratio']),int(a['frame_index'])):a for a in pv}
for ratio in RATIOS:
    frames=sorted({int(a['frame_index'])for a in pv if float(a['ratio'])==ratio})
    for arm,base in [('S','P0'),('L','P0'),('G','P0'),('L','C'),('P0','C')]+[('L_Q'+str(q),'P0')for q in range(4)]:
        for m in ['psnr','ssim','lpips','photo']:
            d=np.array([float(pmap[arm,ratio,f][m])-float(pmap[base,ratio,f][m])for f in frames]);pairs.append(dict(arm=arm,base=base,ratio=ratio,metric=m,views=len(d),mean=float(d.mean()),median=float(np.median(d)),p10=float(np.quantile(d,.1)),p90=float(np.quantile(d,.9)),improved_view_fraction=float(np.mean(d>0 if m in ['psnr','ssim']else d<0))))
write('paired_statistics.csv',pairs)
summary=dict(all_passed=True,N_candidate=N,baseline=json.loads((ROOT/'baseline_reproduction.json').read_text()),construction=json.loads((ROOT/'construction_checks.json').read_text()),quality=quality,survival=survival,rank_statistics=rankstats,symmetric_difference=sym,spatial=coverage,paired=pairs,render_counts=json.loads((ROOT/'quality_complete.json').read_text()))
(ROOT/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print('DIAGNOSTICS_PASS',len(inventory),len(tr),flush=True)

"""统一same-K诊断、paired质量统计与中文报告；不改参数或ranking。"""
from common import *
from experiment_data import load
from evaluate import write_csv
import csv,math
P=read(ROOT/'preregistration.json');pop=read(ROOT/'population.json');ratios=P['ratios'];arms=P['arms'];metrics=['psnr','ssim','lpips','photo']
assert read(ROOT/'quality_complete.json')['all_passed']
x,b,nodes=load();ids=x['row_ids'];N=len(ids);W=x['raw_I'];total=float(W.sum())
rows=list(csv.DictReader(open(ROOT/'quality_per_view.csv')));sums=list(csv.DictReader(open(ROOT/'quality_quality.csv')))
Q={(z['arm'],float(z['ratio'])):{m:float(z[m])for m in metrics}for z in sums};V={(z['arm'],float(z['ratio']),int(z['frame_index'])):z for z in rows}
frames=read(ROOT/'input_manifest.json')['evaluation_frames'];assert len(V)==16*len(frames)
orders={a:b[a+'_order']for a in arms};ranks={};masks={};overlap=[];red=[];coverage=[];composition=[];changes=[]
for a,o in orders.items():
    rank=np.empty(N,int);rank[o]=np.arange(1,N+1);ranks[a]=rank
    for ratio in ratios:masks[a,ratio]=rank<=int(N*ratio+.5)
events=list(csv.DictReader(open(ROOT/'ranking_emit_trace.csv')));event={int(e['row_id']):e for e in events}
leafmass={n.index:n.mass for n in nodes.values()if n.leaf};rootmass={int(b['root'][n.members[0]]):n.mass for n in nodes.values()if n.level==1}
rootcount=len(rootmass);leafcount=len(leafmass)
# 区域定义仅基于冻结几何/Raw SI，完全不读取质量。
counts=np.bincount(b['root'],minlength=rootcount);mass=np.array([rootmass[i]for i in range(rootcount)])
dense=np.lexsort((np.arange(rootcount),-counts))[:max(1,int(np.ceil(.1*rootcount)))];important=np.lexsort((np.arange(rootcount),-mass))[:max(1,int(np.ceil(.1*rootcount)))]
save(ROOT/'region_definition.json',dict(rule='top10% occupied roots by occupancy and separately top10% by Raw SI mass; ties root ordinal',dense_roots=dense.tolist(),important_roots=important.tolist(),quality_not_used=True))
for a in arms:
    for ratio in ratios:
        mask=masks[a,ratio];K=int(mask.sum());values=np.zeros(N);pairs={t:0 for t in [.8,.9,.95]}
        for c in np.unique(b['fixed_cell'][mask]):
            g=np.flatnonzero(mask&(b['fixed_cell']==c));n=len(g)
            if n<2:continue
            d=((x['xyz'][g,None,:]-x['xyz'][g][None,:,:])**2).sum(2);sig=.5*(x['radius'][g,None]+x['radius'][g][None,:]);k=np.exp(-d/(2*sig*sig+1e-12));np.fill_diagonal(k,0)
            values[g]=k.max(1);upper=k[np.triu_indices(n,1)]
            for t in pairs:pairs[t]+=int((upper>t).sum())
        red.append(dict(arm=a,ratio=ratio,K=K,mean_redundancy=float(values[mask].mean()),median_redundancy=float(np.median(values[mask])),p90_redundancy=float(np.quantile(values[mask],.9)),**{'pairs_gt_'+str(t):p for t,p in pairs.items()}))
        lh=np.unique(b['leaf'][mask]);rh=np.unique(b['root'][mask]);selected_counts=np.bincount(b['root'][mask],minlength=rootcount)
        coverage.append(dict(arm=a,ratio=ratio,K=K,selected_final_leaves=len(lh),selected_roots=len(rh),leaf_fraction=len(lh)/leafcount,root_fraction=len(rh)/rootcount,hit_leaf_Raw_mass_fraction=sum(leafmass[int(i)]for i in lh)/total,hit_root_Raw_mass_fraction=sum(rootmass[int(i)]for i in rh)/total,rows_per_root_mean=float(selected_counts.mean()),dense_roots_budget_fraction=int(mask[np.isin(b['root'],dense)].sum())/K,important_roots_budget_fraction=int(mask[np.isin(b['root'],important)].sum())/K))
        comp=dict(arm=a,ratio=ratio,K=K,C_final_representatives=int((mask&b['rep_mask']).sum()),C_representative_fraction=float(b['rep_mask'][mask].mean()))
        for level in [1,2,3]:comp['C_L'+str(level)+'_rep_count']=int((mask&(b['rep_level']==level)).sum())
        if a=='F3_ProgressiveMarginal':
            ev=[event[int(i)]for i in ids[mask]]
            for phase in ['internal_rep','final_rep','marginal_tail']:comp[phase+'_count']=sum(e['phase']==phase for e in ev)
            for level in [1,2,3]:comp['emitted_L'+str(level)+'_count']=sum(int(e['level'])==level for e in ev)
        composition.append(comp)
for ratio in ratios:
    for first,second in [('A_Raw','C_Hierarchical'),('A_Raw','F3_ProgressiveMarginal'),('B_FixedMarginal','C_Hierarchical'),('B_FixedMarginal','F3_ProgressiveMarginal'),('C_Hierarchical','F3_ProgressiveMarginal')]:
        am=masks[first,ratio];bm=masks[second,ratio];inter=int((am&bm).sum());union=int((am|bm).sum());overlap.append(dict(comparison=first+'_vs_'+second,ratio=ratio,K=int(am.sum()),intersection=inter,Jaccard=inter/union,added=int((bm&~am).sum()),removed=int((am&~bm).sum())))
        if second=='F3_ProgressiveMarginal':
            for label,mask in [('added',bm&~am),('removed',am&~bm)]:
                for i in np.flatnonzero(mask):changes.append(dict(comparison=first+'_vs_'+second,ratio=ratio,change=label,row_id=int(ids[i]),x=x['xyz'][i,0],y=x['xyz'][i,1],z=x['xyz'][i,2],raw_SI=W[i],C_score=b['C_Hierarchical_score'][i],C_rank=int(ranks['C_Hierarchical'][i]),F3_rank=int(ranks[second][i]),F3_phase=event[int(ids[i])]['phase'],F3_depth=event[int(ids[i])]['level']))
write_csv('selected_redundancy.csv',red);write_csv('spatial_budget_coverage.csv',coverage);write_csv('ranking_overlap.csv',overlap);write_csv('ranking_selected_row_changes.csv',changes)
allkeys=list(dict.fromkeys(k for z in composition for k in z));write_csv('progressive_composition.csv',[{k:z.get(k,'')for k in allkeys}for z in composition])
paired=[];diff=[];perdelta=[]
comparisons=[('C_Hierarchical','A_Raw'),('B_FixedMarginal','A_Raw'),('F3_ProgressiveMarginal','A_Raw'),('F3_ProgressiveMarginal','C_Hierarchical'),('F3_ProgressiveMarginal','B_FixedMarginal')]
for a,ref in comparisons:
    for ratio in ratios:
        delta={m:Q[a,ratio][m]-Q[ref,ratio][m]for m in metrics};diff.append(dict(comparison=a+'_minus_'+ref,ratio=ratio,**{'delta_'+m:v for m,v in delta.items()}))
        for m in metrics:
            vals=np.array([float(V[a,ratio,f][m])-float(V[ref,ratio,f][m])for f in frames]);np.testing.assert_allclose(vals.mean(),delta[m],rtol=1e-10,atol=1e-10)
            paired.append(dict(comparison=a+'_minus_'+ref,ratio=ratio,metric=m,mean=float(vals.mean()),median=float(np.median(vals)),p10=float(np.quantile(vals,.1)),p90=float(np.quantile(vals,.9)),positive_delta_fraction=float((vals>0).mean()),improved_view_fraction=float(((vals>0)if m in ['psnr','ssim']else(vals<0)).mean())))
        for f in frames:perdelta.append(dict(comparison=a+'_minus_'+ref,ratio=ratio,frame_index=f,**{'delta_'+m:float(V[a,ratio,f][m])-float(V[ref,ratio,f][m])for m in metrics}))
write_csv('quality_deltas.csv',diff);write_csv('paired_statistics.csv',paired);write_csv('paired_per_view_deltas.csv',perdelta)
R={(z['arm'],z['ratio']):z for z in red}
def dp(a,ref):return np.array([Q[a,r]['psnr']-Q[ref,r]['psnr']for r in ratios])
ca=dp('C_Hierarchical','A_Raw');fc=dp('F3_ProgressiveMarginal','C_Hierarchical');fb=dp('F3_ProgressiveMarginal','B_FixedMarginal');fa=dp('F3_ProgressiveMarginal','A_Raw')
f3red=all(R['F3_ProgressiveMarginal',ratios[i]]['mean_redundancy']<R['C_Hierarchical',ratios[i]]['mean_redundancy']for i in [2,3]);combined=bool(fc.mean()>=0 and fc[0]>0 and f3red)
bmean=float(np.mean([R['B_FixedMarginal',r]['mean_redundancy']for r in ratios]));fmean=float(np.mean([R['F3_ProgressiveMarginal',r]['mean_redundancy']for r in ratios]));bp=sum(R['B_FixedMarginal',r]['pairs_gt_0.9']for r in ratios);fp=sum(R['F3_ProgressiveMarginal',r]['pairs_gt_0.9']for r in ratios)
strong=bool(combined and fb.mean()>=-.02 and fmean<=1.1*bmean and fp<=max(1,math.ceil(1.1*bp)))
cgate=bool((ca>=-.02).sum()>=3 and(ca[:2].mean()>.10 or(ca[:2].max()>.20 and ca[:2].min()>=-.02)))
result=dict(dataset='fr2_xyz',C_delta_PSNR_vs_A=ca.tolist(),F3_delta_PSNR_vs_A=fa.tolist(),F3_delta_PSNR_vs_C=fc.tolist(),F3_delta_PSNR_vs_B=fb.tolist(),F3_mean_delta_PSNR_C=float(fc.mean()),F3_mean_delta_PSNR_B=float(fb.mean()),C_primary_PSNR_gate=cgate,F3_combined_supported=combined,F3_strong_improvement=strong,F3_high_redundancy_gate=f3red,F3_vs_B_PSNR='EXCEEDS'if fb.mean()>=.02 else'MATCHES'if fb.mean()>=-.02 else'BELOW',strong_redundancy=dict(B_mean=bmean,F3_mean=fmean,B_pairs_gt09=bp,F3_pairs_gt09=fp),single_frozen_map=True,zero_tuning=True,online_claim=False)
save(ROOT/'mechanism_result.json',result)
for path,val in read(ROOT/'frozen_sources.json').items():assert sha(path)==val['sha256'],path
for n,v in read(ROOT/'reused_source_checks.json').items():assert sha(ROOT/n)==v['current_sha256']and v['exact']
assert read(ROOT/'ranking_sanity.json')['all_passed']and read(ROOT/'independent_replay_checks.json')['all_passed']
prsha=sha(ROOT/'preregistration.json');save(ROOT/'final_checks.json',dict(all_passed=True,frozen_sources_unchanged=True,reused_math_unchanged=True,preregistration_sha256=prsha,optimizer_updates=0,same_K=True,nested=True,deterministic=True,unique_rows=True,quality_cases=16,view_renders=len(rows)))
summary=dict(dataset='fr2_xyz',population=pop,ratios=ratios,quality=[dict(arm=z['arm'],ratio=float(z['ratio']),K=int(z['K']),views=int(z['views']),**{m:float(z[m])for m in metrics})for z in sums],mechanism=result,checks=read(ROOT/'final_checks.json'))
save(ROOT/'summary.json',summary)
print('ANALYSIS_COMPLETE',json.dumps(result),flush=True)

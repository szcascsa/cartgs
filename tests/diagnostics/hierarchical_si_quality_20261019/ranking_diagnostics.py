"""排序与机制诊断；不读取quality，不改变任何ranking。"""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text())
def save(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def write(n,records):
    with open(ROOT/n,'w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def stats(a):
    a=np.asarray(a);return {'count':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'p10':float(np.quantile(a,.1)),'p90':float(np.quantile(a,.9))}if len(a)else {'count':0,'mean':None,'median':None,'p10':None,'p90':None}
def ranks(a):
    _,inverse,counts=np.unique(a,return_inverse=True,return_counts=True)
    return (np.cumsum(counts)-.5*(counts-1))[inverse]
assert read('ranking_sanity.json')['all_passed']
x=dict(np.load(ROOT/'ranking_inputs.npz'));r=dict(np.load(ROOT/'arm_rankings.npz'))
ids=r['row_ids'];N=len(ids);ratios=read('preregistration.json')['ratios'];arms=read('preregistration.json')['primary']+read('preregistration.json')['diagnostic']
W=x['raw_I'];rep=r['rep_mask'];level=r['rep_level'];leaf=r['leaf'];root=r['root'];cell=r['fixed_cell'];total=W.sum()
trees=list(csv.DictReader(open(ROOT/'structure_run1/sparse_hash_target_2048_tree.csv')))
leaves={int(n['node']):n for n in trees if n['leaf']=='1'}
orders={a:r[a+'_order']for a in arms};masks={}
for a in arms:
    for ratio in ratios:
        m=np.zeros(N,bool);m[orders[a][:int(ratio*N+.5)]]=True;masks[a,ratio]=m
overlap=[];changes=[]
for ratio in ratios:
    K=int(ratio*N+.5);C=masks['C_Hierarchical',ratio]
    for baseline in ['A_Raw','B_FixedMarginal','D_RepresentativeOnly','E_L1Only']:
        A=masks[baseline,ratio];inter=int((C&A).sum())
        overlap.append({'ratio':ratio,'K':K,'comparison':'C_vs_'+baseline,'intersection':inter,'added':int((C&~A).sum()),'removed':int((A&~C).sum()),'Jaccard':inter/int((C|A).sum())})
        if baseline in ['A_Raw','B_FixedMarginal']:
            for change,mask in [('added',C&~A),('removed',A&~C)]:
                for i in np.flatnonzero(mask):changes.append({'ratio':ratio,'comparison':'C_vs_'+baseline,'change':change,'row_id':int(ids[i]),'raw_I':float(W[i]),'hierarchical_SI':float(r['C_Hierarchical_score'][i]),'representative':bool(rep[i]),'level':int(level[i]),'leaf':int(leaf[i]),'root':int(root[i]),'x':float(x['xyz'][i,0]),'y':float(x['xyz'][i,1]),'z':float(x['xyz'][i,2])})
write('ranking_overlap.csv',overlap);write('ranking_selected_set_changes.csv',changes)
composition=[];levels=[];coverage=[];residual=[]
res_ratio=np.zeros(N);res_ratio[~rep]=r['C_Hierarchical_score'][~rep]/(W[~rep]+1e-12)
assert ((res_ratio[~rep]>=0)&(res_ratio[~rep]<=1+1e-12)).all()
residual.append({'ratio':0,'population':'all_non_representatives',**stats(res_ratio[~rep]),'cover_ge_0.9':int((res_ratio[~rep]<=.1).sum()),'cover_ge_0.95':int((res_ratio[~rep]<=.05).sum()),'cover_ge_0.99':int((res_ratio[~rep]<=.01).sum())})
for ratio in ratios:
    K=int(ratio*N+.5)
    for a in arms:
        m=masks[a,ratio];nr=int((m&rep).sum());res=m&~rep
        composition.append({'arm':a,'ratio':ratio,'K':K,'representative_count':nr,'representative_fraction':nr/K,'residual_count':int(res.sum()),'residual_fraction':float(res.sum()/K),'positive_score_count':int((r[a+'_score'][m]>0).sum())})
        for l in [1,2,3]:
            count=int((m&rep&(level==l)).sum());levels.append({'arm':a,'ratio':ratio,'level':l,'count':count,'fraction_of_selected_representatives':count/max(1,nr),'fraction_of_K':count/K})
        hit=np.unique(leaf[m]);mass=sum(float(leaves[int(i)]['raw_mass'])for i in hit)
        coverage.append({'arm':a,'ratio':ratio,'K':K,'selected_leaves':len(hit),'total_leaves':len(leaves),'selected_leaf_fraction':len(hit)/len(leaves),'selected_Raw_SI_leaf_mass':mass,'selected_Raw_SI_leaf_mass_fraction':mass/total})
    for name,m in [('selected_non_representatives',masks['C_Hierarchical',ratio]&~rep),('unselected_non_representatives',~masks['C_Hierarchical',ratio]&~rep)]:
        residual.append({'ratio':ratio,'population':name,**stats(res_ratio[m]),'cover_ge_0.9':int((res_ratio[m]<=.1).sum()),'cover_ge_0.95':int((res_ratio[m]<=.05).sum()),'cover_ge_0.99':int((res_ratio[m]<=.01).sum())})
write('hierarchical_topk_composition.csv',composition);write('representative_level_composition.csv',levels);write('leaf_budget_coverage.csv',coverage);write('residual_statistics.csv',residual)
raw_rank=np.empty(N,int);raw_rank[orders['A_Raw']]=np.arange(1,N+1);C_rank=np.empty(N,int);C_rank[orders['C_Hierarchical']]=np.arange(1,N+1)
details=[]
for i in np.flatnonzero(rep):
    n=leaves[int(leaf[i])];details.append({'row_id':int(ids[i]),'raw_I':float(W[i]),'Delta':float(r['rep_delta'][i]),'Delta_over_raw':float(r['rep_delta'][i]/(W[i]+1e-12)),'level':int(level[i]),'leaf':int(leaf[i]),'root':int(root[i]),'leaf_occupancy':int(n['occupancy']),'leaf_Q':float(n['Q']),'leaf_Raw_SI_mass':float(n['raw_mass']),'global_Raw_rank':int(raw_rank[i]),'global_Hierarchical_rank':int(C_rank[i])})
write('representative_delta_statistics.csv',details)
reps=np.flatnonzero(rep);delta_order=reps[np.lexsort((ids[reps],-r['rep_delta'][reps]))];raw_rep_order=reps[np.lexsort((ids[reps],-W[reps]))]
corr=float(np.corrcoef(ranks(W[rep]),ranks(r['rep_delta'][rep]))[0,1]);delta_overlap=[]
for ratio in ratios:
    for label,K in [('candidate_K',int(ratio*N+.5)),('representative_population_K',int(ratio*len(reps)+.5))]:
        a=set(delta_order[:K]);b=set(raw_rep_order[:K]);delta_overlap.append({'ratio':ratio,'population':label,'K':K,'Jaccard':len(a&b)/len(a|b)})
write('representative_delta_rank_jaccard.csv',delta_overlap)
save('representative_delta_summary.json',{'representatives':len(reps),'Spearman_Delta_vs_Raw':corr,'Delta':stats(r['rep_delta'][rep]),'raw_I':stats(W[rep]),'Delta_over_Raw':stats(r['rep_delta'][rep]/(W[rep]+1e-12))})
# Uniform frozen Fixed-h8 geometry metric, independent of Hierarchical leaves.
redundancy=[];maxima={}
for a in arms:
    for ratio in ratios:
        m=masks[a,ratio];out=np.zeros(N);pair_counts={t:0 for t in [.8,.9,.95]}
        for c in np.unique(cell[m]):
            g=np.flatnonzero(m&(cell==c));n=len(g)
            if n<2:continue
            d=((x['xyz'][g,None,:]-x['xyz'][g][None,:,:])**2).sum(2);sigma=.5*(x['radius'][g,None]+x['radius'][g][None,:]);k=np.exp(-d/(2*sigma*sigma+1e-12));np.fill_diagonal(k,0)
            out[g]=k.max(1);upper=k[np.triu_indices(n,1)]
            for t in pair_counts:pair_counts[t]+=int((upper>t).sum())
        maxima[a,ratio]=out
        redundancy.append({'arm':a,'ratio':ratio,'K':int(m.sum()),**stats(out[m]),**{'unordered_pairs_gt_'+str(t):v for t,v in pair_counts.items()}})
write('selected_redundancy.csv',redundancy)
regions=[];definition={}
for domain,assignment,keys in [('fixed_h8',cell,r['fixed_keys']),('Sparse_root',root,np.unique(np.floor(x['xyz']/read('ranking_sanity.json')['structure']['h_root']).astype(np.int64),axis=0))]:
    n=int(assignment.max()+1);occ=np.bincount(assignment,minlength=n);mass=np.bincount(assignment,weights=W,minlength=n)
    dense=np.lexsort((np.arange(n),-occ))[:int(np.ceil(.1*n))];mass_order=np.lexsort((np.arange(n),-mass));important=mass_order[:np.searchsorted(np.cumsum(mass[mass_order]),.8*mass.sum())+1]
    top10=np.lexsort((np.arange(n),-occ))[:10]
    definition[domain]={'cells':n,'dense_top10percent':dense.tolist(),'important_cover80percent':important.tolist(),'top10_dense':top10.tolist(),'key_tie_break':'lexicographic ascending'}
    for label,chosen in [('dense_top10percent',dense),('important_cover80percent',important),('top10_dense',top10)]:
        eligible=np.isin(assignment,chosen)
        for a in arms:
            for ratio in ratios:
                m=masks[a,ratio];local=m&eligible;count=int(local.sum())
                regions.append({'domain':domain,'region':label,'arm':a,'ratio':ratio,'K':int(m.sum()),'region_count':len(chosen),'region_candidate_count':int(eligible.sum()),'region_Raw_mass_fraction':float(W[eligible].sum()/total),'selected_count':count,'budget_fraction':float(count/m.sum()),'uniform_geometry_mean_redundancy_inside_region':float(maxima[a,ratio][local].mean())if count else 0})
write('dense_region_budget_share.csv',regions);save('region_definitions.json',definition)
save('ranking_diagnostics_complete.json',{'all_passed':True,'same_geometry_metric':True,'geometry_partition':'frozen Fixed-h8','geometry_only':True,'quality_read':False,'Spearman_Delta_vs_Raw':corr,'representatives':int(rep.sum()),'non_representatives':int((~rep).sum())})
print('RANKING_DIAGNOSTICS_PASS',corr,flush=True)

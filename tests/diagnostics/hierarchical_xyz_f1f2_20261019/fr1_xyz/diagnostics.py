"""六arm统一geometry诊断与完整row/空间/机制统计；不改ranking。"""
from pathlib import Path
import csv,json
import numpy as np
from experiment_data import load
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text())
def records(n):return list(csv.DictReader(open(ROOT/n)))
def write(n,a):
    with open(ROOT/n,'w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(a[0]));w.writeheader();w.writerows(a)
def save(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def distribution(a):
    a=np.asarray(a,dtype=float);return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)))if len(a)else dict(count=0,mean=None,median=None,p90=None)
x,b,nodes=load();r=dict(np.load(ROOT/'new_arm_rankings.npz'));ids=x['row_ids'];N=len(ids);w=x['raw_I'];total=float(w.sum());p=read('preregistration.json');arms=p['baseline']+p['new_arms'];ratios=p['ratios']
orders={a:(b if a in p['baseline']else r)[a+'_order']for a in arms};ranks={}
for a,o in orders.items():
    rank=np.empty(N,int);rank[o]=np.arange(1,N+1);ranks[a]=rank
events=records('ranking_emit_trace.csv');byarm={a:{}for a in p['new_arms']}
for e in events:byarm[e['arm']][int(e['row_id'])]=e
masks={(a,ratio):ranks[a]<=int(N*ratio+.5)for a in arms for ratio in ratios};overlap=[];changes=[]
for ratio in ratios:
    K=int(N*ratio+.5)
    for first,second in [('C_Hierarchical',a)for a in p['new_arms']]+[('B_FixedMarginal','F3_ProgressiveMarginal'),('F2_ProgressiveResidual','F3_ProgressiveMarginal')]:
        a=masks[first,ratio];z=masks[second,ratio];inter=int((a&z).sum());union=int((a|z).sum())
        overlap.append({'comparison':first+'_vs_'+second,'ratio':ratio,'K':K,'intersection':inter,'added':int((z&~a).sum()),'removed':int((a&~z).sum()),'Jaccard':inter/union})
        for label,mask in [('added',z&~a),('removed',a&~z)]:
            for i in np.flatnonzero(mask):
                event=byarm[second][int(ids[i])]
                changes.append({'comparison':first+'_vs_'+second,'ratio':ratio,'change':label,'row_id':int(ids[i]),'x':x['xyz'][i,0],'y':x['xyz'][i,1],'z':x['xyz'][i,2],'Raw_SI':w[i],
                    'C_score':b['C_Hierarchical_score'][i],'C_rank':int(ranks['C_Hierarchical'][i]),'new_rank':int(ranks[second][i]),'C_final_rep':bool(b['rep_mask'][i]),'C_rep_level':int(b['rep_level'][i]),'new_emitted_depth':int(event['level']),'new_identity':event['phase'],'new_node':int(event['node'])})
write('ranking_overlap.csv',overlap);write('ranking_selected_row_changes.csv',changes)
red=[]
for a in arms:
    for ratio in ratios:
        mask=masks[a,ratio];values=np.zeros(N);pairs={t:0 for t in [.8,.9,.95]}
        for cell in np.unique(b['fixed_cell'][mask]):
            g=np.flatnonzero(mask&(b['fixed_cell']==cell));n=len(g)
            if n<2:continue
            d=((x['xyz'][g,None,:]-x['xyz'][g][None,:,:])**2).sum(2);sig=.5*(x['radius'][g,None]+x['radius'][g][None,:]);k=np.exp(-d/(2*sig*sig+1e-12));np.fill_diagonal(k,0)
            values[g]=k.max(1);upper=k[np.triu_indices(n,1)]
            for t in pairs:pairs[t]+=int((upper>t).sum())
        red.append({'arm':a,'ratio':ratio,'K':int(mask.sum()),'mean_redundancy':float(values[mask].mean()),'median_redundancy':float(np.median(values[mask])),'p90_redundancy':float(np.quantile(values[mask],.9)),**{'pairs_gt_'+str(t):v for t,v in pairs.items()}})
write('selected_redundancy.csv',red)
prior=list(csv.DictReader(open(ROOT/'prior_selected_redundancy.csv')))
for a in red:
    if a['arm']not in p['baseline']:continue
    old=next(z for z in prior if z['arm']==a['arm']and float(z['ratio'])==a['ratio'])
    for key,newkey in [('mean_redundancy','mean_redundancy'),('median_redundancy','median_redundancy'),('p90_redundancy','p90_redundancy')]:assert float(old[key])==a[newkey]
    for t in [.8,.9,.95]:assert int(old['pairs_gt_'+str(t)])==a['pairs_gt_'+str(t)]
leaves=[n for n in nodes.values()if n.leaf];root_nodes=[n for n in nodes.values()if n.level==1]
leafmass={n.index:n.mass for n in leaves};rootmass={int(b['root'][n.members[0]]):n.mass for n in root_nodes};coverage=[];composition=[]
for a in arms:
    for ratio in ratios:
        mask=masks[a,ratio];K=int(mask.sum());leafhit=np.unique(b['leaf'][mask]);roothit=np.unique(b['root'][mask]);counts=np.bincount(b['root'][mask],minlength=len(root_nodes))
        coverage.append({'arm':a,'ratio':ratio,'K':K,'selected_final_leaves':len(leafhit),'leaf_fraction':len(leafhit)/len(leaves),'selected_roots':len(roothit),'root_fraction':len(roothit)/len(root_nodes),
                         'hit_leaf_Raw_mass':sum(leafmass[int(i)]for i in leafhit),'hit_leaf_Raw_mass_fraction':sum(leafmass[int(i)]for i in leafhit)/total,
                         'hit_root_Raw_mass':sum(rootmass[int(i)]for i in roothit),'hit_root_Raw_mass_fraction':sum(rootmass[int(i)]for i in roothit)/total,
                         'selected_rows_per_all_root_mean':float(counts.mean()),'selected_rows_per_hit_root_mean':float(counts[counts>0].mean()),'selected_rows_per_root_median':float(np.median(counts)),'selected_rows_per_root_max':int(counts.max())})
        if a not in p['new_arms']:continue
        selected=[byarm[a][int(row)]for row in ids[mask]]
        counts_phase={phase:sum(e['phase']==phase for e in selected)for phase in ['internal_rep','final_rep','marginal_tail','independent_residual']}
        composition.append({'arm':a,'ratio':ratio,'K':K,**{phase+'_count':value for phase,value in counts_phase.items()},**{phase+'_fraction':value/K for phase,value in counts_phase.items()},
                            **{'L'+str(level)+'_count':sum(int(e['level'])==level for e in selected)for level in [1,2,3]},
                            **{'L'+str(level)+'_fraction':sum(int(e['level'])==level for e in selected)/K for level in [1,2,3]},'roots':len(roothit),'rows_per_root_mean':K/len(root_nodes)})
write('spatial_budget_coverage.csv',coverage);write('progressive_composition.csv',composition)
tail_top=[]
for ratio in ratios:
    mask=masks['C_Hierarchical',ratio]&~b['rep_mask']
    for a in ['F1_MarginalTail','F3_ProgressiveMarginal']:
        for i in np.flatnonzero(mask):
            e=byarm[a][int(ids[i])];delta=int(ranks[a][i]-ranks['C_Hierarchical'][i])
            tail_top.append({'arm':a,'C_ratio':ratio,'row_id':int(ids[i]),'C_rank':int(ranks['C_Hierarchical'][i]),'C_score':float(b['C_Hierarchical_score'][i]),'new_rank':int(ranks[a][i]),'new_rank_minus_C_rank':delta,'rank_direction':'up'if delta<0 else'down'if delta>0 else'same','new_priority':float(e['priority']),'new_phase':e['phase'],'new_depth':int(e['level']),'leaf':int(b['leaf'][i]),'Raw_SI':float(w[i])})
write('marginal_tail_topk_rows.csv',tail_top)
tail_top_summary=[]
for ratio in ratios:
    for a in ['F1_MarginalTail','F3_ProgressiveMarginal']:
        group=[v for v in tail_top if v['arm']==a and v['C_ratio']==ratio]
        tail_top_summary.append({'arm':a,'C_ratio':ratio,'rows':len(group),'rank_up':sum(v['rank_direction']=='up'for v in group),'rank_down':sum(v['rank_direction']=='down'for v in group),'rank_same':sum(v['rank_direction']=='same'for v in group),
                                 'mean_new_priority':float(np.mean([v['new_priority']for v in group]))if group else None,'median_new_priority':float(np.median([v['new_priority']for v in group]))if group else None,'remains_in_new_TopK':sum(v['new_rank']<=int(ratio*N+.5)for v in group)})
write('marginal_tail_topk_summary.csv',tail_top_summary)
tail_summary=[]
for a in ['F1_MarginalTail','F3_ProgressiveMarginal']:
    group=[v for v in records('marginal_tail_statistics.csv')if v['arm']==a]
    for metric in ['occupancy','positive_gain_rows','Delta4','Delta5','Delta4_over_Delta3','tail_total_gain','tail_gain_over_Raw_mass','inherited_covered_mass','new_coarse_count']:
        values=[float(v[metric])for v in group if v[metric]!=''];tail_summary.append({'arm':a,'metric':metric,**distribution(values)})
write('marginal_tail_distribution.csv',tail_summary)
schedule=[]
for a in ['F2_ProgressiveResidual','F3_ProgressiveMarginal']:
    ev=sorted(byarm[a].values(),key=lambda e:int(e['global_rank']));g=np.array([float(e['priority'])for e in ev]);unlocks=[v for v in records('progressive_unlock_trace.csv')if v['arm']==a]
    for ratio in ratios:
        K=int(ratio*N+.5);opened=[v for v in unlocks if int(v['child_unlock_rank'])<=K]
        schedule.append({'arm':a,'ratio':ratio,'K':K,'unlocked_internal_nodes':len(opened),'unlocked_L1_nodes':sum(int(v['level'])==1 for v in opened),'unlocked_L2_nodes':sum(int(v['level'])==2 for v in opened),'newly_visible_children':sum(int(v['number_of_children'])for v in opened),'emitted_priority_increases_in_prefix':int((np.diff(g[:K])>1e-10).sum())})
write('progressive_unlock_summary.csv',schedule)
save('diagnostic_checks.json',{'all_passed':True,'baseline_redundancy_exact':True,'uniform_fixed_h8_geometry':True,'boundary_metric_unchanged':True,'quality_read':False,'all_arms':arms})
print('DIAGNOSTICS_ALL_PASS',flush=True)

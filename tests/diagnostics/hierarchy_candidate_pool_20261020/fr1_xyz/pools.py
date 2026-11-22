"""纯结构candidate generation及Raw/C全局排序；不导入Progressive或读取GT/quality。"""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent
RATIOS=[.01,.05,.10,.15]
def read(name):return json.loads((ROOT/name).read_text())
def write(name,rows):
    assert rows
    with open(ROOT/name,'w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(name,value):(ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def invert(order):
    a=np.empty(len(order),np.int64);a[order]=np.arange(1,len(order)+1);return a
def generate(records,ids,raw,score):
    N=len(ids);occ=[[]for _ in ids];levels=np.zeros((N,3),bool);positions=np.zeros((N,3),bool);final=np.zeros(N,bool)
    for n in records:
        for k in range(1,4):
            key='rep'+str(k)
            if not n[key]:continue
            row=int(n[key]);i=int(np.searchsorted(ids,row));assert i<N and ids[i]==row
            l=int(n['level']);levels[i,l-1]=True;positions[i,k-1]=True;final[i]|=n['leaf']=='1'
            occ[i].append(dict(node=int(n['node']),level=l,position=k,final_leaf=n['leaf']=='1'))
    F=np.flatnonzero(final);M=np.flatnonzero(levels.any(1));assert np.isin(F,M).all()
    orders={}
    for prefix,pool in [('F',F),('M',M)]:
        for scorer,values in [('Raw',raw),('C',score)]:orders[prefix+'-'+scorer]=pool[np.lexsort((ids[pool],-values[pool]))]
    return F,M,levels,positions,occ,orders
def main():
    x=dict(np.load(ROOT/'ranking_inputs.npz'));b=dict(np.load(ROOT/'rankings.npz'));ids=x['row_ids'];N=len(ids)
    records=list(csv.DictReader(open(ROOT/'old_tree/hierarchy_tree.csv')))
    result=generate(records,ids,x['raw_I'],b['C0_score']);again=generate(records,ids,x['raw_I'],b['C0_score'])
    F,M,levels,positions,occ,orders=result
    for a,z in zip(result[:5],again[:5]):
        if isinstance(a,list):assert a==z
        else:np.testing.assert_array_equal(a,z)
    for arm,order in orders.items():np.testing.assert_array_equal(order,again[-1][arm])
    assert np.array_equal(F,np.flatnonzero(b['old_rep_mask']))
    for arm,pool in [('F-C',F),('M-C',M)]:np.testing.assert_array_equal(orders[arm],b['C0_order'][np.isin(b['C0_order'],pool)])
    ranks={a:invert(b[key+'_order'])for a,key in [('Raw','A'),('B','B'),('C','C0')]}
    covers=np.zeros(N)
    for n in records:
        if n['leaf']!='1':continue
        g=np.flatnonzero(b['old_leaf']==int(n['node']));reps=np.array([np.searchsorted(ids,int(n['rep'+str(k)]))for k in range(1,4)if n['rep'+str(k)]],int)
        d=((x['xyz'][reps,None,:]-x['xyz'][g][None,:,:])**2).sum(2);sig=.5*(x['radius'][reps,None]+x['radius'][g][None,:]);k=np.exp(-d/(2*sig*sig+1e-12));k[reps[:,None]==g[None,:]]=1.;covers[g]=k.max(0)
    np.testing.assert_allclose(b['C0_score'][~b['old_rep_mask']],x['raw_I'][~b['old_rep_mask']]*(1-covers[~b['old_rep_mask']]),rtol=1e-9,atol=1e-8)
    metadata=[];byrow={}
    for i in M:
        n=records[int(b['old_leaf'][i])];ls=np.flatnonzero(levels[i])+1
        a=dict(source_row=int(ids[i]),x=float(x['xyz'][i,0]),y=float(x['xyz'][i,1]),z=float(x['xyz'][i,2]),Raw_SI=float(x['raw_I'][i]),is_L1_rep=bool(levels[i,0]),is_L2_rep=bool(levels[i,1]),is_L3_rep=bool(levels[i,2]),is_final_leaf_rep=bool(b['old_rep_mask'][i]),rep_positions=json.dumps([int(v+1)for v in np.flatnonzero(positions[i])]),node_rep_occurrences=len(occ[i]),node_rep_positions=json.dumps(occ[i]),shallowest_level=int(ls.min()),deepest_level=int(ls.max()),final_leaf_ID=int(b['old_leaf'][i]),final_leaf_Q=float(n['Q']),final_leaf_occupancy=int(n['occupancy']),C_score=float(b['C0_score'][i]),C_global_rank=int(ranks['C'][i]),Raw_global_rank=int(ranks['Raw'][i]),B_global_rank=int(ranks['B'][i]))
        metadata.append(a);byrow[int(i)]=a
    write('candidate_pool_rows.csv',metadata)
    occurrences=sum(map(len,occ));poolsummary=[]
    for label,pool in [('F',F),('M',M)]:
        count=sum(sum(v['final_leaf']for v in occ[i])for i in pool)if label=='F'else occurrences
        poolsummary.append(dict(N_candidate=N,pool=label,pool_size=len(pool),pool_fraction=len(pool)/N,compression_ratio=N/len(pool),total_node_rep_occurrences=count,unique_rep_rows=len(pool),duplicate_occurrence_ratio=(count-len(pool))/count))
    write('candidate_pool_summary.csv',poolsummary)
    overlap=dict(unique_L1=int(levels[:,0].sum()),unique_L2=int(levels[:,1].sum()),unique_L3=int(levels[:,2].sum()),unique_final=len(F),L1_intersect_L2=int((levels[:,0]&levels[:,1]).sum()),L1_intersect_L3=int((levels[:,0]&levels[:,2]).sum()),L2_intersect_L3=int((levels[:,1]&levels[:,2]).sum()),L1_intersect_L2_intersect_L3=int(levels.all(1).sum()))
    write('level_overlap_statistics.csv',[overlap])
    recall=[];capacity=[];missed=[];added=[];composition=[];selected={};nested={}
    for r in RATIOS:
        K=int(r*N+.5)
        for label,pool in [('F',F),('M',M)]:recall.append(dict(ratio=r,K=K,pool=label,pool_size=len(pool),**{a+'_recall':float(np.isin(b[key+'_order'][:K],pool).sum()/K)for a,key in [('C','C0'),('B','B'),('Raw','A')]}))
        for arm,key in [('A','A'),('B','B'),('C','C0'),('P0','P0')]:selected[f'{arm}_{r}']=np.sort(ids[b[key+'_order'][:K]])
        for arm,o in orders.items():
            ok=len(o)>=K;capacity.append(dict(arm=arm,ratio=r,K=K,pool_size=len(o),status='SUFFICIENT'if ok else'CAPACITY INSUFFICIENT'))
            if not ok:continue
            pick=o[:K];assert len(pick)==len(np.unique(pick))==K and np.isin(pick,np.arange(N)).all();selected[f'{arm}_{r}']=np.sort(ids[pick])
            prev=nested.get(arm,set());assert prev<=set(map(int,pick));nested[arm]=set(map(int,pick))
        for i in b['C0_order'][:K]:
            if i in byrow:continue
            assert not b['old_rep_mask'][i]
            missed.append(dict(ratio=r,K=K,source_row=int(ids[i]),x=x['xyz'][i,0],y=x['xyz'][i,1],z=x['xyz'][i,2],Raw_SI=x['raw_I'][i],C_score=b['C0_score'][i],C_rank=int(ranks['C'][i]),B_rank=int(ranks['B'][i]),final_leaf_ID=int(b['old_leaf'][i]),is_residual=True,cover=covers[i],independent_residual=x['raw_I'][i]*(1-covers[i])))
        if len(M)>=K:
            pick=orders['M-C'][:K];ctop=set(map(int,b['C0_order'][:K]));added.extend(dict(ratio=r,K=K,**byrow[int(i)])for i in pick if i not in ctop)
            composition.append(dict(ratio=r,K=K,M_C_selected=K,final_pool_rows=int(np.isin(pick,F).sum()),internal_only_rows=int((~np.isin(pick,F)).sum()),internal_only_fraction=float((~np.isin(pick,F)).mean()),**{'selected_with_L'+str(l+1)+'_identity':int(levels[pick,l].sum())for l in range(3)},**{'internal_only_with_L'+str(l+1)+'_identity':int((levels[pick,l]&~b['old_rep_mask'][pick]).sum())for l in range(3)}))
    for name,a in [('candidate_recall.csv',recall),('capacity_checks.csv',capacity),('selected_pool_composition.csv',composition)]:write(name,a)
    # 即使某场景没有miss/added，也输出有schema的空CSV。
    for name,a,header in [('missed_c_rows.csv',missed,['ratio','K','source_row','x','y','z','Raw_SI','C_score','C_rank','B_rank','final_leaf_ID','is_residual','cover','independent_residual']),('added_candidate_rows.csv',added,['ratio','K']+list(metadata[0]))]:
        with open(ROOT/name,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=header);w.writeheader();w.writerows(a)
    np.savez_compressed(ROOT/'pool_rankings.npz',P_final=F,P_multi=M,levels=levels,positions=positions,cover=covers,**{a+'_order':o for a,o in orders.items()})
    np.savez_compressed(ROOT/'selected_sets.npz',**selected)
    save('pool_checks.json',dict(all_passed=True,F_subset_M=True,unique=True,candidate_only=True,structural_tree_source=True,P0_events_not_read=True,no_frontier_or_unlock=True,Raw_or_C_scorer_only=True,source_row_tie_break=True,complete_pool_permutations=True,nested_prefixes=True,repeat_exact=True,GT_or_quality_read=False,capacity=capacity,pools=poolsummary,level_overlap=overlap))
    print('POOL_CORRECTNESS_PASS',poolsummary,flush=True)
if __name__=='__main__':main()

"""仅读取冻结ranking/tree/Raw；不读取GT、图像、质量文件。"""
from pathlib import Path
import csv,json
import numpy as np
from tree_io import nodes_from_tree
ROOT=Path(__file__).resolve().parent
RATIOS=[.01,.05,.10,.15]
def rank(order):
    a=np.empty(len(order),np.int64);a[order]=np.arange(1,len(order)+1);return a
def qbin(q):return int(np.searchsorted([.25,.50,.65],q,side='right'))
def construct(P,C,events,nodes,K,only_bin=None):
    selected=set(map(int,P[:K]));ctop=set(map(int,C[:K]));used=set();trace=[]
    erow={e['index']:e for e in events}
    cr=rank(C)
    for i in map(int,P[:K]):
        e=erow[i];n=nodes[e['node']]
        if e['phase']!='internal_rep'or i in ctop or (only_bin is not None and qbin(n.quality)!=only_bin):continue
        candidates=n.members[np.argsort(cr[n.members],kind='stable')]
        proxy=next((int(j)for j in candidates if int(j)not in selected and int(j)not in used),None)
        if proxy is not None:
            assert proxy in n.members and i in selected
            selected.remove(i);selected.add(proxy);used.add(proxy)
        trace.append(dict(original=i,replacement=proxy,node=n.index,P0_rank=e['rank'],C_rank_original=int(cr[i]),C_rank_replacement=None if proxy is None else int(cr[proxy]),success=proxy is not None,Q=n.quality,level=n.level,local_rep_index=e['local_position'],same_subtree=True,original_C_topK=i in ctop,replacement_C_topK=proxy in ctop if proxy is not None else False))
    assert len(selected)==K
    return np.array(sorted(selected),np.int64),trace
def refill(P,C,internal,K):
    ctop=set(map(int,C[:K]));s={int(i)for i in P[:K]if not(internal[i]and int(i)not in ctop)}
    for i in map(int,C):
        if len(s)==K:break
        s.add(i)
    assert len(s)==K
    return np.array(sorted(s),np.int64)
def write(name,rows):
    with open(ROOT/name,'w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    x=dict(np.load(ROOT/'ranking_inputs.npz'));b=dict(np.load(ROOT/'rankings.npz'));ids=x['row_ids'];N=len(ids)
    nodes,records=nodes_from_tree(ROOT/'old_tree/hierarchy_tree.csv',x['xyz'],ids,b['old_root'])
    events=json.loads((ROOT/'P0_rebuilt_events.json').read_text())['events'];P=b['P0_order'];C=b['C0_order'];internal=np.zeros(N,bool)
    types=np.empty(N,object);depth=np.zeros(N,int);ei={}
    for e in events:
        i=e['index'];types[i]={'internal_rep':'INTERNAL_REP','final_rep':'FINAL_LEAF_REP','independent_residual':'RESIDUAL'}[e['phase']];depth[i]=e['level'];internal[i]=e['phase']=='internal_rep';ei[i]=e
    S=np.array([i for i in P if not internal[i]],np.int64);assert len(S)>=int(.15*N+.5)
    np.testing.assert_array_equal(S,P[~internal[P]])
    ranks={a:rank(b[k+'_order'])for a,k in [('Raw','A'),('B','B'),('C','C0'),('P0','P0')]}
    def descendant_leaves(n):return 1 if n.leaf else sum(descendant_leaves(nodes[j])for j in n.children)
    inv=[]
    for i in P:
        i=int(i);e=ei[i];n=nodes[e['node']];rec=records[n.index];leaf=nodes[int(b['old_leaf'][i])]
        z=dict(global_rank=e['rank'],source_row=int(ids[i]),x=x['xyz'][i,0],y=x['xyz'][i,1],z=x['xyz'][i,2],Raw_SI=x['raw_I'][i],C_score=b['C0_score'][i],C_rank=int(ranks['C'][i]),B_rank=int(ranks['B'][i]),Raw_rank=int(ranks['Raw'][i]),emit_type=types[i],depth=e['level'],node=n.index,root=int(b['old_root'][i]),node_key=str(n.key[:3]),node_path=n.key[3],node_occupancy=len(n.members),node_raw_mass=n.mass,node_Q=n.quality,local_rep_index=e['local_position']if internal[i]else '',local_Delta=e['priority'],node_was_split=not n.leaf,child_count=len(n.children),descendant_final_leaves=descendant_leaves(n),descendant_rows=len(n.members),C_final_identity='FINAL_LEAF_REP'if b['old_rep_mask'][i]else'RESIDUAL',C_final_leaf=int(b['old_leaf'][i]),C_final_level=leaf.level,matches_structural_provisional_ID=i in n.original_reps)
        inv.append(z)
    write('internal_rep_inventory.csv',inv)
    selections={};trace=[];root_parity=[]
    for ratio in RATIOS:
        K=int(ratio*N+.5)
        for arm,key in [('A','A'),('B','B'),('C','C0'),('P0','P0')]:selections[f'{arm}_{ratio}']=np.sort(ids[b[key+'_order'][:K]])
        selections[f'S_{ratio}']=np.sort(ids[S[:K]])
        for arm,binid in [('L',None)]+[('L_Q'+str(q),q)for q in range(4)]:
            order,t=construct(P,C,events,nodes,K,binid);again,t2=construct(P,C,events,nodes,K,binid);np.testing.assert_array_equal(order,again);assert t==t2
            np.testing.assert_array_equal(np.bincount(b['old_root'][order]),np.bincount(b['old_root'][P[:K]]))
            selections[f'{arm}_{ratio}']=ids[order]
            if arm=='L':
                for z in t:
                    i=z.pop('original');j=z.pop('replacement');n=nodes[z['node']];trace.append(dict(ratio=ratio,K=K,original_internal_row=int(ids[i]),replacement_row=''if j is None else int(ids[j]),node_key=str(n.key[:3]),node_path=n.key[3],replacement_Raw_SI=''if j is None else float(x['raw_I'][j]),replacement_final_leaf=''if j is None else int(b['old_leaf'][j]),**z))
                root_parity.append(dict(ratio=ratio,root_budget_distribution_exact=True,replacement_count=sum(z['success']for z in t),replacement_failed=sum(not z['success']for z in t)))
        selections[f'G_{ratio}']=ids[refill(P,C,internal,K)]
        # 独立的set-comprehension重放G。
        retained={int(i)for i in P[:K]if not(internal[i]and i not in set(C[:K]))};missing=[int(i)for i in C if i not in retained]
        np.testing.assert_array_equal(selections[f'G_{ratio}'],ids[np.array(sorted(retained|set(missing[:K-len(retained)])))])
        for key,v in selections.items():
            if key.endswith('_'+str(ratio)):assert len(v)==K and len(np.unique(v))==K and np.isin(v,ids).all()
    write('local_replacement_trace.csv',trace)
    np.savez_compressed(ROOT/'diagnostic_selected_sets.npz',**selections)
    np.savez_compressed(ROOT/'diagnostic_metadata.npz',S_order=S,internal=internal,emit_type=types.astype(str),depth=depth,**{k+'_rank':v for k,v in ranks.items()})
    (ROOT/'construction_checks.json').write_text(json.dumps(dict(all_passed=True,N_candidate=N,internal_total=int(internal.sum()),S_remaining=len(S),S_stable_filter_exact=True,S_nested=True,L_G_ratio_specific=True,all_same_K_unique_candidates=True,deterministic_repeat_exact=True,L_same_subtree=True,L_root_budget_parity=root_parity,G_exact_C_refill=True,no_GT_or_quality_read=True),indent=2)+'\n')
    print('CONSTRUCTION_PASS',N,len(S),root_parity,flush=True)
if __name__=='__main__':main()

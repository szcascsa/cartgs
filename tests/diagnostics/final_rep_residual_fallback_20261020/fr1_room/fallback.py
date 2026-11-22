"""只读冻结Raw/C/tree/精确cover；不读取GT、图像或质量。"""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent;RATIOS=[.01,.05,.1,.15]
def read(n):return json.loads((ROOT/n).read_text())
def save(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write(n,rows,header=None):
    with open(ROOT/n,'w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=header or list(rows[0]));w.writeheader();w.writerows(rows)
def rank(order):
    z=np.empty(len(order),np.int64);z[order]=np.arange(1,len(order)+1);return z
def distribution(a):
    a=np.asarray(a,float)
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)))if len(a)else dict(count=0,mean=None,median=None,p10=None,p90=None)
def qbin(q):return int(np.searchsorted([.25,.50,.65,.80],q,side='right'))
def generate(ids,mask,C,score,raw,gap):
    F=np.flatnonzero(mask);U=np.flatnonzero(~mask);res=raw*gap;D={}
    for r in RATIOS:D[r]=C[:int(r*len(ids)+.5)][~mask[C[:int(r*len(ids)+.5)]]]
    m=len(D[.15]);detail={name:U[np.lexsort((ids[U],-v[U]))][:m]for name,v in [('RES',res),('RAW',raw),('GAP',gap)]}
    orders={};pools={}
    for arm,key in [('RF-C','RES'),('RF-Raw','RAW'),('RF-Gap','GAP')]:
        p=np.union1d(F,detail[key]);pools[arm]=p;orders[arm]=p[np.lexsort((ids[p],-score[p]))]
    return F,U,D,detail,pools,orders
def main():
    x=dict(np.load(ROOT/'ranking_inputs.npz'));b=dict(np.load(ROOT/'rankings.npz'));e=dict(np.load(ROOT/'exact_residual.npz'));ids=x['row_ids'];N=len(ids);C=b['C0_order'];mask=b['old_rep_mask'];raw=x['raw_I'];score=b['C0_score']
    assert read('coverage_checks.json')['all_passed'];np.testing.assert_array_equal(e['residual'][~mask],score[~mask]);np.testing.assert_array_equal(raw*e['gap'],e['residual'])
    tree=list(csv.DictReader(open(ROOT/'old_tree/hierarchy_tree.csv')));structural=np.unique([int(n['rep'+str(k)])for n in tree if n['leaf']=='1'for k in range(1,4)if n['rep'+str(k)]])
    np.testing.assert_array_equal(ids[mask],structural)
    z=generate(ids,mask,C,score,raw,e['gap']);F,U,D,detail,pools,orders=z;repeat=generate(ids,mask,C,score,raw,e['gap'])
    for a,v in orders.items():np.testing.assert_array_equal(v,repeat[-1][a]);assert len(v)==len(np.unique(v))and set(v)==set(pools[a])
    np.testing.assert_array_equal(detail['RES'],D[.15]);prefix=U[np.lexsort((ids[U],-e['residual'][U]))];previous=set();requirements=[]
    ranks=dict(C=rank(C),B=rank(b['B_order']),Raw=rank(b['A_order']))
    ur=np.zeros(N,int);ur[U[np.lexsort((ids[U],-raw[U]))]]=np.arange(1,len(U)+1)
    selections={};recall=[];proxy=[];nested={};sizes=[]
    for r in RATIOS:
        K=int(r*N+.5);d=D[r];m=len(d);np.testing.assert_array_equal(d,prefix[:m]);assert previous<=set(d);previous=set(d)
        requirements.append(dict(ratio=r,K=K,representative_rows=K-m,residual_rows=m,residual_fraction_of_K=m/K,residual_fraction_of_N=m/N,hit_residual_leaves=len(np.unique(b['old_leaf'][d]))))
        for arm,key in [('A','A'),('B','B'),('C','C0')]:selections[f'{arm}_{r}']=np.sort(ids[b[key+'_order'][:K]])
        Fo=F[np.lexsort((ids[F],-score[F]))];selections[f'F_{r}']=np.sort(ids[Fo[:K]])
        for arm,p in dict(F=F,**pools).items():recall.append(dict(arm=arm,ratio=r,K=K,candidate_pool_size=len(p),C_total_recall=float(np.isin(C[:K],p).mean()),C_representative_recall=1.,C_residual_recall=float(np.isin(d,p).mean())if m else None))
        for label,selected_detail in detail.items():
            inter=int(np.isin(d,selected_detail).sum());proxy.append(dict(proxy=label,ratio=r,K=K,reference_residual_N=m,proxy_residual_N=len(selected_detail),recalled_residual_N=inter,residual_recall=inter/m if m else None,intersection_D15=int(np.isin(D[.15],selected_detail).sum())))
        for arm,o in orders.items():
            pick=o[:K];assert len(pick)==len(np.unique(pick))==K and np.isin(pick,np.arange(N)).all();assert nested.get(arm,set())<=set(pick);nested[arm]=set(pick);selections[f'{arm}_{r}']=np.sort(ids[pick])
        np.testing.assert_array_equal(orders['RF-C'][:K],C[:K]);np.testing.assert_array_equal(selections[f'RF-C_{r}'],selections[f'C_{r}'])
    for arm,p in dict(F=F,**pools).items():sizes.append(dict(arm=arm,full_N=N,Final_reps=len(F),Added_residual=len(p)-len(F),total_candidate=len(p),F_fraction=len(F)/N,residual_fallback_fraction=(len(p)-len(F))/N,retained_fraction=len(p)/N,compression_ratio=N/len(p)))
    assert len({len(p)for p in pools.values()})==1
    def rowdata(i):
        leaf=tree[int(b['old_leaf'][i])]
        return dict(source_row=int(ids[i]),x=float(x['xyz'][i,0]),y=float(x['xyz'][i,1]),z=float(x['xyz'][i,2]),Raw_SI=float(raw[i]),cover=float(e['cover'][i]),gap=float(e['gap'][i]),residual_score=float(e['residual'][i]),C_rank=int(ranks['C'][i]),B_rank=int(ranks['B'][i]),Raw_rank=int(ranks['Raw'][i]),Raw_rank_percentile=float(ranks['Raw'][i]/N),U_Raw_rank_percentile=float(ur[i]/len(U))if not mask[i]else 0.,final_leaf_ID=int(b['old_leaf'][i]),leaf_Q=float(leaf['Q']),leaf_occupancy=int(leaf['occupancy']))
    detail_sets={k:set(v)for k,v in detail.items()};Dsets={r:set(v)for r,v in D.items()}
    allrows=[dict(**rowdata(int(i)),in_D15_RES=bool(i in detail_sets['RES']),in_D15_RAW=bool(i in detail_sets['RAW']),in_D15_GAP=bool(i in detail_sets['GAP']),**{'selected_C_'+str(r):bool(i in Dsets[r])for r in RATIOS})for i in U]
    write('residual_pool_rows.csv',allrows);fn=[];fp=[]
    for name in ['RAW','GAP']:
        fn.extend(dict(proxy=name,**rowdata(int(i)))for i in np.setdiff1d(D[.15],detail[name]));fp.extend(dict(proxy=name,**rowdata(int(i)))for i in np.setdiff1d(detail[name],D[.15]))
    header=['proxy']+list(rowdata(int(U[0])))
    write('proxy_false_negative_rows.csv',fn,header);write('proxy_false_positive_rows.csv',fp,header)
    leafstats=[];qstats=[];coverage=[]
    for n in tree:
        if n['leaf']!='1':continue
        node=int(n['node']);g=U[b['old_leaf'][U]==node];a=dict(leaf=node,level=int(n['level']),occupancy=int(n['occupancy']),Q=float(n['Q']),representative_count=sum(bool(n['rep'+str(k)])for k in range(1,4)),Raw_SI_mass=float(n['raw_mass']),residual_candidate_N=len(g),residual_score_mass=float(e['residual'][g].sum()))
        for r in RATIOS:
            s=D[r][b['old_leaf'][D[r]]==node];a.update({f'selected_residual_N_{r}':len(s),f'selected_residual_Raw_mass_{r}':float(raw[s].sum()),f'selected_residual_score_mass_{r}':float(e['residual'][s].sum())})
        leafstats.append(a)
    for r in RATIOS:
        selected=np.zeros(N,bool);selected[D[r]]=True
        for q in range(5):
            g=U[np.array([qbin(float(tree[int(b['old_leaf'][i])]['Q']))==q for i in U])];s=g[selected[g]]
            qstats.append(dict(ratio=r,Q_bin=['<.25','.25-.50','.50-.65','.65-.80','>=.80'][q],residual_candidate_N=len(g),selected_residual_N=len(s),selected_fraction_of_bin=len(s)/len(g)if len(g)else 0.,selected_fraction_of_all_D=len(s)/len(D[r])if len(D[r])else 0.,candidate_residual_score_mass=float(e['residual'][g].sum()),selected_residual_score_mass=float(e['residual'][s].sum())))
        for label,g in [('selected',D[r]),('nonselected',U[~selected[U]])]:
            for metric,v in [('cover',e['cover']),('gap',e['gap']),('Raw_SI',raw),('residual_score',e['residual']),('Raw_global_rank_percentile',ranks['Raw']/N),('Raw_U_rank_percentile',ur/len(U))]:coverage.append(dict(ratio=r,population=label,metric=metric,**distribution(v[g])))
    for name,rows in [('residual_requirement_by_ratio.csv',requirements),('residual_proxy_comparison.csv',proxy),('candidate_pool_summary.csv',sizes),('candidate_recall.csv',recall),('leaf_residual_statistics.csv',leafstats),('q_residual_statistics.csv',qstats),('coverage_residual_statistics.csv',coverage)]:write(name,rows)
    np.savez_compressed(ROOT/'fallback_rankings.npz',F=F,U=U,**{'D_'+str(r):d for r,d in D.items()},**{'D15_'+k:v for k,v in detail.items()},**{a+'_pool':p for a,p in pools.items()},**{a+'_order':o for a,o in orders.items()})
    np.savez_compressed(ROOT/'selected_sets.npz',**selections)
    save('fallback_checks.json',dict(all_passed=True,N_candidate=N,F_exact_final_mask=True,U_complement=True,residual_row_exact=True,D_nested_and_prefix_exact=True,D15_RES_exact=True,RF_C_all_selected_order_and_sets_exact=True,proxy_pool_sizes_equal=True,unique=True,nested=True,repeated_exact=True,construct_no_GT_quality=True,m15=len(D[.15]),requirements=requirements,pool_summary=sizes))
    print('FALLBACK_CORRECTNESS_PASS',sizes,requirements,flush=True)
if __name__=='__main__':main()

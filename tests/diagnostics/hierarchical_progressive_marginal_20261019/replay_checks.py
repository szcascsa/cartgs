"""独立从emit trace重算每节点greedy/coverage并重放全局head unlock，不调用ranker。"""
from pathlib import Path
import json,csv,heapq
import numpy as np
from experiment_data import load
ROOT=Path(__file__).resolve().parent
x,b,nodes=load();ids=x['row_ids'];N=len(ids);W=x['raw_I'];X=x['xyz'];R=x['radius']
traces=list(csv.DictReader(open(ROOT/'ranking_emit_trace.csv')));new=np.load(ROOT/'new_arm_rankings.npz');maxerr=0.;checked=0
for arm in ['F1_MarginalTail','F2_ProgressiveResidual','F3_ProgressiveMarginal']:
    ev=[e for e in traces if e['arm']==arm];per={i:[]for i in nodes};coarse={i:[]for i in nodes};parent={c:i for i,n in nodes.items()for c in n.children}
    for e in ev:
        n=int(e['node']);per[n].append(e)
        if e['phase']in ['internal_rep','final_rep']:coarse[n].append(int(np.searchsorted(ids,int(e['row_id']))))
    for node,n in nodes.items():
        group=per[node]
        if not group:continue
        g=n.members;anc=[];ancestor=parent.get(node)
        if arm!='F1_MarginalTail':
            while ancestor is not None:anc+=coarse[ancestor];ancestor=parent.get(ancestor)
        cover=np.zeros(len(g));taken=np.isin(g,anc)
        if anc:
            distance=((X[np.array(anc),None,:]-X[g][None,:,:])**2).sum(2);sigma=.5*(R[np.array(anc),None]+R[g][None,:]);k=np.exp(-distance/(2*sigma*sigma+1e-12));k[np.array(anc)[:,None]==g[None,:]]=1.;cover=k.max(0)
        # Geometry kernel independently built from xyz/radius, including explicit diagonal.
        d=((X[g,None,:]-X[g][None,:,:])**2).sum(2);sig=.5*(R[g,None]+R[g][None,:]);kernel=np.exp(-d/(2*sig*sig+1e-12));np.fill_diagonal(kernel,1.)
        last=float('inf');residual=None;sequence=[]
        for e in group:
            row=int(e['row_id']);local=int(np.searchsorted(ids[g],row));assert int(ids[g[local]])==row and not taken[local]
            if e['phase']=='independent_residual':
                if residual is None:residual=W[g]*(1-cover)
                gains=residual.copy()
            else:gains=(np.maximum(kernel-cover[None,:],0)*W[g][None,:]).sum(1)
            gains[taken]=-np.inf;j=int(np.lexsort((ids[g],-gains))[0]);gain=float(gains[j]);assert j==local,('reference greedy ID mismatch',arm,node,row,int(ids[g[j]]))
            error=abs(gain-float(e['priority']));maxerr=max(maxerr,error);assert error<=1e-8+5e-12*max(1,gain)
            if e['phase']!='independent_residual':
                assert gain<=last+1e-10*max(1,n.mass);last=gain;cover=np.maximum(cover,kernel[j])
            taken[j]=True;sequence.append(int(g[j]));checked+=1
        if arm=='F1_MarginalTail':assert sequence[:3]==n.original_reps
    # Independent frontier simulation from recorded local next items and immutable topology.
    heads=[];positions={};activated=set();emitted=[]
    def activate(node):
        assert node not in activated;activated.add(node);positions[node]=0
        if per[node]:
            item=per[node][0];heapq.heappush(heads,(-float(item['priority']),nodes[node].key,int(item['row_id']),node))
        elif arm!='F1_MarginalTail':
            for child in nodes[node].children:activate(child)
    for n in nodes.values():
        if (n.leaf if arm=='F1_MarginalTail'else n.level==1):activate(n.index)
    while heads:
        priority,key,row,node=heapq.heappop(heads);emitted.append(int(np.searchsorted(ids,row)));positions[node]+=1
        t=positions[node]
        if t<len(per[node]):
            item=per[node][t];heapq.heappush(heads,(-float(item['priority']),key,int(item['row_id']),node))
        elif arm!='F1_MarginalTail':
            for child in nodes[node].children:activate(child)
    np.testing.assert_array_equal(emitted,new[arm+'_order']);assert len(set(emitted))==N
    print('REPLAY_PASS',arm,flush=True)
result={'all_passed':True,'rows_gains_argmax_checked':checked,'max_gain_error':maxerr,'all_frontiers_replayed_exact':True,'ancestor_coverage_checked':True,'global_unique_rows':True,'F1_first3_exact':True,'algorithm_parameters_changed':False}
(ROOT/'independent_replay_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)

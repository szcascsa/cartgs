"""不调用ranker，独立复算P0/P32 greedy/ancestor cover/固定residual并重放frontier。"""
from common import *
from tree_io import nodes_from_tree
import csv,heapq
x=dict(np.load(ROOT/'ranking_inputs.npz'));b=dict(np.load(ROOT/'rankings.npz'));ids=x['row_ids'];N=len(ids);X=x['xyz'];R=x['radius'];W=x['raw_I'];checked=0;maxerr=0.
for arm,which,prefix in [('P0','old','old'),('P32','new','new')]:
    nodes,_=nodes_from_tree(ROOT/(which+'_tree/hierarchy_tree.csv'),X,ids,b[prefix+'_root']);events=list(csv.DictReader(open(ROOT/(arm+'_emit_trace.csv'))));per={i:[]for i in nodes};coarse={i:[]for i in nodes};parent={c:i for i,n in nodes.items()for c in n.children}
    for e in events:
        n=int(e['node']);per[n].append(e)
        if e['phase']in ['internal_rep','final_rep']:coarse[n].append(int(np.searchsorted(ids,int(e['row_id']))))
    for i,n in nodes.items():
        if not per[i]:continue
        g=n.members;anc=[];v=parent.get(i)
        while v is not None:anc+=coarse[v];v=parent.get(v)
        cover=np.zeros(len(g));taken=np.isin(g,anc)
        if anc:
            a=np.array(anc);dist=((X[a,None,:]-X[g][None,:,:])**2).sum(2);sigma=.5*(R[a,None]+R[g][None,:]);k=np.exp(-dist/(2*sigma*sigma+1e-12));k[a[:,None]==g[None,:]]=1;cover=k.max(0)
        dist=((X[g,None,:]-X[g][None,:,:])**2).sum(2);sigma=.5*(R[g,None]+R[g][None,:]);kernel=np.exp(-dist/(2*sigma*sigma+1e-12));np.fill_diagonal(kernel,1);residual=None;last=float('inf')
        for e in per[i]:
            row=int(e['row_id']);j=int(np.searchsorted(ids[g],row));assert int(ids[g[j]])==row and not taken[j]
            if e['phase']=='independent_residual':
                if residual is None:residual=W[g]*(1-cover)
                gains=residual.copy()
            else:gains=(np.maximum(kernel-cover[None,:],0)*W[g][None,:]).sum(1)
            gains[taken]=-np.inf;best=int(np.lexsort((ids[g],-gains))[0]);assert best==j
            value=float(gains[j]);maxerr=max(maxerr,abs(value-float(e['priority'])));np.testing.assert_allclose(value,float(e['priority']),rtol=5e-12,atol=1e-8)
            if e['phase']!='independent_residual':assert value<=last+1e-10*max(1,n.mass);last=value;cover=np.maximum(cover,kernel[j])
            taken[j]=True;checked+=1
    heads=[];positions={};activated=set();order=[]
    def activate(i):
        assert i not in activated;activated.add(i);positions[i]=0
        if per[i]:
            e=per[i][0];heapq.heappush(heads,(-float(e['priority']),nodes[i].key,int(e['row_id']),i))
        else:
            for c in nodes[i].children:activate(c)
    for n in nodes.values():
        if n.level==1:activate(n.index)
    while heads:
        priority,key,row,i=heapq.heappop(heads);order.append(int(np.searchsorted(ids,row)));positions[i]+=1;t=positions[i]
        if t<len(per[i]):e=per[i][t];heapq.heappush(heads,(-float(e['priority']),key,int(e['row_id']),i))
        else:
            for c in nodes[i].children:activate(c)
    np.testing.assert_array_equal(order,b[arm+'_order']);assert len(set(order))==N
save(ROOT/'independent_replay_checks.json',dict(all_passed=True,rows_checked=checked,max_gain_error=maxerr,P32_argmax_ancestor_unlock_exact=True,no_F1_F3_executed=True))
print('INDEPENDENT_REPLAY_PASS',checked,maxerr,flush=True)

"""按原midpoint规则重建node membership；ranking数学仍使用原progressive_ranker。"""
from pathlib import Path
import csv
import numpy as np
from progressive_ranker import Node
def nodes_from_tree(path,xyz,ids,root):
    records=list(csv.DictReader(open(path)));nodes={};by_path={}
    for a in records:
        i=int(a['node']);r=int(a['root']);p=a['path'];key=tuple(int(a['key_'+b])for b in 'xyz')+(p,)
        if not p:g=np.flatnonzero(root==r)
        else:
            parent=by_path[r,p[:-1]];g=nodes[parent].members
            mid=np.array([float(records[parent][b+'min'])+.5*(float(records[parent][b+'max'])-float(records[parent][b+'min']))for b in 'xyz'])
            bit=(xyz[g]>=mid).astype(np.int64);child=bit[:,0]+2*bit[:,1]+4*bit[:,2];g=g[child==int(p[-1])]
        assert len(g)==int(a['occupancy'])and np.all(np.diff(ids[g])>0)
        reps=[int(np.searchsorted(ids,int(a['rep'+str(k)])))for k in range(1,4)if a['rep'+str(k)]]
        gains=[float(a['delta'+str(k)])for k in range(1,4)if a['rep'+str(k)]]
        nodes[i]=Node(i,key,int(a['level']),g,[],reps,gains,float(a['raw_mass']),float(a['Q']))
        if p:nodes[by_path[r,p[:-1]]].children.append(i)
        by_path[r,p]=i
    for a in records:
        n=nodes[int(a['node'])];assert n.leaf==(a['leaf']=='1')
        if not n.leaf:assert sum(len(nodes[c].members)for c in n.children)==len(n.members)
    return nodes,records

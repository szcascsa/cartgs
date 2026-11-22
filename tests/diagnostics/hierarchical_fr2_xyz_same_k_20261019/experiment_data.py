from pathlib import Path
import csv
import numpy as np
from progressive_ranker import Node
ROOT=Path(__file__).resolve().parent
def load():
    data=dict(np.load(ROOT/'ranking_inputs.npz'));base=dict(np.load(ROOT/'arm_rankings.npz'))
    ids=data['row_ids'];xyz=data['xyz'];radius=data['radius'];raw=data['raw_I']
    records=list(csv.DictReader(open(ROOT/'structure_run1/sparse_hash_target_2048_tree.csv')))
    nodes={};root_members={};by_path={}
    for a in records:
        index=int(a['node']);root=int(a['root']);path=a['path'];key=tuple(int(a['key_'+b])for b in 'xyz')+(path,)
        if not path:
            g=np.flatnonzero(base['root']==root);root_members[root]=g
        else:
            parent=by_path[root,path[:-1]];g=nodes[parent].members
            mid=np.array([float(records[parent][b+'min'])+.5*(float(records[parent][b+'max'])-float(records[parent][b+'min']))for b in 'xyz'])
            bit=(xyz[g]>=mid).astype(np.int64);child=bit[:,0]+2*bit[:,1]+4*bit[:,2];g=g[child==int(path[-1])]
        assert len(g)==int(a['occupancy'])and np.all(np.diff(ids[g])>0)
        reps=[int(np.searchsorted(ids,int(a['rep'+str(k)])))for k in range(1,4)if a['rep'+str(k)]]
        gains=[float(a['delta'+str(k)])for k in range(1,4)if a['rep'+str(k)]]
        nodes[index]=Node(index,key,int(a['level']),g,[],reps,gains,float(a['raw_mass']),float(a['Q']))
        if path:nodes[by_path[root,path[:-1]]].children.append(index)
        by_path[root,path]=index
    for a in records:
        n=nodes[int(a['node'])];assert n.leaf==(a['leaf']=='1')
        if not n.leaf:assert sum(len(nodes[c].members)for c in n.children)==len(n.members)
    return data,base,nodes

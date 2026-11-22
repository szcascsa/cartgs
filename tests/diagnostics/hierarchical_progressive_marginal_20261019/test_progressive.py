import json
import numpy as np
from progressive_ranker import Node,Cell,full_leaf_marginal,progressive,validate_ranking

def node(i,level,members,children=(),root=0,path=''):
    return Node(i,(root,0,0,path),level,np.array(members,dtype=np.int64),list(children))
def fixture(x,w=None,radius=.01):
    x=np.array(x,dtype=float);return x,np.full(len(x),radius),np.ones(len(x))if w is None else np.array(w,dtype=float),np.arange(len(x),dtype=np.int64)
tests={}
# 1/6: no-split leaf, first3 explain >.8; two identical residuals compete in the tail.
x,r,w,ids=fixture([[v,0,0]for v in [-3,-1,1,3,3]],[10,9,8,3,3]);n=node(0,1,range(5))
n.original_reps=[0,1,2];n.original_gains=[10,9,8];n.mass=w.sum();n.quality=27/33
f1=full_leaf_marginal({0:n},x,r,w,ids)
assert f1['leaf_info'][0]['sequence']==[0,1,2,3,4]
assert f1['leaf_info'][0]['gains']==[10,9,8,6,0]
tests['1_first3_and_full_tail']=True;tests['6_duplicate_residual_marginal_drop']=True
# 2: two roots; internal root children can only become visible after three new coarse emits.
x,r,w,ids=fixture([[v,0,0]for v in [0,1,2,3,20,20]],[10,9,8,7,3,2])
nodes={0:node(0,1,[0,1,2,3],[1,2]),1:node(1,2,[0,1],path='0'),2:node(2,2,[2,3],path='1'),3:node(3,1,[4,5],root=1)}
for tail in [False,True]:
    p=progressive(nodes,x,r,w,ids,tail)
    internal=[e for e in p['events']if e['node']==0];desc=[e for e in p['events']if e['node']in [1,2]]
    assert len(internal)==3 and all(e['rank']>internal[-1]['rank']for e in desc)
    assert len(p['order'])==len(set(p['order']))==6
tests['2_unlock_after_capacity']=True;tests['3_ancestor_row_exclusion']=True
# 4/5: conditioned gains respond to similarity, not depth alone.
x,r,w,ids=fixture([[0,0,0],[0,0,0],[100,0,0]],[10,8,7]);selected=np.array([True,False,False]);child=node(1,2,[1,2],path='0')
conditioned=Cell(child,x,r,w,ids,selected,[0]);unconditioned=Cell(child,x,r,w,ids,selected,[])
assert conditioned.cover[0]==1 and conditioned.cover[1]==0
assert conditioned.next_marginal()==(1,7.)
assert unconditioned.next_marginal()==(0,8.)
tests['4_high_similarity_inherited_gain_suppression']=True;tests['5_independent_ancestor_no_suppression']=True
# Repeated ancestor paths, each child contains its ancestors' rows; no duplicate emission.
x,r,w,ids=fixture([[i*.001,0,0]for i in range(12)],np.arange(12,0,-1),radius=1e-8)
nodes={0:node(0,1,range(12),[1]),1:node(1,2,range(12),[2],path='0'),2:node(2,3,range(12),path='00')}
for tail in [False,True]:
    p=progressive(nodes,x,r,w,ids,tail);repeat=progressive(nodes,x,r,w,ids,tail)
    np.testing.assert_array_equal(p['order'],repeat['order']);assert p['events']==repeat['events'] and p['unlocks']==repeat['unlocks']
    assert len(p['order'])==len(set(p['order']))==12
    validate_ranking(p['order'],12)
    assert [e['node']for e in p['events'][:3]]==[0]*3
    assert [e['node']for e in p['events'][3:6]]==[1]*3
    assert [e['node']for e in p['events'][6:]]==[2]*6
tests['7_each_row_once_repeated_path']=True;tests['8_repeat_exact']=True;tests['9_nested_prefixes']=True
tests['10_tree_structure_no_mutation']=True
assert [n.children for n in nodes.values()]==[[1],[2],[]]
# Exhausted child must not emit, including the case where all rows were ancestor reps.
x,r,w,ids=fixture([[0,0,0],[1,0,0]])
small={0:node(0,1,[0,1],[1]),1:node(1,2,[0,1],path='0')}
for tail in [False,True]:
    out=progressive(small,x,r,w,ids,tail);assert out['order'].tolist()==[0,1]
tests['empty_remaining_child']=True
print(json.dumps({'all_passed':all(tests.values()),'tests':tests},indent=2))

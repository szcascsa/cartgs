"""离线FP64精确ranking reference；不构建/修改Sparse tree或production score。"""
from dataclasses import dataclass,field
import heapq
import numpy as np

EPS=1e-12
@dataclass
class Node:
    index:int
    key:tuple
    level:int
    members:np.ndarray
    children:list=field(default_factory=list)
    original_reps:list=field(default_factory=list)
    original_gains:list=field(default_factory=list)
    mass:float=0.
    quality:float=0.
    @property
    def leaf(self):return not self.children

def geometry(x,r,first,second):
    distance=((x[first,None,:]-x[second][None,:,:])**2).sum(2)
    sigma=.5*(r[first,None]+r[second][None,:])
    k=np.exp(-distance/(2*sigma*sigma+EPS)).clip(0,1)
    k[first[:,None]==second[None,:]]=1.
    return k

class Cell:
    def __init__(self,node,x,r,w,ids,selected=None,ancestors=()):
        self.node=node;self.g=node.members;self.ids=ids;self.w=w[self.g]
        self.k=geometry(x,r,self.g,self.g)
        self.cover=np.zeros(len(self.g)) if not len(ancestors) else geometry(x,r,np.array(ancestors,dtype=np.int64),self.g).max(0)
        self.initial_cover=self.cover.copy()
        self.taken=np.zeros(len(self.g),bool)if selected is None else selected[self.g].copy()
        self.sequence=[];self.gains=[];self.ranks=[];self.residual_order=None;self.residual_pos=0
    def next_marginal(self):
        if self.taken.all():return None
        gain=(np.maximum(self.k-self.cover[None,:],0)*self.w[None,:]).sum(1)
        gain[self.taken]=-np.inf
        best=int(np.lexsort((self.ids[self.g],-gain))[0]);value=float(gain[best])
        assert np.isfinite(value)and value>=-EPS
        if self.gains:assert value<=self.gains[-1]+1e-10*max(1,float(self.w.sum())), 'Diminishing returns violation'
        return best,value
    def emit_marginal(self,j,gain,rank):
        assert not self.taken[j]
        self.taken[j]=True;self.cover=np.maximum(self.cover,self.k[j])
        self.sequence.append(int(self.g[j]));self.gains.append(float(gain));self.ranks.append(rank)
    def start_residual(self):
        self.residual=(self.w*(1-self.cover)).clip(0,None)
        remaining=np.flatnonzero(~self.taken)
        self.residual_order=remaining[np.lexsort((self.ids[self.g[remaining]],-self.residual[remaining]))]
    def next_residual(self):
        if self.residual_pos==len(self.residual_order):return None
        j=int(self.residual_order[self.residual_pos]);return j,float(self.residual[j])

def validate_ranking(order,N):
    assert len(order)==N and np.array_equal(np.sort(order),np.arange(N)), 'Ranking must emit each candidate once'
    previous=set()
    for ratio in [.01,.05,.1,.15]:
        current=set(order[:int(ratio*N+.5)]);assert previous<=current;previous=current

def full_leaf_marginal(nodes,x,r,w,ids,verify_original=True):
    sequences={};leaf_info={};heads=[];events=[];gain_by_row=np.zeros(len(w))
    for n in sorted(nodes.values(),key=lambda n:n.key):
        if not n.leaf:continue
        s=Cell(n,x,r,w,ids)
        while (item:=s.next_marginal())is not None:
            j,gain=item;s.emit_marginal(j,gain,0)
        prefix=min(3,len(n.members))
        if verify_original:
            assert s.sequence[:prefix]==n.original_reps, ('F1 first3 IDs changed',n.key,s.sequence[:prefix],n.original_reps)
            np.testing.assert_allclose(s.gains[:prefix],n.original_gains,rtol=5e-12,atol=1e-8)
        assert abs(sum(s.gains)-w[n.members].sum())<=1e-8+1e-9*w[n.members].sum()
        sequences[n.index]=(s.sequence,s.gains);leaf_info[n.index]={'sequence':s.sequence,'gains':s.gains,'initial_ancestor_cover_mass':0.,'coarse_count':prefix,'inherited_rows':0}
        if s.sequence:heapq.heappush(heads,(-s.gains[0],n.key,int(ids[s.sequence[0]]),n.index,0))
    order=[]
    while heads:
        minus,key,row,node,t=heapq.heappop(heads);seq,gains=sequences[node];i=seq[t]
        assert int(ids[i])==row;order.append(i);gain_by_row[i]=-minus
        events.append({'index':i,'node':node,'level':nodes[node].level,'phase':'final_rep'if t<3 else'marginal_tail','priority':-minus,'rank':len(order),'local_position':t+1})
        if t+1<len(seq):heapq.heappush(heads,(-gains[t+1],key,int(ids[seq[t+1]]),node,t+1))
    order=np.array(order,dtype=np.int64);validate_ranking(order,len(w))
    for n,(seq,gains)in sequences.items():
        inv=np.empty(len(w),int);inv[order]=np.arange(len(w));assert np.all(np.diff(inv[seq])>0)
    return {'order':order,'events':events,'leaf_info':leaf_info,'gain':gain_by_row,'unlocks':[]}

def progressive(nodes,x,r,w,ids,marginal_tail):
    N=len(w);selected=np.zeros(N,bool);heap=[];states={};info={};unlocks={};events=[];order=[]
    def activate(node_id,ancestors,activation_rank,parent):
        n=nodes[node_id];assert node_id not in info
        s=Cell(n,x,r,w,ids,selected,ancestors);states[node_id]=s
        info[node_id]={'ancestors':list(ancestors),'activation_rank':activation_rank,'parent':parent,'coarse_rows':[],'coarse_ranks':[],'coarse_gains':[],
                       'sequence':[],'gains':[],'phases':[],'initial_ancestor_cover_mass':float(s.w@s.initial_cover),'inherited_rows':len(ancestors),'coarse_count':0}
        if not n.leaf:unlocks[node_id]={'node':node_id,'level':n.level,'mass':n.mass,'Q':n.quality,'activation_rank':activation_rank,'unlock_rank':None,'children':list(n.children)}
        expose(node_id)
    def finish(node_id):
        n=nodes[node_id];data=info[node_id]
        if not n.leaf:
            assert unlocks[node_id]['unlock_rank']is None
            unlocks[node_id]['unlock_rank']=len(order)
            ancestors=data['ancestors']+data['coarse_rows']
            for child in n.children:activate(child,ancestors,len(order),node_id)
        states.pop(node_id,None)
    def expose(node_id):
        n=nodes[node_id];s=states[node_id]
        coarse_done=len(s.sequence)>=3 or s.taken.all()
        if coarse_done and not n.leaf:finish(node_id);return
        if s.taken.all():finish(node_id);return
        if not coarse_done:phase='final_rep'if n.leaf else'internal_rep';item=s.next_marginal()
        elif marginal_tail:phase='marginal_tail';item=s.next_marginal()
        else:
            phase='independent_residual'
            if s.residual_order is None:s.start_residual()
            item=s.next_residual()
        if item is None:finish(node_id);return
        j,gain=item;i=int(s.g[j]);assert not selected[i]
        heapq.heappush(heap,(-gain,n.key,int(ids[i]),node_id,j,phase))
    for n in sorted(nodes.values(),key=lambda n:n.key):
        if n.level==1:activate(n.index,[],0,None)
    while heap:
        minus,key,row,node_id,j,phase=heapq.heappop(heap);s=states[node_id];n=nodes[node_id];i=int(s.g[j]);gain=-minus
        assert int(ids[i])==row and not selected[i]
        selected[i]=True;order.append(i);rank=len(order)
        if phase!='independent_residual':s.emit_marginal(j,gain,rank)
        else:s.taken[j]=True;s.residual_pos+=1  # Deliberately no residual coverage update.
        data=info[node_id];data['sequence'].append(i);data['gains'].append(gain);data['phases'].append(phase)
        if phase in ['internal_rep','final_rep']:
            data['coarse_rows'].append(i);data['coarse_ranks'].append(rank);data['coarse_gains'].append(gain);data['coarse_count']+=1
        events.append({'index':i,'node':node_id,'level':n.level,'phase':phase,'priority':gain,'rank':rank,'local_position':len(data['sequence'])})
        if data['parent']is not None:assert rank>unlocks[data['parent']]['unlock_rank'],'Child emitted before unlock'
        expose(node_id)
    order=np.array(order,dtype=np.int64);validate_ranking(order,N);assert selected.all()
    event_by_node={}
    for event in events:event_by_node.setdefault(event['node'],[]).append(event['rank'])
    traces=[]
    for node,data in unlocks.items():
        coarse=info[node];child_ranks=[v for child in data['children']for v in event_by_node.get(child,[])]
        traces.append({**data,'coarse_rows':coarse['coarse_rows'],'coarse_ranks':coarse['coarse_ranks'],'coarse_gains':coarse['coarse_gains'],'first_child_item_rank':min(child_ranks)if child_ranks else None})
    gains=np.zeros(N)
    for e in events:gains[e['index']]=e['priority']
    return {'order':order,'events':events,'gain':gains,'unlocks':traces,'leaf_info':{i:a for i,a in info.items()if nodes[i].leaf},'node_info':info}

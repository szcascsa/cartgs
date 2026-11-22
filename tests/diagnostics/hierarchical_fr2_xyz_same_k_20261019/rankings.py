"""只用冻结几何/Raw SI构建A/B/C/F3；渲染前完成全部排序gate。"""
from common import *
import time,csv,struct,subprocess,sys
from fixed_marginal_reference import build
from progressive_ranker import progressive,Cell,validate_ranking
torch.set_num_threads(2)
os.environ['LD_LIBRARY_PATH']=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
with open(ROOT/'synthetic_tests.log','w')as log:subprocess.run([sys.executable,'-m','unittest','test_progressive'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
t=tensors(ROOT/'contributions.pt');allx=t['xyz'].numpy().astype(float);alls=t['scale'].numpy().astype(float);C=t['contribution'].numpy().astype(float)
assert t['maturity'].eq(1).all()and len(C)==len(read(ROOT/'input_manifest.json')['bank_frames'])
valid=np.isfinite(C).all(0)&(C>=0).all(0)&np.isfinite(allx).all(1)&np.isfinite(alls).all(1)&(alls>0).all(1)
observed=np.where(np.isfinite(C),C,0).sum(0)>0;candidate=valid&observed;ids=np.flatnonzero(candidate);N=len(ids);raw=C.mean(0)
x=allx[ids];s=alls[ids];w=raw[ids];r=np.exp(np.log(s).mean(1));d50=float(2*np.median(r));h=8*d50
np.savez_compressed(ROOT/'ranking_inputs.npz',row_ids=ids,xyz=x,scale=s,radius=r,contribution=C[:,ids],raw_I=w,candidate_mask=candidate)
np.savez_compressed(ROOT/'frozen_input_arrays.npz',xyz=allx,scale=alls,raw=raw,valid=valid,observed=observed)
dtype=np.dtype([('xyz','<f8',(3,)),('scale','<f8',(3,)),('raw','<f8'),('valid','u1'),('observed','u1')]);packed=np.empty(len(raw),dtype)
for key,a in [('xyz',allx),('scale',alls),('raw',raw),('valid',valid),('observed',observed)]:packed[key]=a
with open(ROOT/'hierarchy_input.bin','wb')as f:f.write(struct.pack('<q',len(raw)));packed.tofile(f)
with open(ROOT/'C_build.log','w')as log:subprocess.run([str(ROOT/'build/hierarchy_fixture'),str(ROOT/'hierarchy_input.bin'),str(ROOT/'structure_run1')],stdout=log,stderr=subprocess.STDOUT,check=True)
with open(ROOT/'structure_run1/sparse_hash_target_2048.bin','rb')as f:
    nt=struct.unpack('<q',f.read(8))[0];score=np.frombuffer(f.read(nt*8),'<f8').copy();leaf=np.frombuffer(f.read(nt*8),'<i8').copy();assert not f.read()
assert nt==len(raw)and np.all(leaf[ids]>=0)and np.all(leaf[~candidate]==-1)
tele=read(ROOT/'structure_run1/sparse_hash_target_2048_telemetry.json');rootkeys=np.floor(x/tele['h_root']).astype(np.int64);keys,root=np.unique(rootkeys,axis=0,return_inverse=True)
records=list(csv.DictReader(open(ROOT/'structure_run1/sparse_hash_target_2048_tree.csv')))
rep=np.zeros(nt,bool);level=np.zeros(nt,int);delta=np.zeros(nt)
for n in records:
    if n['leaf']!='1':continue
    for j in range(1,4):
        if n['rep'+str(j)]:
            row=int(n['rep'+str(j)]);assert not rep[row];rep[row]=True;level[row]=int(n['level']);delta[row]=float(n['delta'+str(j)])
np.testing.assert_array_equal(score[rep],delta[rep]);np.testing.assert_allclose(score[ids].sum(),w.sum(),rtol=1e-6,atol=1e-8)
np.savez_compressed(ROOT/'arm_rankings.npz',row_ids=ids,root=root,leaf=leaf[ids],rep_mask=rep[ids],rep_level=level[ids],rep_delta=delta[ids])
from experiment_data import load
data,base,nodes=load()
for n in nodes.values():
    state=Cell(n,x,r,w,ids)
    for j,gain in zip(n.original_reps,n.original_gains):
        jj,gg=state.next_marginal();assert int(state.g[jj])==j
        np.testing.assert_allclose(gg,gain,rtol=5e-12,atol=1e-8);state.emit_marginal(jj,gg,0)
    q=float(w[n.members]@state.cover)/(n.mass+1e-12)
    np.testing.assert_allclose(q,n.quality,rtol=1e-11,atol=1e-11)
    assert n.leaf==(n.quality>=.8 or n.level==3)
print('C_INDEPENDENT_ALL_NODE_FIRST3_Q_PASS',len(nodes),flush=True)
start=time.perf_counter();b,profile=build(x,r,w,ids,h);B_ms=(time.perf_counter()-start)*1000
b2,_=build(x,r,w,ids,h)
for k in b:np.testing.assert_array_equal(b[k],b2[k])
for cell in range(len(b['keys'])):
    seq=b['rank_local'][b['order'][b['cell'][b['order']]==cell]];np.testing.assert_array_equal(seq,np.arange(1,len(seq)+1))
print('B_EXACT_REPEAT_LOCAL_PREDECESSOR_PASS',flush=True)
start=time.perf_counter();f3=progressive(nodes,x,r,w,ids,True);F3_ms=(time.perf_counter()-start)*1000
repeat=progressive(nodes,x,r,w,ids,True)
np.testing.assert_array_equal(f3['order'],repeat['order']);np.testing.assert_array_equal(f3['gain'],repeat['gain']);assert f3['events']==repeat['events']and f3['unlocks']==repeat['unlocks']
orders={'A_Raw':np.lexsort((ids,-w)),'B_FixedMarginal':b['order'],'C_Hierarchical':np.lexsort((ids,-score[ids])),'F3_ProgressiveMarginal':f3['order']}
ordinal=np.empty(N,int);ordinal[f3['order']]=np.arange(N,0,-1)
scores={'A_Raw':w,'B_FixedMarginal':b['gain'],'C_Hierarchical':score[ids],'F3_ProgressiveMarginal':ordinal}
selection={}
for arm,order in orders.items():
    validate_ranking(order,N);previous=set()
    for ratio in read(ROOT/'protocol.json')['ratios']:
        K=int(N*ratio+.5);sel=np.sort(ids[order[:K]]);assert len(set(sel))==K and previous<=set(sel);previous=set(sel);selection[arm+'_'+str(ratio)]=sel
np.savez_compressed(ROOT/'arm_rankings.npz',**base,**{a+'_order':v for a,v in orders.items()},**{a+'_score':v for a,v in scores.items()},fixed_cell=b['cell'],fixed_keys=b['keys'])
np.savez_compressed(ROOT/'new_arm_rankings.npz',F3_ProgressiveMarginal_order=f3['order'],F3_ProgressiveMarginal_priority=f3['gain'])
np.savez_compressed(ROOT/'selected_rows.npz',**selection)
def write(n,rows):
    with open(ROOT/n,'w',newline='')as f:ww=csv.DictWriter(f,fieldnames=list(rows[0]));ww.writeheader();ww.writerows(rows)
events=[]
for e in f3['events']:
    n=nodes[e['node']];events.append(dict(arm='F3_ProgressiveMarginal',row_id=int(ids[e['index']]),global_rank=e['rank'],priority=e['priority'],node=n.index,root=int(root[e['index']]),level=e['level'],phase=e['phase'],local_position=e['local_position']))
write('ranking_emit_trace.csv',events)
unlock=[]
for e in f3['unlocks']:
    n=nodes[e['node']];unlock.append(dict(node=n.index,level=n.level,Raw_SI_mass=n.mass,Q=n.quality,activation_rank=e['activation_rank'],child_unlock_rank=e['unlock_rank'],number_of_children=len(e['children']),first_child_item_rank=e['first_child_item_rank'],coarse_rows=json.dumps([int(ids[i])for i in e['coarse_rows']]),coarse_ranks=json.dumps(e['coarse_ranks']),coarse_gains=json.dumps(e['coarse_gains'])))
if unlock:write('progressive_unlock_trace.csv',unlock)
subprocess.run([sys.executable,str(ROOT/'export_tree.py')],check=True)
weighted=sum(n.mass*n.quality for n in nodes.values()if n.leaf)/float(w.sum())
save(ROOT/'population.json',dict(N_total=nt,N_candidate=N,valid_count=int(valid.sum()),observed_count=int(observed.sum()),Bank=len(C),evaluation=len(read(ROOT/'input_manifest.json')['evaluation_frames']),d50=d50,h=h,occupied_roots=len(keys),h_root=tele['h_root'],final_leaves=sum(n.leaf for n in nodes.values()),final_representatives=int(rep.sum()),weighted_leaf_Q=weighted))
save(ROOT/'ranking_sanity.json',dict(all_passed=True,N_candidate=N,complete_permutations=True,same_K={str(ratio):int(N*ratio+.5)for ratio in read(ROOT/'protocol.json')['ratios']},nested=True,C_repeat_exact=True,F3_repeat_exact=True,B_repeat_exact=True,B_predecessor_order=True,all_C_node_reps_Q_independent_check=True,mass_relative_error=abs(score[ids].sum()-w.sum())/w.sum(),input_sha256=sha(ROOT/'hierarchy_input.bin'),tree_sha256=sha(ROOT/'structure_run1/sparse_hash_target_2048_tree.csv'),runtime_ms=dict(B=B_ms,C=tele['build_ms'],F3=F3_ms),quality_started=False))
print('RANKING_SANITY_ALL_PASS',read(ROOT/'population.json'),flush=True)

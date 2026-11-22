"""唯一新变量fixed32*d50；C数学与P算法原样。"""
from common import *
from progressive_ranker import progressive,Cell,validate_ranking
from tree_io import nodes_from_tree
from ranking_io import traces
import struct,time
assert read(ROOT.parent/'baseline_all_passed.json')['all_passed']
os.environ['LD_LIBRARY_PATH']=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
torch.set_num_threads(2)
x=dict(np.load(ROOT/'ranking_inputs.npz'));b=dict(np.load(ROOT/'rankings.npz'));ids=x['row_ids'];N=len(ids)
assert np.isfinite(x['scale']).all()and(x['scale']>0).all()
with open(ROOT/'new_C_build.log','w')as log:subprocess.run([str(ROOT.parent/'build/hierarchy_fixture'),str(ROOT/'hierarchy_input.bin'),str(ROOT/'new_tree'),'32'],stdout=log,stderr=subprocess.STDOUT,check=True)
t=read(ROOT/'new_tree/hierarchy_telemetry.json');assert t['h_root']==32*t['d50']and'target_occupied_roots'not in t and'bracket_low'not in t
np.testing.assert_allclose(t['d50'],2*np.median(x['radius']),rtol=1e-14,atol=0)
with open(ROOT/'new_tree/importance.bin','rb')as f:
    n=struct.unpack('<q',f.read(8))[0];score=np.frombuffer(f.read(n*8),'<f8').copy();leaf=np.frombuffer(f.read(n*8),'<i8').copy();assert not f.read()
candidate=np.zeros(n,bool);candidate[ids]=True;assert np.isfinite(score).all()and(score>=0).all()and(leaf[ids]>=0).all()and(leaf[~candidate]==-1).all()and(score[~candidate]==0).all()
keys,root=np.unique(np.floor(x['xyz']/t['h_root']).astype(np.int64),axis=0,return_inverse=True);assert len(keys)==t['achieved_occupied_roots']
nodes,records=nodes_from_tree(ROOT/'new_tree/hierarchy_tree.csv',x['xyz'],ids,root);rep=np.zeros(N,bool);level=np.zeros(N,int);maxmass=0.;maxgainerr=0.
for node in nodes.values():
    s=Cell(node,x['xyz'],x['radius'],x['raw_I'],ids)
    for index,gain in zip(node.original_reps,node.original_gains):
        j,v=s.next_marginal();assert int(s.g[j])==index;maxgainerr=max(maxgainerr,abs(v-gain));np.testing.assert_allclose(v,gain,rtol=5e-12,atol=1e-8);s.emit_marginal(j,v,0)
    q=float(x['raw_I'][node.members]@s.cover)/(node.mass+1e-12);np.testing.assert_allclose(q,node.quality,rtol=1e-11,atol=1e-11)
    assert node.leaf==(node.level==3 or node.quality>=.8)
    if node.leaf:
        expected=x['raw_I'][node.members]*(1-s.cover)
        for index,gain in zip(node.original_reps,node.original_gains):expected[np.flatnonzero(node.members==index)[0]]=gain;rep[index]=True;level[index]=node.level
        np.testing.assert_allclose(score[ids[node.members]],expected,rtol=1e-10,atol=1e-8)
        err=abs(expected.sum()-node.mass);maxmass=max(maxmass,err/max(node.mass,1e-12));assert err<=1e-8+1e-6*node.mass
start=time.perf_counter();p=progressive(nodes,x['xyz'],x['radius'],x['raw_I'],ids,False);ms=(time.perf_counter()-start)*1000;repeat=progressive(nodes,x['xyz'],x['radius'],x['raw_I'],ids,False)
np.testing.assert_array_equal(p['order'],repeat['order']);np.testing.assert_array_equal(p['gain'],repeat['gain']);assert p['events']==repeat['events']and p['unlocks']==repeat['unlocks']
traces(ROOT,'P32',p,nodes,ids,root,t['d50'])
ordinal=np.empty(N,int);ordinal[p['order']]=np.arange(N,0,-1);c_order=np.lexsort((ids,-score[ids]));np.testing.assert_array_equal(np.lexsort((ids,-ordinal)),p['order'])
b.update(C32_order=c_order,C32_score=score[ids],P32_order=p['order'],P32_score=ordinal,P32_priority=p['gain'],new_root=root,new_keys=keys,new_leaf=leaf[ids],new_rep_mask=rep,new_rep_level=level)
for arm in ['A','B','C0','P0','C32','P32']:validate_ranking(b[arm+'_order'],N)
np.savez_compressed(ROOT/'rankings.npz',**b)
np.testing.assert_allclose(score[ids].sum(),x['raw_I'].sum(),rtol=1e-6,atol=1e-8)
save(ROOT/'new_ranking_checks.json',dict(all_passed=True,N_candidate=N,new_h_over_d50=t['h_root']/t['d50'],occupied_roots=len(keys),new_first3_argmax_checked=True,max_gain_error=maxgainerr,max_leaf_mass_relative=maxmass,global_mass_relative=abs(score[ids].sum()-x['raw_I'].sum())/x['raw_I'].sum(),C32_repeat_exact=True,P32_repeat_exact=True,complete_permutations=True,unique_rows=True,nested=True,ordinal_order_exact=True,no_F1_F3_executed=True,progressive_ms=ms,new_tree_sha256=sha(ROOT/'new_tree/hierarchy_tree.csv'),new_hierarchy_ms=t['build_ms'],new_grid_ms=t['root_grouping_ms']))
print('NORMALIZED_RANKING_PASS',t['h_root'],len(keys),flush=True)

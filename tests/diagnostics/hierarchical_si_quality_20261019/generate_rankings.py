"""冻结排序：重跑当前C两次，复现旧B；E仅重用L1临时代表。"""
from pathlib import Path
import csv,json,hashlib,subprocess,struct,time
import numpy as np
from fixed_marginal_reference import build as fixed_reference
ROOT=Path(__file__).resolve().parent
S=Path('/home/prmi/zwj/redundancy-si-ranking-audit-20261019')
G=Path('/home/prmi/zwj/geometry-marginal-online-20261019')
H=Path('/home/prmi/zwj/sparse-hash-root-grid-20261019')
def read(p):return json.loads(Path(p).read_text())
def save(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def binary(p):
    with open(p,'rb')as f:
        n=struct.unpack('<q',f.read(8))[0];score=np.frombuffer(f.read(n*8),'<f8').copy();leaf=np.frombuffer(f.read(n*8),'<i8').copy();assert not f.read()
    return score,leaf
assert read(ROOT/'preparation.json')['all_passed']
x=dict(np.load(ROOT/'ranking_inputs.npz'));f=dict(np.load(ROOT/'fr1_room_inputs.npz'))
ids=x['row_ids'];N=len(ids);candidate=f['valid'].astype(bool)&f['observed'].astype(bool)
np.testing.assert_array_equal(np.flatnonzero(candidate),ids)
for a,b in [('xyz','xyz'),('scale','scale'),('raw_I','raw')]:np.testing.assert_array_equal(x[a],f[b][ids])
np.testing.assert_array_equal(x['raw_I'],x['contribution'].mean(0));assert x['contribution'].shape==(69,50330)
assert N==50330
old=dict(np.load(S/'rankings.npz'));orders={};scores={}
orders['A_Raw']=np.lexsort((ids,-x['raw_I']));scores['A_Raw']=x['raw_I']
np.testing.assert_array_equal(orders['A_Raw'],old['Raw_order'])
t=time.perf_counter();b,profile=fixed_reference(x['xyz'],x['radius'],x['raw_I'],ids,read(ROOT/'population.json')['h'])
for key in b:np.testing.assert_array_equal(b[key],old['GeometryOnly8_'+key])
for key in b:np.testing.assert_array_equal(b[key],np.load(G/'reference.npz')[key])
orders['B_FixedMarginal']=b['order'];scores['B_FixedMarginal']=b['gain'];B_ms=(time.perf_counter()-t)*1000
for c in range(len(b['keys'])):
    local=b['rank_local'][b['order'][b['cell'][b['order']]==c]]
    np.testing.assert_array_equal(local,np.arange(1,len(local)+1))
print('FIXED_MARGINAL_FULL_RANKING_EXACT',N,flush=True)
before=read(ROOT/'audit_before.json')['frozen_local_source'];expected=read(ROOT/'prior_source_expected.json')
for name in ['include/hierarchical_representative_si.h','src/hierarchical_representative_si.cpp','include/sparse_hash_root_grid.h','src/sparse_hash_root_grid.cpp']:
    assert before[name]==expected[name],name
    assert hashlib.sha256((H/'source'/name).read_bytes()).hexdigest()==expected[name],name
for run in [1,2]:
    with open(ROOT/f'hierarchy_run{run}.log','w')as log:
        subprocess.run([str(H/'build/sparse_root_test'),str(ROOT/'fr1_room_input.bin'),str(ROOT/f'structure_run{run}')],check=True,stdout=log,stderr=subprocess.STDOUT)
    print('HIERARCHY_REBUILD_PASS',run,flush=True)
c,leaf=binary(ROOT/'structure_run1/sparse_hash_target_2048.bin');c2,leaf2=binary(ROOT/'structure_run2/sparse_hash_target_2048.bin')
np.testing.assert_array_equal(c,c2);np.testing.assert_array_equal(leaf,leaf2)
prior_c,prior_leaf=binary(H/'fixture/sparse_hash_target_2048.bin');np.testing.assert_array_equal(c,prior_c);np.testing.assert_array_equal(leaf,prior_leaf)
tree=list(csv.DictReader(open(ROOT/'structure_run1/sparse_hash_target_2048_tree.csv')))
assert tree==list(csv.DictReader(open(ROOT/'structure_run2/sparse_hash_target_2048_tree.csv')))
assert tree==list(csv.DictReader(open(H/'fixture/sparse_hash_target_2048_tree.csv')))
telem=read(ROOT/'structure_run1/sparse_hash_target_2048_telemetry.json')
assert telem['achieved_occupied_roots']==2048 and abs(telem['h_root']-.29986755746366806)<1e-14
assert sum(telem['leaves_by_level'])==5601 and sum(telem['representatives_by_level'])==13715
weighted=sum(float(n['raw_mass'])*float(n['Q'])for n in tree if n['leaf']=='1')/telem['raw_sum']
assert abs(weighted-.8837622487608384)<1e-14
rep=np.zeros(len(c),bool);level=np.zeros(len(c),np.int64);delta=np.zeros(len(c));root=np.full(len(c),-1,np.int64)
keys=np.floor(f['xyz'][ids]/telem['h_root']).astype(np.int64);unique,inverse=np.unique(keys,axis=0,return_inverse=True);root[ids]=inverse
for n in tree:
    if n['leaf']!='1':continue
    for k in range(1,4):
        if n['rep'+str(k)]:
            row=int(n['rep'+str(k)]);assert not rep[row];rep[row]=True;level[row]=int(n['level']);delta[row]=float(n['delta'+str(k)])
assert rep.sum()==13715;np.testing.assert_array_equal(c[rep],delta[rep])
scores['C_Hierarchical']=c[ids];orders['C_Hierarchical']=np.lexsort((ids,-c[ids]))
np.testing.assert_array_equal(orders['C_Hierarchical'],np.lexsort((ids,-c2[ids])))
scores['D_RepresentativeOnly']=np.where(rep[ids],c[ids],0);orders['D_RepresentativeOnly']=np.lexsort((ids,-scores['D_RepresentativeOnly']))
# Diagnostic E: the same L1 node's already-recorded greedy representatives and Delta, without splitting.
e=np.zeros(len(c));r_all=np.exp(np.log(f['scale']).mean(1))
for n in tree:
    if int(n['level'])!=1:continue
    members=ids[inverse==int(n['root'])];assert len(members)==int(n['occupancy']);cover=np.zeros(len(members))
    for k in range(1,4):
        if not n['rep'+str(k)]:continue
        row=int(n['rep'+str(k)]);sigma=.5*(r_all[row]+r_all[members]);kernel=np.exp(-((f['xyz'][row]-f['xyz'][members])**2).sum(1)/(2*sigma*sigma+1e-12));kernel[members==row]=1
        cover=np.maximum(cover,kernel)
    e[members]=f['raw'][members]*(1-cover)
    for k in range(1,4):
        if n['rep'+str(k)]:e[int(n['rep'+str(k)])]=float(n['delta'+str(k)])
    assert abs(e[members].sum()-f['raw'][members].sum())<1e-8+1e-6*f['raw'][members].sum()
    if n['leaf']=='1':np.testing.assert_allclose(e[members],c[members],rtol=1e-12,atol=1e-10)
scores['E_L1Only']=e[ids];orders['E_L1Only']=np.lexsort((ids,-e[ids]))
ratios=read(ROOT/'preregistration.json')['ratios'];sameK={};selection={}
for arm,order in orders.items():
    np.testing.assert_array_equal(np.sort(order),np.arange(N));previous=set();sameK[arm]={}
    assert np.isfinite(scores[arm]).all()and(scores[arm]>=0).all()
    for ratio in ratios:
        K=int(np.floor(ratio*N+.5));sel=np.sort(ids[order[:K]]);assert len(set(sel))==K and previous<=set(sel);previous=set(sel)
        sameK[arm][str(ratio)]=K;selection[arm+'_'+str(ratio)]=sel
np.savez_compressed(ROOT/'arm_rankings.npz',row_ids=ids,**{a+'_order':v for a,v in orders.items()},**{a+'_score':v for a,v in scores.items()},rep_mask=rep[ids],rep_level=level[ids],rep_delta=delta[ids],leaf=leaf[ids],root=root[ids],fixed_cell=b['cell'],fixed_keys=b['keys'])
np.savez_compressed(ROOT/'selected_rows.npz',**selection)
save('ranking_sanity.json',{'all_passed':True,'N_total':len(c),'N_candidate':N,'all_arms_identical_population':True,'same_K':sameK,'nested':True,'complete_permutations':True,'C_repeat_complete_ranking_exact':True,'C_prior_score_tree_exact':True,'B_prior_complete_ranking_exact':True,'raw_contributions_unchanged':True,'structure':{'occupied_roots':2048,'h_root':telem['h_root'],'leaves':5601,'representatives':13715,'weighted_leaf_Q':weighted},'runtime_ms':{'B_reference':B_ms,'C_build':telem['build_ms']},'E_uses_same_root_provisional_representatives':True})
(ROOT/'protocol.json').write_bytes((ROOT/'preregistration.json').read_bytes())
print('RANKING_SANITY_ALL_PASS',sameK['C_Hierarchical'],flush=True)

"""独立结构验证与统计，不训练、不渲染、不读取图像/质量。"""
from pathlib import Path
import csv,json,struct,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text())
def save(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
x=dict(np.load(ROOT/'fr1_room_inputs.npz'));candidate=x['valid'].astype(bool)&x['observed'].astype(bool)
rows=np.flatnonzero(candidate);xyz=x['xyz'];weights=x['raw'];radius=np.exp(np.log(x['scale']).mean(1));total=weights[rows].sum()
assert len(rows)==50330 and len(weights)==52635
def binary(p):
    with open(p,'rb')as f:
        n=struct.unpack('<q',f.read(8))[0];score=np.frombuffer(f.read(n*8),'<f8').copy();assignment=np.frombuffer(f.read(n*8),'<i8').copy();assert not f.read()
    return score,assignment
def distribution(a):
    a=np.asarray(a);return {k:float(v)for k,v in zip(['min','mean','median','p90','p95','p99','max'],[a.min(),a.mean(),*np.quantile(a,[.5,.9,.95,.99]),a.max()])}if len(a)else dict.fromkeys(['min','mean','median','p90','p95','p99','max'],0.)
def verify_backend(stem):
    telemetry=read('fixture/'+stem+'_telemetry.json')
    nodes=list(csv.DictReader(open(ROOT/'fixture'/f'{stem}_tree.csv')))
    score,assignment=binary(ROOT/'fixture'/f'{stem}.bin')
    assert np.isfinite(score).all()and(score>=0).all()and(score[~candidate]==0).all()
    assert(assignment[~candidate]==-1).all()and(assignment[candidate]>=0).all()
    roots=[n for n in nodes if int(n['level'])==1];keylist=[tuple(int(n['key_'+a])for a in 'xyz')for n in roots]
    assert keylist==sorted(set(keylist))
    h=telemetry['h_root'];by_root={}
    if stem=='sparse_hash_target_2048':
        keys=np.floor(xyz[rows]/h).astype(np.int64)
        unique,inverse=np.unique(keys,axis=0,return_inverse=True)
        assert unique.tolist()==[list(k)for k in keylist]
        for i,n in enumerate(roots):by_root[int(n['root'])]=rows[inverse==i]
        assert telemetry['origin']==[0,0,0]and telemetry['target_occupied_roots']==2048
        trace=list(csv.DictReader(open(ROOT/'fixture'/f'{stem}_h_search.csv')))
        verified_trace=[]
        for sample in trace:
            sample_h=float(sample['h']);count=len(np.unique(np.floor(xyz[rows]/sample_h).astype(np.int64),axis=0))
            assert count==int(sample['occupied']);verified_trace.append((sample_h,count))
        optimum=min(verified_trace,key=lambda v:(abs(v[1]-2048),-v[0]))
        assert optimum==(h,len(roots))
        assert np.isclose(telemetry['h_seed'],16*np.median(radius[rows]),rtol=1e-14)
        save('independent_h_search.json',{'all_passed':True,'candidate_only':True,'samples_checked':len(trace),'tested_minimum_error':abs(optimum[1]-2048),'chosen_h':h,'achieved':len(roots),'tie_break_larger_h':True,'global_optimum_claim':False})
    else:
        dims=np.array(telemetry['dimensions']);lo=np.array(telemetry['AABB_min']);size=np.array(telemetry['cell_size'])
        keys=np.floor((xyz[rows]-lo)/size).astype(np.int64);keys=np.maximum(0,np.minimum(keys,dims-1))
        for n,k in zip(roots,keylist):by_root[int(n['root'])]=rows[(keys==k).all(1)]
    expected=np.zeros(len(score));leaf_errors=[];gain_errors=[];q_errors=[];reps=[];node_rows={}
    for node in nodes:
        level=int(node['level']);key=int(node['node']);path=node['path'];members=by_root[int(node['root'])]
        # Reconstruct membership using root and child >=midpoint bits, matching the declared partition.
        parent_root=next(n for n in roots if n['root']==node['root'])
        low=np.array([float(parent_root[a+'min'])for a in 'xyz']);high=np.array([float(parent_root[a+'max'])for a in 'xyz'])
        if stem=='sparse_hash_target_2048':
            np.testing.assert_allclose(high-low,h,rtol=1e-12,atol=1e-13)
            np.testing.assert_array_equal(low,np.array([int(node['key_'+a])for a in 'xyz'])*h)
        for digit in path:
            mid=low+.5*(high-low);bits=(xyz[members]>=mid).astype(np.int64);children=bits[:,0]+2*bits[:,1]+4*bits[:,2]
            child=int(digit);members=members[children==child]
            for a in range(3):
                if child&(1<<a):low[a]=mid[a]
                else:high[a]=mid[a]
        assert len(members)==int(node['occupancy']);assert 1<=level<=3 and len(path)==level-1
        node_rows[key]=members
        np.testing.assert_allclose([float(node[a+'min'])for a in 'xyz'],low,rtol=0,atol=1e-12)
        np.testing.assert_allclose([float(node[a+'max'])for a in 'xyz'],high,rtol=0,atol=1e-12)
        w=weights[members];cover=np.zeros(len(members));selected=[];deltas=[]
        for k in range(1,4):
            if not node['rep'+str(k)]:continue
            rep=int(node['rep'+str(k)]);assert rep in members and rep not in selected
            sigma=.5*(radius[rep]+radius[members]);dist=((xyz[members]-xyz[rep])**2).sum(1)
            kernel=np.exp(-dist/(2*sigma*sigma+1e-12));kernel[members==rep]=1
            gain=np.sum(w*np.maximum(kernel-cover,0));recorded=float(node['delta'+str(k)])
            gain_errors.append(abs(gain-recorded));assert np.isclose(gain,recorded,rtol=5e-12,atol=1e-8)
            cover=np.maximum(cover,kernel);selected.append(rep);deltas.append(recorded)
        q=(w*cover).sum()/(w.sum()+1e-12);q_errors.append(abs(q-float(node['Q'])))
        assert np.isclose(q,float(node['Q']),rtol=5e-12,atol=1e-12)
        assert (node['leaf']=='1')==(level==3 or q>=.8)
        assert len(selected)==min(3,len(members))
        if node['leaf']=='1':
            assert(assignment[members]==key).all();expected[members]=w*(1-cover)
            for rep,delta in zip(selected,deltas):expected[rep]=delta
            leaf_errors.append(abs(score[members].sum()-w.sum()));reps.extend(selected)
        else:
            children=[c for c in nodes if c['root']==node['root']and len(c['path'])==level and c['path'].startswith(path)]
            assert children and sum(int(c['occupancy'])for c in children)==len(members)
    np.testing.assert_allclose(expected,score,rtol=5e-12,atol=1e-8)
    assert len(reps)==len(set(reps));assert abs(total-score.sum())<=1e-6*total
    leaves=[n for n in nodes if n['leaf']=='1'];unresolved=[n for n in leaves if int(n['level'])==3 and float(n['Q'])<.8]
    mass=np.array([float(n['raw_mass'])for n in roots]);occupancy=np.array([int(n['occupancy'])for n in roots]);maxN=int(occupancy.argmax());maxM=int(mass.argmax())
    result={'backend':stem,'telemetry':telemetry,'root_occupancy':distribution(occupancy),'root_raw_mass':distribution(mass),
            'target_error':abs(len(roots)-2048),'target_relative_error':abs(len(roots)-2048)/2048,
            'key_min':np.min(np.array(keylist),axis=0).tolist(),'key_max':np.max(np.array(keylist),axis=0).tolist(),
            'largest_N_root':{'key':keylist[maxN],'N':int(occupancy[maxN]),'N_fraction':float(occupancy[maxN]/len(rows)),'raw_mass':float(mass[maxN]),'raw_mass_fraction':float(mass[maxN]/total)},
            'largest_SI_root':{'key':keylist[maxM],'N':int(occupancy[maxM]),'raw_mass':float(mass[maxM]),'raw_mass_fraction':float(mass[maxM]/total)},
            'local_similarity_elements':sum(min(3,int(n['occupancy']))*(int(n['occupancy'])**2+int(n['occupancy']))for n in nodes),
            'leaves':len(leaves),'representatives':len(reps),'L3_unresolved':len(unresolved),'L3_unresolved_fraction':len(unresolved)/max(1,telemetry['leaves_by_level'][2]),
            'L3_unresolved_raw_mass':sum(float(n['raw_mass'])for n in unresolved),'L3_unresolved_mass_fraction':sum(float(n['raw_mass'])for n in unresolved)/total,
            'weighted_leaf_Q':sum(float(n['raw_mass'])*float(n['Q'])for n in leaves)/total,
            'verification':{'all_passed':True,'mass_absolute':float(abs(total-score.sum())),'max_leaf_mass_error':float(max(leaf_errors,default=0)),
                            'max_gain_error':float(max(gain_errors,default=0)),'max_Q_error':float(max(q_errors,default=0)),'max_output_reconstruction_error':float(np.max(np.abs(expected-score))),
                            'candidate_mask_exact':True,'parent_discard':True,'child_occupancy_exact':True,'cubic_roots':stem=='sparse_hash_target_2048'}}
    levels=[]
    for level in range(1,4):
        ns=[n for n in nodes if int(n['level'])==level];masses=np.array([float(n['raw_mass'])for n in ns]);q=np.array([float(n['Q'])for n in ns]);occ=np.array([int(n['occupancy'])for n in ns])
        levels.append({'level':level,'nodes':len(ns),'split':sum(n['leaf']=='0'for n in ns),'leaves':sum(n['leaf']=='1'for n in ns),'representatives':sum(sum(bool(n['rep'+str(k)])for k in range(1,4))for n in ns if n['leaf']=='1'),
                       'occupancy':distribution(occ),'Q':distribution(q),'Q_p10':float(np.quantile(q,.1))if len(q)else 0.,'weighted_Q':float(np.average(q,weights=masses))if masses.sum()else 0.})
    result['levels']=levels
    with open(ROOT/f'{stem}_root_statistics.csv','w',newline='')as f:
        writer=csv.writer(f);writer.writerow(['key_x','key_y','key_z','occupancy','raw_mass','N_fraction','raw_mass_fraction'])
        for key,n,m in zip(keylist,occupancy,mass):writer.writerow([*key,int(n),float(m),n/len(rows),m/total])
    np.savez_compressed(ROOT/(stem+'_output.npz'),hierarchical_importance=score,leaf_assignment=assignment,row_ids=x['row_ids'])
    return result
old=verify_backend('legacy_aabb_2048');new=verify_backend('sparse_hash_target_2048')
prior_score,prior_assignment=binary(ROOT/'old_reference/fr1_room_output.bin');score,assignment=binary(ROOT/'fixture/legacy_aabb_2048.bin')
np.testing.assert_array_equal(prior_score,score);np.testing.assert_array_equal(prior_assignment,assignment)
prior_nodes=list(csv.DictReader(open(ROOT/'old_reference/fr1_room_tree.csv')));current_nodes=list(csv.DictReader(open(ROOT/'fixture/legacy_aabb_2048_tree.csv')))
assert len(prior_nodes)==len(current_nodes)
for a,b in zip(prior_nodes,current_nodes):assert all(a[k]==b[k]for k in a), (a,b)
assert old['telemetry']['cells_by_level'][0]==36 and old['root_occupancy']['max']==39334
gates={'occupied_roots_near_target':new['target_relative_error']<=.1,'root_max_occupancy_substantially_reduced':new['root_occupancy']['max']<old['root_occupancy']['max']/2,
       'largest_root_not_majority':new['largest_N_root']['N_fraction']<.5 and new['largest_SI_root']['raw_mass_fraction']<.5,
       'unresolved_raw_mass_reduced':new['L3_unresolved_mass_fraction']<old['L3_unresolved_mass_fraction'],
       'weighted_Q_increased':new['weighted_leaf_Q']>old['weighted_leaf_Q'],'mass_and_representative_parity':new['verification']['all_passed'] and old['verification']['all_passed']}
gates={key:bool(value) for key,value in gates.items()}
result={'decision':'SPARSE ROOT STRUCTURAL GO'if all(gates.values())else'SPARSE ROOT STRUCTURAL NO-GO','gates':gates,'legacy_reference_bitwise_equal':True,'quality_evaluation':False,'online_run':False,'legacy':old,'sparse':new}
save('sparse_root_results.json',result)
print(json.dumps({'decision':result['decision'],'gates':gates,'h_root':new['telemetry']['h_root'],'occupied':new['telemetry']['cells_by_level'][0],'max_occupancy':new['root_occupancy']['max'],'weighted_Q':new['weighted_leaf_Q'],'unresolved_mass_fraction':new['L3_unresolved_mass_fraction']},indent=2),flush=True)

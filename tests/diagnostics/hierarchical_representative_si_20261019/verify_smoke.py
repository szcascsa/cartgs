"""独立 NumPy 公式检查：不运行 hierarchy，不读 GT，不改变参数。"""
from pathlib import Path
import csv,json,struct
import numpy as np
ROOT=Path(__file__).resolve().parent
x=dict(np.load(ROOT/'fr1_room_inputs.npz'));t=json.loads((ROOT/'smoke/fr1_room_telemetry.json').read_text())
nodes=list(csv.DictReader(open(ROOT/'smoke/fr1_room_tree.csv')))
with open(ROOT/'smoke/fr1_room_output.bin','rb')as f:
    n=struct.unpack('<q',f.read(8))[0];out=np.frombuffer(f.read(n*8),'<f8').copy();assignment=np.frombuffer(f.read(n*8),'<i8').copy();assert not f.read()
assert n==len(x['raw']) and np.isfinite(out).all() and (out>=0).all()
candidate=x['valid'].astype(bool)&x['observed'].astype(bool)
assert (assignment[candidate]>=0).all() and (assignment[~candidate]==-1).all() and (out[~candidate]==0).all()
radius=np.exp(np.log(x['scale']).mean(axis=1));expected=np.zeros(n);mass_errors=[];gain_errors=[];cover_errors=[]
rng=np.random.default_rng(20261019);sampled_argmax=0;representatives=[];node_levels=[];leaf_mass=[];leaf_Q=[]
for node in nodes:
    key=int(node['node']);level=int(node['level']);node_levels.append(level)
    lo=np.array([float(node[a+'min'])for a in 'xyz']);hi=np.array([float(node[a+'max'])for a in 'xyz'])
    rows=np.flatnonzero(candidate & (x['xyz']>=lo).all(1) & (x['xyz']<hi).all(1))
    assert len(rows)==int(node['occupancy'])
    w=x['raw'][rows];cover=np.zeros(len(rows));mass=w.sum();gains=[];reps=[]
    assert abs(mass-float(node['raw_mass']))<=max(1e-9,1e-12*mass)
    for k in range(1,4):
        if not node['rep'+str(k)]:continue
        rep=int(node['rep'+str(k)]);assert rep in rows and rep not in reps
        r=.5*(radius[rep]+radius[rows]);d2=((x['xyz'][rows]-x['xyz'][rep])**2).sum(axis=1)
        kernel=np.exp(-d2/(2*r*r+1e-12));kernel[rows==rep]=1.
        gain=np.sum(w*np.maximum(kernel-cover,0));recorded=float(node['delta'+str(k)])
        gain_errors.append(abs(gain-recorded));assert np.isclose(gain,recorded,rtol=2e-12,atol=1e-8)
        # Fixed read-only sample checks selected gain dominates candidates; full dense oracle is in synthetic tests.
        choices=np.setdiff1d(rng.choice(rows,min(16,len(rows)),replace=False),reps)
        if len(choices):
            r=.5*(radius[choices,None]+radius[rows][None,:]);d2=((x['xyz'][choices,None,:]-x['xyz'][rows][None,:,:])**2).sum(axis=2)
            kk=np.exp(-d2/(2*r*r+1e-12));kk[choices[:,None]==rows[None,:]]=1
            alternatives=(np.maximum(kk-cover,0)*w).sum(axis=1)
            assert alternatives.max()<=recorded+max(1e-8,2e-12*abs(recorded));sampled_argmax+=len(choices)
        cover=np.maximum(cover,kernel);reps.append(rep);gains.append(recorded)
    assert len(reps)==min(3,len(rows))
    quality=(w*cover).sum()/(mass+1e-12)
    cover_errors.append(abs(quality-float(node['Q'])));assert np.isclose(quality,float(node['Q']),rtol=2e-12,atol=1e-12)
    leaf=node['leaf']=='1';assert leaf==(level==3 or quality>=.8)
    assert len(node['path'])==level-1
    if leaf:
        assert (assignment[rows]==key).all()
        expected[rows]=w*(1-cover)
        for rep,gain in zip(reps,gains):expected[rep]=gain
        mass_errors.append(abs(out[rows].sum()-mass));representatives.extend(reps);leaf_mass.append(mass);leaf_Q.append(quality)
    else:
        children=[c for c in nodes if c['root']==node['root'] and len(c['path'])==level and c['path'].startswith(node['path'])]
        assert children and sum(int(c['occupancy'])for c in children)==len(rows)
assert max(node_levels)<=3 and len(representatives)==len(set(representatives))
np.testing.assert_allclose(out,expected,rtol=2e-12,atol=1e-8)
assert abs(out.sum()-x['raw'][candidate].sum())<=1e-6*x['raw'][candidate].sum()
assert np.prod(t['dimensions'])==2048 and t['initial_cells']==2048
unresolved=[c for c in nodes if c['leaf']=='1' and int(c['level'])==3 and float(c['Q'])<.8]
low_Q_mass=sum(float(c['raw_mass'])for c in unresolved)
result={'all_passed':True,'population':n,'candidate_N':int(candidate.sum()),'nodes':len(nodes),'leaves':len(mass_errors),'representatives':len(representatives),'mass_error_max_leaf':max(mass_errors),'independent_gain_max_abs':max(gain_errors),'independent_Q_max_abs':max(cover_errors),'reconstructed_output_max_abs':float(np.abs(expected-out).max()),'sampled_alternative_gain_checks':sampled_argmax,'parent_scores_discarded':True,'invalid_rows_not_promoted':True,'leaf_output_reconstruction':True,'unresolved_L3_count':len(unresolved),'unresolved_L3_fraction':len(unresolved)/t['leaves_by_level'][2],'unresolved_L3_raw_mass_fraction':low_Q_mass/t['raw_sum'],'raw_mass_weighted_leaf_Q':float(np.average(leaf_Q,weights=leaf_mass))}
(ROOT/'independent_smoke_verification.json').write_text(json.dumps(result,indent=2)+'\n')
np.savez_compressed(ROOT/'hierarchical_fr1_room_output.npz',hierarchical_importance=out,leaf_assignment=assignment,row_ids=x['row_ids'])
print(json.dumps(result),flush=True)

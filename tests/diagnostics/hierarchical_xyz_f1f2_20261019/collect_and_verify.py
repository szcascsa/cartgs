"""本地独立核对SHA、complete same-K ranking、metric均值及所有机制gate。"""
from pathlib import Path
import json,hashlib,tarfile,csv
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return list(csv.DictReader(open(p)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with tarfile.open(ROOT/'xyz_f1_f2_mechanism_evidence.tar.gz','r:gz')as tar:
    for m in tar.getmembers():assert m.isfile()and(OUT/m.name).resolve().is_relative_to(OUT.resolve()),m.name
    tar.extractall(OUT,filter='data')
bundle=read(OUT/'bundle_manifest.json')
for name,v in bundle['files'].items():
    p=OUT/name;assert p.stat().st_size==v['bytes']and sha(p)==v['sha256'],name
results={};total=0
for scene in ['fr1_xyz','fr2_xyz']:
    p=OUT/scene;reg=read(p/'preregistration.json');a=np.load(p/'arm_rankings.npz');b=np.load(p/'new_arm_rankings.npz');ids=a['row_ids'];N=len(ids);total+=24*len(read(p/'input_manifest.json')['evaluation_frames'])
    arms=reg['baseline']+reg['new_arms']
    for arm in arms:
        source=a if arm in reg['baseline']else b;order=source[arm+'_order'];np.testing.assert_array_equal(np.sort(order),np.arange(N));previous=set()
        assert np.isfinite(source[arm+'_score']).all()
        for ratio in reg['ratios']:
            K=int(ratio*N+.5);sel=set(ids[order[:K]]);assert len(sel)==K and previous<=sel;previous=sel
    np.testing.assert_array_equal(a['F3_ProgressiveMarginal_order'],b['F3_ProgressiveMarginal_order'])
    ev=rows(p/'ranking_emit_trace.csv');tree={int(z['node']):z for z in rows(p/'structure_run1/sparse_hash_target_2048_tree.csv')}
    for arm in reg['new_arms']:
        events=[z for z in ev if z['arm']==arm];assert len(events)==N and len(set(int(z['row_id'])for z in events))==N
        assert all(np.isfinite(float(z['priority']))and float(z['priority'])>=-1e-12 for z in events)
    grouped={}
    for e in ev:
        if e['arm']=='F1_MarginalTail':grouped.setdefault(int(e['node']),[]).append(int(e['row_id']))
    for node,sequence in grouped.items():
        n=tree[node];assert n['leaf']=='1';assert sequence[:3]==[int(n['rep'+str(j)])for j in range(1,4)if n['rep'+str(j)]]
    pv=rows(p/'quality_per_view.csv');q=rows(p/'quality_summary.csv');frames=read(p/'input_manifest.json')['evaluation_frames'];assert len(pv)==24*len(frames)and len(q)==24
    maxerror=0.
    for z in q:
        group=[v for v in pv if v['arm']==z['arm']and float(v['ratio'])==float(z['ratio'])];assert len(group)==len(frames)
        for metric in ['psnr','ssim','lpips','photo']:
            values=np.array([float(v[metric])for v in group]);assert np.isfinite(values).all();err=abs(values.mean()-float(z[metric]));maxerror=max(maxerror,err);assert err<1e-12
    Q={(z['arm'],float(z['ratio'])):float(z['psnr'])for z in q};R={(z['arm'],float(z['ratio'])):z for z in rows(p/'selected_redundancy.csv')};res=read(p/'mechanism_result.json')
    delta={arm:np.array([Q[arm,r]-Q['C_Hierarchical',r]for r in reg['ratios']])for arm in reg['new_arms']}
    for arm,v in delta.items():np.testing.assert_allclose(v,res['delta_PSNR_vs_C'][arm],rtol=0,atol=1e-12)
    f1red=all(float(R['F1_MarginalTail',r]['mean_redundancy'])<=.9*float(R['C_Hierarchical',r]['mean_redundancy'])and int(R['F1_MarginalTail',r]['pairs_gt_0.9'])<=int(R['C_Hierarchical',r]['pairs_gt_0.9'])for r in [.1,.15])
    f1=bool(f1red and delta['F1_MarginalTail'][2:].mean()>=-.03);f2=bool(delta['F2_ProgressiveResidual'][0]>=.05 and np.all(delta['F2_ProgressiveResidual'][1:]>=-.03));f3=bool(delta['F3_ProgressiveMarginal'].mean()>=0 and delta['F3_ProgressiveMarginal'][0]>0 and all(float(R['F3_ProgressiveMarginal',r]['mean_redundancy'])<float(R['C_Hierarchical',r]['mean_redundancy'])for r in [.1,.15]))
    assert[f1,f2,f3]==[res['F1_supported'],res['F2_supported'],res['F3_supported']]
    assert read(p/'final_checks.json')['all_passed']and read(p/'independent_replay_checks.json')['all_passed']
    for path in p.glob('*.csv'):
        for record in rows(path):assert all(str(v).lower()not in ['nan','inf','-inf','infinity','-infinity']for v in record.values()),path.name
    results[scene]=dict(all_passed=True,N_candidate=N,complete_same_K_nested=True,F1_first3_exact=True,aggregate_max_abs_error=maxerror,F1_supported=f1,F2_supported=f2,F3_supported=f3,baseline_max_abs=read(p/'baseline_reproduction.json')['per_view_max_abs'])
result=dict(all_passed=True,files_verified=len(bundle['files']),total_renders=total,scenes=results,bundle_sha256=sha(ROOT/'xyz_f1_f2_mechanism_evidence.tar.gz'))
(OUT/'local_delivery_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

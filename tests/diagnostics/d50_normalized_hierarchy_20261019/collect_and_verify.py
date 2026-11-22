"""本地独立验证SHA、d50/root keys、same-K、metric聚合及A-D/Strong判定。"""
from pathlib import Path
import json,hashlib,tarfile,csv
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(open(p)))
with tarfile.open(ROOT/'d50_normalized_hierarchy_evidence.tar.gz','r:gz')as tar:
    for m in tar.getmembers():assert m.isfile()and(OUT/m.name).resolve().is_relative_to(OUT.resolve()),m.name
    tar.extractall(OUT,filter='data')
manifest=read(OUT/'bundle_manifest.json')
for name,v in manifest['files'].items():
    p=OUT/name;assert p.stat().st_size==v['bytes']and sha(p)==v['sha256'],name
reg=read(OUT/'preregistration.json');old={};new={};quality={};summary={};Ntotal=0
for s in reg['scenes']:
    p=OUT/s;x=np.load(p/'ranking_inputs.npz');b=np.load(p/'rankings.npz');ids=x['row_ids'];N=len(ids);Ntotal+=N
    frozen=np.load(p/'frozen_arrays.npz');np.testing.assert_array_equal(np.flatnonzero(frozen['valid']&frozen['observed']),ids)
    np.testing.assert_array_equal(x['raw_I'],x['contribution'].mean(0));assert np.isfinite(x['scale']).all()and(x['scale']>0).all()
    t=read(p/'new_tree/hierarchy_telemetry.json');expected=2*np.median(np.exp(np.log(x['scale']).mean(1)));np.testing.assert_allclose(t['d50'],expected,rtol=1e-14,atol=0);assert t['h_root']==32*t['d50']and'target_occupied_roots'not in t and'bracket_low'not in t
    keys,root=np.unique(np.floor(x['xyz']/t['h_root']).astype(np.int64),axis=0,return_inverse=True);np.testing.assert_array_equal(root,b['new_root']);assert len(keys)==t['achieved_occupied_roots']
    for a in reg['arms']:
        order=b[a+'_order'];np.testing.assert_array_equal(np.sort(order),np.arange(N));assert np.isfinite(b[a+'_score']).all();prev=set()
        for r in reg['ratios']:
            K=int(r*N+.5);selected=set(ids[order[:K]]);assert len(selected)==K and prev<=selected;prev=selected
    np.testing.assert_allclose(b['C32_score'].sum(),x['raw_I'].sum(),rtol=1e-6,atol=1e-8)
    pv=rows(p/'quality_per_view.csv');q=rows(p/'quality_summary.csv');frames=read(p/'input_manifest.json')['evaluation_frames'];assert len(q)==24 and len(pv)==24*len(frames)
    Q={(z['arm'],float(z['ratio'])):z for z in q};quality[s]=Q;maxerr=0.
    for z in q:
        group=[a for a in pv if a['arm']==z['arm']and float(a['ratio'])==float(z['ratio'])];assert len(group)==len(frames)
        for m in ['psnr','ssim','lpips','photo']:
            v=np.array([float(a[m])for a in group]);assert np.isfinite(v).all();err=abs(v.mean()-float(z[m]));assert err<1e-12;maxerr=max(maxerr,err)
    old[s]=[float(Q['P0',r]['psnr'])-float(Q['C0',r]['psnr'])for r in reg['ratios']];new[s]=[float(Q['P32',r]['psnr'])-float(Q['C32',r]['psnr'])for r in reg['ratios']]
    assert read(p/'baseline_reproduction.json')['all_passed']and max(read(p/'baseline_reproduction.json')['per_view_max_abs'].values())==0
    assert read(p/'independent_replay_checks.json')['all_passed']and read(p/'new_ranking_checks.json')['all_passed']
    summary[s]=dict(all_passed=True,N_candidate=N,new_h_over_d50=t['h_root']/t['d50'],new_roots=len(keys),aggregate_max_abs=maxerr,old_progressive_delta=old[s],new_progressive_delta=new[s])
res=read(OUT/'mechanism_result.json');A=bool(new['fr2_xyz'][0]-old['fr2_xyz'][0]>=.05);B=bool(np.mean(new['fr1_xyz'][1:3])-np.mean(old['fr1_xyz'][1:3])>=.05);C=bool(new['fr1_xyz'][0]>=.1);D=bool(np.mean(new['fr1_room'])-np.mean(old['fr1_room'])>=-.03)
assert[A,B,C,D]==list(res['gates'].values());assert all([A,B,C,D])==res['primary_supported']
strong=bool(all(np.mean(new[s])>=0 for s in reg['scenes'])and sum(new[s][0]>0 for s in reg['scenes'])>=2 and all(min(new[s])>=-.05 for s in reg['scenes']));assert strong==res['strong_supported']
for s in reg['scenes']:
    np.testing.assert_allclose(res['old_progressive_delta_PSNR'][s],old[s],rtol=0,atol=1e-12);np.testing.assert_allclose(res['new_progressive_delta_PSNR'][s],new[s],rtol=0,atol=1e-12)
assert read(OUT/'delivery_checks.json')['report_sha256']==sha(OUT/'D50_NORMALIZED_HIERARCHY_AUDIT.md')
result=dict(all_passed=True,files_verified=len(manifest['files']),scenes=summary,primary_gates_recomputed=[A,B,C,D],strong_recomputed=strong,independent_replay_rows=2*Ntotal,bundle_sha256=sha(ROOT/'d50_normalized_hierarchy_evidence.tar.gz'))
(OUT/'local_delivery_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

"""本地安全解包与独立数值核对，不需要Torch/GPU。"""
from pathlib import Path
import json,hashlib,tarfile,csv
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with tarfile.open(ROOT/'fr2_xyz_same_k_evidence.tar.gz','r:gz')as tar:
    for member in tar.getmembers():assert member.isfile()and(OUT/member.name).resolve().is_relative_to(OUT.resolve()),member.name
    tar.extractall(OUT,filter='data')
manifest=read(OUT/'bundle_manifest.json')
for name,value in manifest['files'].items():
    path=OUT/name;assert path.stat().st_size==value['bytes']and sha(path)==value['sha256'],name
p=read(OUT/'preregistration.json');pop=read(OUT/'population.json');r=np.load(OUT/'arm_rankings.npz');ids=r['row_ids'];N=len(ids)
assert N==pop['N_candidate']
for a in p['arms']:
    order=r[a+'_order'];np.testing.assert_array_equal(np.sort(order),np.arange(N));previous=set()
    for ratio in p['ratios']:
        K=int(np.floor(ratio*N+.5));selected=set(ids[order[:K]]);assert len(selected)==K and previous<=selected;previous=selected
per=list(csv.DictReader(open(OUT/'quality_per_view.csv')));agg=list(csv.DictReader(open(OUT/'quality_summary.csv')))
frames=read(OUT/'input_manifest.json')['evaluation_frames'];bank=read(OUT/'input_manifest.json')['bank_frames'];assert not(set(frames)&set(bank))
assert len(per)==16*len(frames)and len(agg)==16
maxerror=0.
for row in agg:
    group=[v for v in per if v['arm']==row['arm']and float(v['ratio'])==float(row['ratio'])]
    assert len(group)==len(frames)and set(int(v['frame_index'])for v in group)==set(frames)
    for metric in ['psnr','ssim','lpips','photo']:
        values=np.array([float(v[metric])for v in group]);assert np.isfinite(values).all();err=abs(values.mean()-float(row[metric]));maxerror=max(maxerror,err);assert err<1e-12
q={(a['arm'],float(a['ratio'])):float(a['psnr'])for a in agg};res=read(OUT/'mechanism_result.json')
fc=np.array([q['F3_ProgressiveMarginal',ratio]-q['C_Hierarchical',ratio]for ratio in p['ratios']]);np.testing.assert_allclose(fc,res['F3_delta_PSNR_vs_C'],rtol=0,atol=1e-12)
red={(a['arm'],float(a['ratio'])):float(a['mean_redundancy'])for a in csv.DictReader(open(OUT/'selected_redundancy.csv'))}
highred=all(red['F3_ProgressiveMarginal',ratio]<red['C_Hierarchical',ratio]for ratio in [.1,.15])
combined=bool(fc.mean()>=0 and fc[0]>0 and highred)
assert res['F3_combined_supported']==combined
ca=np.array([q['C_Hierarchical',ratio]-q['A_Raw',ratio]for ratio in p['ratios']])
cgate=bool((ca>=-.02).sum()>=3 and(ca[:2].mean()>.10 or(ca[:2].max()>.20 and ca[:2].min()>=-.02)))
assert res['C_primary_PSNR_gate']==cgate
assert read(OUT/'delivery_checks.json')['all_passed']and read(OUT/'independent_replay_checks.json')['all_passed']
result=dict(all_passed=True,files_verified=len(manifest['files']),bundle_sha256=sha(ROOT/'fr2_xyz_same_k_evidence.tar.gz'),candidate_N=N,same_K=True,nested=True,complete_rankings=True,quality_view_renders=len(per),aggregate_max_abs_error=maxerror,mechanism_gate_recomputed=True,F3_combined_supported_recomputed=combined,C_primary_PSNR_gate_recomputed=cgate)
(OUT/'local_delivery_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

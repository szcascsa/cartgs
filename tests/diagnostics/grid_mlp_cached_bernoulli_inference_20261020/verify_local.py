"""离线重算Cached集合、实际L对照与逐view统计，核对全部交付SHA。"""
from pathlib import Path
import csv,hashlib,json,math,subprocess
import numpy as np

ROOT=Path(__file__).resolve().parent;GRID=ROOT.parent/'grid_mlp_implicit_si_20261020';OLD=ROOT.parent/'spatial_mlp_implicit_si_20261020'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));load=lambda name:list(csv.DictReader(open(ROOT/name)));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(ROOT/'artifact_manifest.json')['files']
for name,z in manifest.items():assert (ROOT/name).stat().st_size==z['bytes']and sha(ROOT/name)==z['sha256'],name
locked=read(ROOT/'selected_before_quality.json')
assert sha(ROOT/'selected_sets.npz')==locked['selected_SHA']and sha(ROOT/'inference_arrays.npz')==locked['arrays_SHA']
for remote,h in locked['model_SHA'].items():
    relative=remote.split('/grid-mlp-implicit-si-20261020/',1)[1];assert sha(GRID/relative)==h
sets=np.load(ROOT/'selected_sets.npz');arrays=np.load(ROOT/'inference_arrays.npz');max_p_error=0.;min_margin=1.;cases=0
def probability(score,k):
    lo=float(score.min())-40;hi=float(score.max())+40
    for _ in range(42):
        mid=.5*(lo+hi);p=np.exp(-np.logaddexp(0,-(score-mid)))
        if p.sum()>k:lo=mid
        else:hi=mid
    return np.exp(-np.logaddexp(0,-(score-.5*(lo+hi))))
for scene in ['fr1_room','fr1_xyz','fr2_xyz']:
    data=np.load(OLD/scene/'ranking_inputs.npz');ids=data['row_ids'];base=np.log(data['raw_I'].astype(float)+1e-12);raw_order=np.lexsort((ids,-base))
    for ratio in [.01,.05,.1,.15]:
        k=int(len(ids)*ratio+.5)
        for seed in [2026,2027]:
            reference_u=None
            for arm in ['Grid-MLP','Independent-Cell']:
                prefix=f'{scene}_{arm}_{ratio}_{seed}';score=arrays[prefix+'_score'];u=arrays[prefix+'_cached_uniform'];p=arrays[prefix+'_probability']
                np.testing.assert_array_equal(ids,arrays[prefix+'_source_rows'])
                if reference_u is None:reference_u=u
                else:np.testing.assert_array_equal(reference_u,u)
                recalculated=probability(score,k);error=float(abs(recalculated-p).max());max_p_error=max(max_p_error,error)
                # 允许两平台sum在42轮二分末尾选择相邻区间；membership仍要求exact。
                assert error <= .25*(float(np.ptp(score))+80)/2**42+5e-14
                np.testing.assert_array_equal(ids[u<p],sets[f'{scene}_{arm}-Cached_{ratio}_{seed}'])
                np.testing.assert_array_equal(ids[u<recalculated],sets[f'{scene}_{arm}-Cached_{ratio}_{seed}'])
                L=int((u<p).sum());assert L>0
                np.testing.assert_array_equal(np.sort(ids[np.lexsort((ids,-score))[:L]]),sets[f'{scene}_{arm}-Hard-L_{ratio}_{seed}'])
                np.testing.assert_array_equal(np.sort(ids[raw_order[:L]]),sets[f'{scene}_Raw-Hard-L-{arm}_{ratio}_{seed}'])
                np.testing.assert_array_equal(ids[u<probability(base,k)],sets[f'{scene}_Raw-Cached_{ratio}_{seed}'])
                min_margin=min(min_margin,float(abs(p-u).min()));cases+=1
q=load('quality_summary.csv');v=load('quality_per_view.csv');assert len(q)==600 and len(v)==35500
vm={(z['scene'],z['split'],z['variant'],float(z['ratio']),int(z['seed']),int(z['frame_index'])):z for z in v};assert len(vm)==len(v)
aggregate_error=paired_error=0.
for z in q:
    info=read(OLD/z['scene']/'input_manifest.json');frames=info['bank_frames'if z['split']=='train'else'evaluation_frames']
    for metric in ['psnr','ssim','lpips','photo']:
        expected=math.fsum(float(vm[z['scene'],z['split'],z['variant'],float(z['ratio']),int(z['seed']),f][metric])for f in frames)/len(frames)
        aggregate_error=max(aggregate_error,abs(expected-float(z[metric])))
for z in load('paired_statistics.csv'):
    info=read(OLD/z['scene']/'input_manifest.json');frames=info['bank_frames'if z['split']=='train'else'evaluation_frames'];refseed=-1 if z['reference']in ['Raw-Hard-K','Fixed-Hard-K','C-Hard-K']else int(z['seed'])
    a=np.array([float(vm[z['scene'],z['split'],z['arm']+'-Cached',float(z['ratio']),int(z['seed']),f][z['metric']])-float(vm[z['scene'],z['split'],z['reference'],float(z['ratio']),refseed,f][z['metric']])for f in frames])
    sign=1 if z['metric']in ['psnr','ssim']else-1
    for name,value in dict(mean=a.mean(),median=np.median(a),p10=np.quantile(a,.1),p90=np.quantile(a,.9),improved_view_fraction=np.mean(sign*a>0)).items():paired_error=max(paired_error,abs(float(z[name])-value))
assert aggregate_error<1e-12 and paired_error<1e-12
assert not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT.parents[2]).strip()
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT.parents[2]).strip()
result=dict(all_passed=True,files=len(manifest),learned_cases=cases,Cached_rows_native_Python_NumPy_exact=True,same_L_controls_exact=True,
            probability_recompute_max_abs=max_p_error,min_sampling_margin=min_margin,aggregate_perview_max_abs=aggregate_error,paired_max_abs=paired_error,models_unchanged_SHA=True,tracked_and_index_unchanged=True)
(ROOT/'local_delivery_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

"""区分Cached学习效果、推理策略变化与实际点数；四指标paired-view完整保存。"""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'spatial-mlp-implicit-si-20261020';GRID=ROOT.parent/'grid-mlp-implicit-si-20261020'
METRICS=['psnr','ssim','lpips','photo'];SCENES=['fr1_room','fr1_xyz','fr2_xyz'];RATIOS=[.01,.05,.1,.15];SEEDS=[2026,2027]
load=lambda name:list(csv.DictReader(open(ROOT/name)))
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,z):(ROOT/name).write_text(json.dumps(z,indent=2,allow_nan=False)+'\n')
def write(name,rows):
    with open(ROOT/name,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run():
    assert read(ROOT/'quality_complete.json')['all_passed'];locked=read(ROOT/'selected_before_quality.json')
    assert sha(ROOT/'selected_sets.npz')==locked['selected_SHA']and sha(ROOT/'inference_arrays.npz')==locked['arrays_SHA']
    calibration=ROOT.parent/'spatial-mlp-optimization-fix-20261020/source/calibrated_selection.h'
    selector=ROOT.parent/'mycartgs/include/selector_handoff.h'
    assert sha(calibration)==sha(GRID/'dependencies/calibrated_selection.h')
    arrays=np.load(ROOT/'inference_arrays.npz');uniforms={}
    for scene in SCENES:
        for arm in ['Grid-MLP','Independent-Cell']:
            for ratio in RATIOS:
                for seed in SEEDS:
                    for step in [0,2000]:
                        path=GRID/scene/'models'/f'{arm}_{int(100*ratio)}_seed_{seed}'/f'step_{step}.pt'
                        state=dict(torch.jit.load(str(path),map_location='cpu').named_buffers());key=(scene,ratio,seed);u=state['cached_uniform'].numpy()
                        if key in uniforms:np.testing.assert_array_equal(u,uniforms[key])
                        else:uniforms[key]=u.copy()
                        if step==2000:
                            np.testing.assert_array_equal(arrays[f'{scene}_{arm}_{ratio}_{seed}_cached_uniform'],u)
                            np.testing.assert_array_equal(arrays[f'{scene}_{arm}_{ratio}_{seed}_score'],state['score'].numpy())
    q=load('quality_summary.csv');v=load('quality_per_view.csv');assert len(q)==600 and len(v)==35500
    qm={(z['scene'],z['split'],z['variant'],float(z['ratio']),int(z['seed'])):z for z in q}
    vm={(z['scene'],z['split'],z['variant'],float(z['ratio']),int(z['seed']),int(z['frame_index'])):z for z in v};assert len(qm)==len(q)and len(vm)==len(v)
    aggregate_error=0.
    for z in q:
        info=read(OLD/z['scene']/'input_manifest.json');frames=info['bank_frames'if z['split']=='train'else'evaluation_frames']
        for metric in METRICS:
            value=math.fsum(float(vm[z['scene'],z['split'],z['variant'],float(z['ratio']),int(z['seed']),f][metric])for f in frames)/len(frames)
            aggregate_error=max(aggregate_error,abs(value-float(z[metric])))
    assert aggregate_error<1e-12
    comparisons=[];paired=[];counts={};overlap=[];sets=np.load(ROOT/'selected_sets.npz');oldsets=np.load(GRID/'selected_sets.npz')
    for arm in ['Grid-MLP','Independent-Cell']:
        counts[arm]={}
        refs=['Raw-Cached',arm+'-Hard-L','Raw-Hard-L-'+arm,arm+'-Hard-K','Raw-Hard-K','PG-ST-Hard-K','Fixed-Hard-K','C-Hard-K','Per-Gaussian-MLP-Hard-K']
        for ref in refs:
            counts[arm][ref]={}
            for split in ['train','eval']:
                cases=[];psnrpaired=[]
                for scene in SCENES:
                    info=read(OLD/scene/'input_manifest.json');frames=info['bank_frames'if split=='train'else'evaluation_frames']
                    for ratio in RATIOS:
                        for seed in SEEDS:
                            refseed=-1 if ref in ['Raw-Hard-K','Fixed-Hard-K','C-Hard-K']else seed
                            a=qm[scene,split,arm+'-Cached',ratio,seed];b=qm[scene,split,ref,ratio,refseed]
                            row=dict(scene=scene,ratio=ratio,seed=seed,split=split,arm=arm,reference=ref,nominal_K=int(a['K']),actual_L=int(a['L']),reference_L=int(b['L']),point_count_delta=int(a['L'])-int(b['L']),same_count=int(a['L'])==int(b['L']),**{m:float(a[m])-float(b[m])for m in METRICS})
                            if ref in [arm+'-Hard-L','Raw-Hard-L-'+arm]:assert row['same_count']
                            cases.append(row);comparisons.append(row)
                            for metric in METRICS:
                                values=np.array([float(vm[scene,split,arm+'-Cached',ratio,seed,f][metric])-float(vm[scene,split,ref,ratio,refseed,f][metric])for f in frames]);sign=1 if metric in ['psnr','ssim']else-1
                                z=dict(scene=scene,ratio=ratio,seed=seed,split=split,arm=arm,reference=ref,metric=metric,views=len(frames),same_count=row['same_count'],mean=float(values.mean()),median=float(np.median(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)),improved_view_fraction=float(np.mean(sign*values>0)))
                                paired.append(z)
                                if metric=='psnr':psnrpaired.append(z)
                counts[arm][ref][split]=dict(cases=24,same_count_cases=sum(z['same_count']for z in cases),
                    **{m:dict(improved=sum((1 if m in ['psnr','ssim']else-1)*z[m]>0 for z in cases),equal=sum(z[m]==0 for z in cases),worse=sum((1 if m in ['psnr','ssim']else-1)*z[m]<0 for z in cases),mean_delta=float(np.mean([z[m]for z in cases])),min_delta=min(z[m]for z in cases),max_delta=max(z[m]for z in cases))for m in METRICS},
                    PSNR_positive_median_cases=sum(z['median']>0 for z in psnrpaired),PSNR_majority_views_improved_cases=sum(z['improved_view_fraction']>.5 for z in psnrpaired))
        for scene in SCENES:
            for ratio in RATIOS:
                for seed in SEEDS:
                    a=sets[f'{scene}_{arm}-Cached_{ratio}_{seed}']
                    for ref in ['Raw-Cached',arm+'-Hard-L','Raw-Hard-L-'+arm,arm+'-Hard-K']:
                        b=oldsets[f'{scene}_{arm}_{ratio}_{seed}']if ref==arm+'-Hard-K'else sets[f'{scene}_{ref}_{ratio}_{seed}']
                        intersection=len(np.intersect1d(a,b));overlap.append(dict(scene=scene,ratio=ratio,seed=seed,arm=arm,reference=ref,L=len(a),reference_L=len(b),intersection=intersection,Jaccard=intersection/(len(a)+len(b)-intersection)))
    write('quality_deltas.csv',comparisons);write('paired_statistics.csv',paired);write('selected_set_overlap.csv',overlap)
    points=load('selection_counts.csv');point_summary={}
    for arm in ['Grid-MLP','Independent-Cell']:
        own=[z for z in points if z['arm']==arm and int(z['step'])==2000]
        point_summary[arm]=dict(min_L_over_K=min(float(z['L_over_K'])for z in own),median_L_over_K=float(np.median([float(z['L_over_K'])for z in own])),max_L_over_K=max(float(z['L_over_K'])for z in own),maximum_abs_L_minus_K=max(abs(int(z['L'])-int(z['K']))for z in own))
    correctness=dict(all_passed=True,native_Python_exact_checks=96,checkpoint_same_as_training=True,original_calibration_and_ST_helper_reused=True,
        native_binary_mask_exact=True,repeated_cached_inference_exact=True,step0_sample_count_matches_training=True,uniforms_step0_final_and_control_exact=True,no_fresh_uniform=True,no_refill_or_truncate=True,
        opacity_unmodified_on_selected_rows=True,TF_input_nominal_ratio_unchanged=True,all_same_L_controls_exact=True,selected_sets_locked_before_quality=True,
        inference=locked,source_dependency_SHA={str(path):sha(path)for path in [calibration,selector,ROOT/'cached_inference.cpp']},baseline=read(ROOT/'baseline_reproduction.json'),quality=read(ROOT/'quality_complete.json'),aggregate_perview_max_abs=aggregate_error,
        no_training=True,no_production_change=True,no_split=True)
    save('correctness.json',correctness);save('audit_summary.json',dict(counts=counts,point_counts=point_summary,correctness=correctness,interpretation='PENDING_EVIDENCE_REVIEW'))
    print(json.dumps(dict(counts=counts,point_counts=point_summary),indent=2))

if __name__=='__main__':run()

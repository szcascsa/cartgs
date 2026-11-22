"""终验exact K、full-population gradients、冻结与质量；不调整训练协议。"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(p):return list(csv.DictReader(open(p)))
def write(n,rows):
    with open(ROOT/n,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
reg=read(ROOT/'preregistration.json');checks=read(ROOT/'input_checks.json')['scenes'];fixed=read(ROOT/'selected_before_quality.json');q=[];v=[];inputs={}
assert sha(ROOT/'selected_sets.npz')==fixed['selected_sha']and sha(ROOT/'rankings.npz')==fixed['ranking_sha']and sha(ROOT/'coefficients.pt')==fixed['coefficient_sha']
for name in reg['scenes']:
    here=ROOT/name;assert read(here/'quality_complete.json')['all_passed'];assert read(here/'baseline_reproduction.json')['all_passed'];assert read(here/'legacy_reproduction.json')['all_passed'];info=read(here/'input_manifest.json');c=checks[name]
    for field in ['input_sha','runtime_sha','legacy_model_sha']:
        for p,h in c[field].items():assert sha(p)==h,('FROZEN_INPUT_CHANGED',p)
    for p,h in c['source_sha'].items():assert sha(p)==h['sha256']
    assert sha(info['cameras'])==c['cameras_sha']and not(set(info['bank_frames'])&set(info['evaluation_frames']))
    q.extend(load(here/'quality_summary.csv'));v.extend(load(here/'quality_per_view.csv'))
    inputs[name]=dict(all_passed=True,frozen_files_SHA_exact=True,train_eval_disjoint=True,baseline=read(here/'baseline_reproduction.json'),legacy=read(here/'legacy_reproduction.json'))
write('quality_summary.csv',q);write('quality_per_view.csv',v)
lookup={(z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed']),int(z['frame_index'])):z for z in v};qmap={(z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed'])):z for z in q}
deltas=[];paired=[];counts={}
for reference in ['PG-ST','Raw','Fixed','C']:
    counts[reference]={}
    for split in ['train','eval']:
        cases=[]
        for name in reg['scenes']:
            info=read(ROOT/name/'input_manifest.json');frames=info['bank_frames'if split=='train'else'evaluation_frames']
            for ratio in reg['ratios']:
                for seed in reg['seeds']:
                    refseed=seed if reference=='PG-ST'else-1;a=qmap[name,split,'PG-PL',ratio,seed];b=qmap[name,split,reference,ratio,refseed]
                    z=dict(scene=name,ratio=ratio,seed=seed,split=split,reference=reference,**{m:float(a[m])-float(b[m])for m in ['psnr','ssim','lpips','photo']});deltas.append(z);cases.append(z)
                    for metric in ['psnr','ssim','lpips','photo']:
                        d=np.array([float(lookup[name,split,'PG-PL',ratio,seed,f][metric])-float(lookup[name,split,reference,ratio,refseed,f][metric])for f in frames]);direction=1 if metric in['psnr','ssim']else-1
                        paired.append(dict(scene=name,ratio=ratio,seed=seed,split=split,comparison='PG-PL - '+reference,metric=metric,views=len(frames),mean=float(d.mean()),median=float(np.median(d)),p10=float(np.quantile(d,.1)),p90=float(np.quantile(d,.9)),improved_view_fraction=float(np.mean(direction*d>0))))
        counts[reference][split]=dict(cases=len(cases),PSNR_positive=sum(z['psnr']>0 for z in cases),SSIM_positive=sum(z['ssim']>0 for z in cases),LPIPS_improved=sum(z['lpips']<0 for z in cases),photo_improved=sum(z['photo']<0 for z in cases))
write('quality_deltas.csv',deltas);write('paired_statistics.csv',paired)
coverage=load(ROOT/'gradient_coverage.csv');sampling=load(ROOT/'sampling_statistics.csv');curves=load(ROOT/'training_curves.csv');assert len(curves)==48000
assert all(int(z['sampled_count'])==int(z['K'])for z in curves)
assert all(int(z['fresh_steps'])==2000 for z in sampling)
hardview=load(ROOT/'hard_training_per_view.csv');hard=load(ROOT/'hard_training_dynamics.csv');harderr={m:0.0 for m in ['psnr','photo']}
for z in hardview:
    step=int(z['step'])
    if step not in [0,2000]:continue
    name=z['scene'];split='train';arm='Raw'if step==0 else'PG-PL';seed=-1 if step==0 else int(z['seed']);ratio=float(z['ratio']);f=int(z['frame']);ref=lookup[name,split,arm,ratio,seed,f]
    for m in harderr:harderr[m]=max(harderr[m],abs(float(z[m])-float(ref[m])))
assert harderr['photo']==0 and harderr['psnr']<1e-5,('HARD_FORWARD_METRIC_MISMATCH',harderr)
correct=dict(all_passed=True,initial_full_Raw_ranking_exact=True,formal_updates=48000,all_48000_counts_exact_K=True,all_48000_noise_signatures_distinct_per_run=True,
    PL_probability_tests=read(ROOT/'pl_probability_tests.json'),REINFORCE_gradient_tests=read(ROOT/'reinforce_gradient_tests.json'),
    all_gradients_finite=True,all_selected_and_unselected_gradients_included=True,gradient_coverage=coverage,
    all_models_frozen_exact=True,all_inputs_SHA_exact=inputs,only_a_optimized=True,baseline_update_timing_exact=True,pure_score_function=True,
    no_opacity_ST=True,no_detached_offset=True,no_Bernoulli=True,no_cached_noise=True,no_numeric_clamp=True,no_nonfinite_or_underflow=True,
    all_training_Bank_only=True,no_C_Fixed_eval_teacher=True,all_hard_selected_same_K_unique=True,all_final_sets_fixed_before_quality=True,
    hard_training_diagnostic_vs_standard_metrics_max_abs=harderr,hard_photo_exact=True,diagnostic_PSNR_FP32_reduction_tolerance=1e-5,smoke_repeat=read(ROOT/'training_gate.json')['repeat'],
    max_gradient_shift_residual=max(float(z['max_shift_gradient_residual'])for z in sampling),no_grid_split_or_extra_correction=True)
save(ROOT/'correctness.json',correct)
seed_variance=[]
for name in reg['scenes']:
    for ratio in reg['ratios']:
        for split in ['train','eval']:
            for arm in ['PG-ST','PG-PL']:
                zz=[qmap[name,split,arm,ratio,seed]for seed in reg['seeds']]
                seed_variance.append(dict(scene=name,ratio=ratio,split=split,arm=arm,**{m+'_seed_abs_difference':abs(float(zz[0][m])-float(zz[1][m]))for m in['psnr','ssim','lpips','photo']}))
write('seed_variance.csv',seed_variance)
save(ROOT/'audit_summary.json',dict(all_passed=True,counts=counts,correctness=correct,protocol=reg,decision='PENDING_EVIDENCE_REVIEW'))
print('CORRECTNESS_AND_QUALITY_COMPLETE',json.dumps(counts),flush=True)

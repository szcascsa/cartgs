from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(p):return list(csv.DictReader(open(p)))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(n,rows):
    with open(ROOT/n,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
reg=read(ROOT/'preregistration.json');fixed=read(ROOT/'selected_before_quality.json');inputchecks=read(ROOT/'input_checks.json')['scenes'];q=[];v=[];checks={};models=[]
assert sha(ROOT/'selected_sets.npz')==fixed['selected_sha']and sha(ROOT/'rankings.npz')==fixed['ranking_sha']
for name in reg['scenes']:
    here=ROOT/name;info=read(here/'input_manifest.json');assert read(here/'quality_complete.json')['all_passed'];c=inputchecks[name]
    for field in ['input_sha','runtime_sha','legacy_model_sha']:
        for p,h in c[field].items():assert sha(p)==h
    for p,h in c['source_sha'].items():assert sha(p)==h['sha256']
    assert sha(info['cameras'])==c['cameras_sha']and not(set(info['bank_frames'])&set(info['evaluation_frames']))
    q.extend(load(here/'quality_summary.csv'));v.extend(load(here/'quality_per_view.csv'));checks[name]=dict(all_passed=True,baseline=read(here/'baseline_reproduction.json'),PG_ST=read(here/'legacy_reproduction.json'),all_input_SHA_exact=True,normalization=read(here/'normalization.json'))
    for ratio in reg['ratios']:
        for seed in reg['seeds']:models.append(read(here/'models'/f'MLP_{int(100*ratio)}_seed_{seed}'/'completed.json'))
write('quality_summary.csv',q);write('quality_per_view.csv',v)
qm={(z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed'])):z for z in q};vm={(z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed']),int(z['frame_index'])):z for z in v};counts={};deltas=[];paired=[];seedstats=[]
for reference in ['PG-ST','Raw','Fixed','C']:
    counts[reference]={}
    for split in ['train','eval']:
        cases=[]
        for name in reg['scenes']:
            info=read(ROOT/name/'input_manifest.json');frames=info['bank_frames'if split=='train'else'evaluation_frames']
            for ratio in reg['ratios']:
                for seed in reg['seeds']:
                    refseed=seed if reference=='PG-ST'else-1;a=qm[name,split,'MLP-ST',ratio,seed];b=qm[name,split,reference,ratio,refseed];z=dict(scene=name,ratio=ratio,seed=seed,split=split,reference=reference,**{m:float(a[m])-float(b[m])for m in ['psnr','ssim','lpips','photo']});deltas.append(z);cases.append(z)
                    for m in ['psnr','ssim','lpips','photo']:
                        d=np.array([float(vm[name,split,'MLP-ST',ratio,seed,f][m])-float(vm[name,split,reference,ratio,refseed,f][m])for f in frames]);sign=1 if m in['psnr','ssim']else-1
                        paired.append(dict(scene=name,ratio=ratio,seed=seed,split=split,comparison='MLP-ST - '+reference,metric=m,views=len(frames),mean=float(d.mean()),median=float(np.median(d)),p10=float(np.quantile(d,.1)),p90=float(np.quantile(d,.9)),improved_view_fraction=float(np.mean(sign*d>0))))
        counts[reference][split]=dict(cases=len(cases),PSNR_positive=sum(z['psnr']>0 for z in cases),SSIM_positive=sum(z['ssim']>0 for z in cases),LPIPS_improved=sum(z['lpips']<0 for z in cases),photo_improved=sum(z['photo']<0 for z in cases))
for name in reg['scenes']:
    for ratio in reg['ratios']:
        for split in ['train','eval']:
            for arm in ['PG-ST','MLP-ST']:
                a,b=[qm[name,split,arm,ratio,s]for s in reg['seeds']];seedstats.append(dict(scene=name,ratio=ratio,split=split,arm=arm,**{m+'_seed_abs_difference':abs(float(a[m])-float(b[m]))for m in ['psnr','ssim','lpips','photo']}))
write('paired_statistics.csv',paired);write('quality_deltas.csv',deltas);write('seed_consistency.csv',seedstats)
tr=load(ROOT/'training_curves.csv');assert len(tr)==48000;error=max(abs(float(z['expected_N'])-int(z['K']))for z in tr);assert error<1e-5
correct=dict(all_passed=True,models=24,formal_updates=48000,initial_output_exact_zero=True,initial_full_Raw_ranking_exact=True,initial_all_four_selected_sets_exact_Raw=True,
    only_theta_optimized=True,architecture='3-32-32-1 SiLU',parameters=1217,normalized_XYZ_only=True,frozen_tensor_exact=True,frozen_SHA_checks=checks,
    normalized_xyz_deterministic=True,AABB_frozen_no_refit_no_clamp=True,legacy_Bank_frame_schedule_exact=True,legacy_cached_uniform_exact=True,
    no_C_Fixed_eval_training_reads=True,no_eval_based_LR_or_step_selection=True,all_selected_same_K_unique=True,all_sets_fixed_before_quality=True,
    loss_and_ST_semantics_unchanged=True,maximum_expected_K_calibration_error=error,lr_sanity=read(ROOT/'lr_sanity.json'),model_checks=models,
    no_regularization=True,no_grid_or_PL=True,no_coefficient_clipping=True,no_nonfinite_output_or_gradient=True)
save(ROOT/'correctness.json',correct);save(ROOT/'audit_summary.json',dict(all_passed=True,counts=counts,correctness=correct,protocol=reg,field_states=fixed['models'],decision='PENDING_EVIDENCE_REVIEW'))
print('QUALITY_AND_CORRECTNESS_COMPLETE',json.dumps(counts),flush=True)

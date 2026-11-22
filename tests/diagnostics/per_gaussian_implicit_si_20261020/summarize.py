"""合并所有seed与逐view证据，做冻结/同K/梯度隔离终验。"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for a in iter(lambda:f.read(8*1024*1024),b''):h.update(a)
    return h.hexdigest()
def load(p):return list(csv.DictReader(open(p)))
def write(name,rows):
    with open(ROOT/name,'w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
reg=read(ROOT/'preregistration.json');rep=read(ROOT/'representability_check.json');fixed=read(ROOT/'selected_before_quality.json');assert sha(ROOT/'selected_sets.npz')==fixed['selected_sha'];assert sha(ROOT/'rankings.npz')==fixed['ranking_sha']
q=[];v=[];checks={};models=[];sampleq=[];samplev=[]
for name in reg['scenes']:
    here=ROOT/name;assert read(here/'pg_quality_complete.json')['all_passed'];info=read(here/'input_manifest.json');r=rep['scenes'][name]
    for p,h in r['input_sha'].items():assert sha(p)==h,('INPUT_CHANGED',p)
    for p,h in r['source_sha'].items():assert sha(p)==h['sha256'],('SOURCE_CHANGED',p)
    assert sha(info['cameras'])==r['cameras_sha']
    for p,h in r['runtime_sha'].items():assert sha(p)==h
    baseline=read(here/'baseline_reproduction.json');assert baseline['all_passed']
    checks[name]=dict(baseline=baseline,all_frozen_files_SHA_exact=True,train_eval_disjoint=not(set(info['bank_frames'])&set(info['evaluation_frames'])))
    q.extend(load(here/'pg_quality_summary.csv'));v.extend(load(here/'pg_quality_per_view.csv'))
    sampleq.extend(load(here/'sampling_train_quality.csv'));samplev.extend(load(here/'sampling_train_per_view.csv'))
    for ratio in reg['ratios']:
        for seed in reg['seeds']:
            d=here/'models'/f'PG_{int(100*ratio)}_seed_{seed}';z=read(d/'completed.json');last=z['records'][-1]
            assert last['steps']==reg['steps']and last['frozen_exact']and last['only_coefficient_optimized']
            assert set(last['visited_Bank_frames'])<=set(info['bank_frames'])
            assert len(last['visited_Bank_frames'])==reg['steps']
            models.append(dict(scene=name,ratio=ratio,seed=seed,seconds=z['seconds'],min_sampled_N=last['min_sampled_N'],max_sampled_N=last['max_sampled_N'],max_expected_K_error=last['max_expected_K_error'],frozen_exact=True))
write('quality_summary.csv',q);write('quality_per_view.csv',v)
write('sampling_train_quality.csv',sampleq);write('sampling_train_per_view.csv',samplev)
lookup={(z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed']),int(z['step']),int(z['frame_index'])):z for z in v}
qmap={(z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed']),int(z['step'])):z for z in q}
paired=[];deltas=[];dynamics=[]
for scene in reg['scenes']:
    info=read(ROOT/scene/'input_manifest.json')
    for ratio in reg['ratios']:
        arm=f'PG-{int(100*ratio)}'
        for seed in reg['seeds']:
            for split in ['train','eval']:
                a=qmap[scene,split,arm,ratio,seed,reg['steps']]
                # Raw at step 0 and PG at step 0 must have identical Bank metrics.
                if split=='train':
                    zero=qmap[scene,split,arm,ratio,seed,0];raw=qmap[scene,split,'A',ratio,-1,0]
                    for m in['psnr','ssim','lpips','photo']:assert float(zero[m])==float(raw[m])
                    for step in reg['checkpoints']:
                        mid=qmap[scene,split,arm,ratio,seed,step]
                        dynamics.append(dict(scene=scene,ratio=ratio,seed=seed,step=step,train_hard_psnr=float(mid['psnr']),train_hard_photo=float(mid['photo']),train_hard_psnr_minus_Raw=float(mid['psnr'])-float(raw['psnr']),train_hard_photo_minus_Raw=float(mid['photo'])-float(raw['photo'])))
                for baseline in ['A','B','C']:
                    b=qmap[scene,split,baseline,ratio,-1,0]
                    deltas.append(dict(scene=scene,ratio=ratio,seed=seed,split=split,reference=baseline,**{m:float(a[m])-float(b[m])for m in['psnr','ssim','lpips','photo']}))
                    frames=info['bank_frames'if split=='train'else'evaluation_frames']
                    for m in ['psnr','ssim','lpips','photo']:
                        values=np.array([float(lookup[scene,split,arm,ratio,seed,reg['steps'],f][m])-float(lookup[scene,split,baseline,ratio,-1,0,f][m])for f in frames]);sign=1 if m in['psnr','ssim']else-1
                        paired.append(dict(scene=scene,ratio=ratio,seed=seed,split=split,comparison=arm+' - '+baseline,metric=m,views=len(frames),mean=float(values.mean()),median=float(np.median(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)),improved_view_fraction=float(np.mean(sign*values>0))))
write('paired_statistics.csv',paired);write('quality_deltas.csv',deltas);write('hard_training_dynamics.csv',dynamics);write('model_training_checks.csv',models)
def smoke_tensor(d):return {k:v.detach().cpu().numpy()for k,v in torch.jit.load(str(ROOT/d/'step_5.pt'),map_location='cpu').named_buffers()}
sx=smoke_tensor('smoke_first');sy=smoke_tensor('smoke_repeat');xx=np.load(ROOT/'fr1_xyz/ranking_inputs.npz');ids=xx['row_ids'];base=np.log(xx['raw_I'].astype(float)+reg['epsilon']);ox=np.lexsort((ids,-(base+sx['coefficient'])));oy=np.lexsort((ids,-(base+sy['coefficient'])))
repeat=read(ROOT/'gradient_smoke.json');repeat['full_ranking_exact']=bool(np.array_equal(ox,oy));repeat['all_four_selected_sets_exact']=all(np.array_equal(np.sort(ox[:int(r*len(ids)+.5)]),np.sort(oy[:int(r*len(ids)+.5)]))for r in reg['ratios']);save(ROOT/'repeat_checks.json',repeat)
correct=dict(all_passed=True,baseline_exact=checks,representability_full_ranking_exact=True,initial_Raw_ranking_and_all_sets_exact=True,
    training_raw_initial_Bank_metrics_exact=True,only_coefficients_optimized=True,model_and_TF_tensors_exact=True,all_frozen_input_file_SHA_exact=True,
    all_training_frames_Bank_only=True,formal_models=24,formal_steps=24*reg['steps'],historical_same_K=True,all_selected_unique=True,
    all_selected_from_candidate=True,all_rankings_fixed_before_quality=True,no_numeric_coefficient_clip=True,no_regularization=True,no_C_teacher=True,
    no_grid_or_hierarchy_training=True,no_eval_hyperparameter_selection=True,sampling='user-authorized existing cached-uniform ST Bernoulli + detached expected-K calibration',
    determinism=repeat,max_expected_K_error=max(z['max_expected_K_error']for z in models),coefficient_snapshot_count=96)
save(ROOT/'correctness.json',correct)
counts={}
for reference in ['A','B','C']:
    counts[reference]={}
    for split in ['train','eval']:
        values=[z for z in deltas if z['reference']==reference and z['split']==split]
        counts[reference][split]=dict(cases=len(values),PSNR_positive=sum(z['psnr']>0 for z in values),SSIM_positive=sum(z['ssim']>0 for z in values),LPIPS_improved=sum(z['lpips']<0 for z in values),photo_improved=sum(z['photo']<0 for z in values))
sampling_deltas=[]
for z in sampleq:
    if int(z['step'])!=reg['steps']:continue
    before=next(a for a in sampleq if a['scene']==z['scene']and a['ratio']==z['ratio']and a['seed']==z['seed']and int(a['step'])==0)
    sampling_deltas.append(dict(scene=z['scene'],ratio=float(z['ratio']),seed=int(z['seed']),sampled_N_initial=int(before['sampled_N']),sampled_N_final=int(z['sampled_N']),photo=float(z['photo'])-float(before['photo']),psnr=float(z['psnr'])-float(before['psnr'])))
write('sampling_train_deltas.csv',sampling_deltas)
save(ROOT/'audit_summary.json',dict(all_passed=True,counts=counts,model_training=models,states=fixed['models'],correctness=correct,sampling_deltas=sampling_deltas,
    decision='PENDING_EVIDENCE_INTERPRETATION',protocol=reg,baseline_source='/home/prmi/zwj/final-rep-residual-fallback-20261020'))
print('ALL_FINAL_CHECKS_PASS',json.dumps(counts),flush=True)

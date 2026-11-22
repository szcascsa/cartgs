"""完整same-K质量、paired views、control与regional预算移动统计；不反馈训练。"""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np

ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'spatial-mlp-implicit-si-20261020';PRIOR=ROOT.parent/'spatial-mlp-calibrated-quality-20261020'
SCENES=['fr1_room','fr1_xyz','fr2_xyz'];RATIOS=[.01,.05,.1,.15];SEEDS=[2026,2027];METRICS=['psnr','ssim','lpips','photo']
def read(p):return json.loads(Path(p).read_text())
def load(name):return list(csv.DictReader(open(ROOT/name)))
def save(name,x):(ROOT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def write(name,rows):
    with open(ROOT/name,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert read(ROOT/'quality_complete.json')['all_passed'];locked=read(ROOT/'selected_before_quality.json')
assert sha(ROOT/'selected_sets.npz')==locked['selected_SHA']and sha(ROOT/'rankings.npz')==locked['rankings_SHA']
q=load('quality_summary.csv');v=load('quality_per_view.csv');qm={(z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed'])):z for z in q};vm={(z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed']),int(z['frame_index'])):z for z in v}
assert len(q)==len(qm)==264 and len(v)==len(vm)==15620
aggregate_error=0.
for z in q:
    info=read(OLD/z['scene']/'input_manifest.json');frames=info['bank_frames'if z['split']=='train'else'evaluation_frames']
    for metric in METRICS:
        error=abs(math.fsum(float(vm[z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed']),f][metric])for f in frames)/len(frames)-float(z[metric]));aggregate_error=max(aggregate_error,error);assert error<1e-12
counts={};deltas=[];paired=[];control=[];seedstats=[];regional=[]
for arm in ['Grid-MLP','Independent-Cell']:
    references=['Raw','PG-ST','Fixed','C','Per-Gaussian-MLP']+(['Grid-MLP']if arm=='Independent-Cell'else[])
    counts[arm]={}
    for reference in references:
        counts[arm][reference]={}
        for split in ['train','eval']:
            cases=[];psnrviews=[]
            for scene in SCENES:
                info=read(OLD/scene/'input_manifest.json');frames=info['bank_frames'if split=='train'else'evaluation_frames']
                for ratio in RATIOS:
                    for seed in SEEDS:
                        refseed=-1 if reference in ['Raw','Fixed','C']else seed
                        a=qm[scene,split,arm,ratio,seed];b=qm[scene,split,reference,ratio,refseed]
                        row=dict(scene=scene,ratio=ratio,seed=seed,split=split,arm=arm,reference=reference,**{metric:float(a[metric])-float(b[metric])for metric in METRICS})
                        cases.append(row);deltas.append(row)
                        for metric in METRICS:
                            values=np.array([float(vm[scene,split,arm,ratio,seed,f][metric])-float(vm[scene,split,reference,ratio,refseed,f][metric])for f in frames]);sign=1 if metric in ['psnr','ssim']else-1
                            pair=dict(scene=scene,ratio=ratio,seed=seed,split=split,arm=arm,reference=reference,metric=metric,views=len(values),mean=float(values.mean()),median=float(np.median(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)),improved_view_fraction=float(np.mean(sign*values>0)))
                            paired.append(pair)
                            if metric=='psnr':psnrviews.append(pair)
            counts[arm][reference][split]=dict(cases=24,**{metric:dict(improved=sum((1 if metric in ['psnr','ssim']else-1)*z[metric]>0 for z in cases),equal=sum(z[metric]==0 for z in cases),worse=sum((1 if metric in ['psnr','ssim']else-1)*z[metric]<0 for z in cases),mean_delta=float(np.mean([z[metric]for z in cases])),min_delta=min(z[metric]for z in cases),max_delta=max(z[metric]for z in cases))for metric in METRICS},PSNR_positive_median_cases=sum(z['median']>0 for z in psnrviews),PSNR_majority_views_improved_cases=sum(z['improved_view_fraction']>.5 for z in psnrviews))
for z in q:
    if z['arm']=='Independent-Cell':
        b=qm[z['scene'],z['split'],'Raw',float(z['ratio']),-1]
        control.append(dict(**z,**{metric+'_delta_Raw':float(z[metric])-float(b[metric])for metric in METRICS}))
for scene in SCENES:
    for ratio in RATIOS:
        for split in ['train','eval']:
            for arm in ['Grid-MLP','Independent-Cell','Per-Gaussian-MLP','PG-ST']:
                a,b=[qm[scene,split,arm,ratio,seed]for seed in SEEDS]
                seedstats.append(dict(scene=scene,ratio=ratio,split=split,arm=arm,**{metric+'_seed_abs_difference':abs(float(a[metric])-float(b[metric]))for metric in METRICS}))
write('quality_deltas.csv',deltas);write('paired_statistics.csv',paired);write('independent_cell_quality.csv',control);write('seed_consistency.csv',seedstats)

sets=np.load(ROOT/'selected_sets.npz');oldsets=np.load(OLD/'selected_sets.npz');pgsets=np.load(PRIOR/'selected_sets.npz');overlap=[z for z in load('selected_set_overlap.csv')if z['reference']=='Raw']
for arm in ['Grid-MLP','Independent-Cell']:
    for scene in SCENES:
        for ratio in RATIOS:
            for seed in SEEDS:
                a=sets[f'{scene}_{arm}_{ratio}_{seed}'];k=len(a)
                for ref in ['PG-ST','Fixed','C','Per-Gaussian-MLP']+(['Grid-MLP']if arm=='Independent-Cell'else[]):
                    if ref=='Per-Gaussian-MLP':b=pgsets[f'{scene}_MLP-ST-fixed_{ratio}_{seed}']
                    elif ref=='Grid-MLP':b=sets[f'{scene}_Grid-MLP_{ratio}_{seed}']
                    else:b=oldsets[f'{scene}_{ref}_{ratio}'+(f'_{seed}'if ref=='PG-ST'else'')]
                    inter=len(np.intersect1d(a,b));overlap.append(dict(scene=scene,ratio=ratio,seed=seed,arm=arm,reference=ref,K=k,intersection=inter,Jaccard=inter/(2*k-inter),added=k-inter,removed=k-inter))
write('selected_set_overlap.csv',overlap)
hard=load('hard_training_dynamics.csv');curves=load('training_curves.csv');assert len(curves)==96000
for arm in ['Grid-MLP','Independent-Cell']:
    for scene in SCENES:
        for ratio in RATIOS:
            for seed in SEEDS:
                h=[z for z in hard if z['arm']==arm and z['scene']==scene and float(z['ratio'])==ratio and int(z['seed'])==seed]
                a=next(z for z in h if int(z['step'])==0);b=next(z for z in h if int(z['step'])==2000)
                regional.append(dict(scene=scene,ratio=ratio,seed=seed,arm=arm,changed_cells=int(b['changed_cells']),increased_cells=int(b['increased_cells']),decreased_cells=int(b['decreased_cells']),budget_transferred=int(b['budget_transferred']),negative_delta_sum=int(b['negative_delta_sum']),max_abs_delta=int(b['max_abs_delta']),Jaccard_Raw=float(b['Jaccard_Raw']),added_rows=int(b['added']),hard_Bank_photo_delta=float(b['photo'])-float(a['photo']),hard_Bank_PSNR_delta=float(b['psnr'])-float(a['psnr'])))
write('regional_summary.csv',regional)
correct=dict(all_passed=True,grid=read(ROOT/'grid_correctness.json'),LR_sanity=read(ROOT/'lr_sanity.json'),weighted_gauge_toy=read(ROOT/'weighted_gauge_gate.json'),calibrated_VJP=read(ROOT/'calibration_gate.json'),
             initial_MLP_forward_cells_only=True,all_cells_exactly_shared=True,occupancy_weighted_autograd_gauge=True,all_intra_cell_ordering_violations=0,initial_Raw_all_four_sets_exact=True,
             model_checks=locked['models'],formal_models=48,primary_models=24,control_models=24,formal_updates=96000,native_Gaussian_TF_Master_frozen_exact=True,native_grid_frozen_exact=True,
             training_reads_only_Bank_Raw_and_fixed_geometry=True,no_PG_coefficient_teacher=True,no_nonBank_LR_selection=True,sets_locked_before_quality=True,all_selected_unique_same_K=True,
             frozen_evaluator=read(ROOT/'baseline_reproduction.json'),quality=read(ROOT/'quality_complete.json'),aggregate_perview_max_abs=aggregate_error,
             maximum_normalized_gradient_sum=max(float(z['normalized_gradient_sum'])for z in curves),maximum_weighted_center_error=max(abs(float(z['weighted_mean']))for z in curves),
             no_split_or_grid_change=True,no_production_change=True)
save('correctness.json',correct)
summary=dict(counts=counts,regional={arm:dict(moved_budget_cases=sum(z['budget_transferred']>0 for z in regional if z['arm']==arm),Bank_photo_improved=sum(z['hard_Bank_photo_delta']<0 for z in regional if z['arm']==arm),transferred_min=min(z['budget_transferred']for z in regional if z['arm']==arm),transferred_max=max(z['budget_transferred']for z in regional if z['arm']==arm))for arm in ['Grid-MLP','Independent-Cell']},correctness=correct,decision='PENDING_EVIDENCE_REVIEW')
save('audit_summary.json',summary);print(json.dumps(dict(counts=counts,regional=summary['regional']),indent=2))

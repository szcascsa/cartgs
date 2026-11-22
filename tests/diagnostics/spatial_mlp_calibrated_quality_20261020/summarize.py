"""正式quality完成后统计；不反馈模型、LR、step或训练。"""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np

ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'spatial-mlp-implicit-si-20261020'
SCENES=['fr1_room','fr1_xyz','fr2_xyz'];RATIOS=[.01,.05,.1,.15];SEEDS=[2026,2027]
METRICS=['psnr','ssim','lpips','photo'];ARM='MLP-ST-fixed';REFS=['MLP-ST-legacy','PG-ST','Raw','Fixed','C']
def read(p):return json.loads(Path(p).read_text())
def load(p):return list(csv.DictReader(open(p)))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def write(p,rows):
    with open(p,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert read(ROOT/'status.json')['stage']=='complete'
assert read(ROOT/'baseline_reproduction.json')['all_passed']
frozen=read(ROOT/'selected_before_quality.json')
assert sha(ROOT/'rankings.npz')==frozen['ranking_sha']and sha(ROOT/'selected_sets.npz')==frozen['selected_sha']
q=load(ROOT/'baseline_quality_summary.csv')+load(ROOT/'fixed_quality_summary.csv')
v=load(ROOT/'baseline_quality_per_view.csv')+load(ROOT/'fixed_quality_per_view.csv')
qm={(z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed'])):z for z in q}
vm={(z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed']),int(z['frame_index'])):z for z in v}
assert len(qm)==len(q)and len(vm)==len(v)
maximum={m:0. for m in METRICS}
for z in q:
    info=read(OLD/z['scene']/'input_manifest.json');frames=info['bank_frames'if z['split']=='train'else'evaluation_frames']
    for m in METRICS:
        value=math.fsum(float(vm[z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed']),frame][m])for frame in frames)/len(frames)
        maximum[m]=max(maximum[m],abs(value-float(z[m])));assert abs(value-float(z[m]))<1e-12
write(ROOT/'quality_summary.csv',q);write(ROOT/'quality_per_view.csv',v)
deltas=[];paired=[];counts={};seedstats=[]
for reference in REFS:
    counts[reference]={}
    for split in ['train','eval']:
        cases=[];broad=[]
        for name in SCENES:
            info=read(OLD/name/'input_manifest.json');frames=info['bank_frames'if split=='train'else'evaluation_frames']
            for ratio in RATIOS:
                for seed in SEEDS:
                    refseed=seed if reference in ['PG-ST','MLP-ST-legacy']else-1
                    a=qm[name,split,ARM,ratio,seed];b=qm[name,split,reference,ratio,refseed]
                    delta=dict(scene=name,ratio=ratio,seed=seed,split=split,reference=reference,**{m:float(a[m])-float(b[m])for m in METRICS});cases.append(delta);deltas.append(delta)
                    for m in METRICS:
                        values=np.array([float(vm[name,split,ARM,ratio,seed,frame][m])-float(vm[name,split,reference,ratio,refseed,frame][m])for frame in frames]);sign=1 if m in ['psnr','ssim']else-1
                        row=dict(scene=name,ratio=ratio,seed=seed,split=split,reference=reference,metric=m,views=len(frames),mean=float(values.mean()),median=float(np.median(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)),improved_view_fraction=float(np.mean(sign*values>0)),equal_view_fraction=float(np.mean(values==0)))
                        paired.append(row)
                        if m=='psnr':broad.append(row)
        counts[reference][split]=dict(cases=len(cases),**{m:dict(improved=sum((1 if m in ['psnr','ssim']else-1)*z[m]>0 for z in cases),equal=sum(z[m]==0 for z in cases),worse=sum((1 if m in ['psnr','ssim']else-1)*z[m]<0 for z in cases),mean_delta=float(np.mean([z[m]for z in cases])),min_delta=min(z[m]for z in cases),max_delta=max(z[m]for z in cases))for m in METRICS},PSNR_paired_positive_median_cases=sum(z['median']>0 for z in broad),PSNR_majority_views_improved_cases=sum(z['improved_view_fraction']>.5 for z in broad))
for name in SCENES:
    for ratio in RATIOS:
        for split in ['train','eval']:
            for arm in [ARM,'MLP-ST-legacy','PG-ST']:
                a,b=[qm[name,split,arm,ratio,seed]for seed in SEEDS]
                seedstats.append(dict(scene=name,ratio=ratio,split=split,arm=arm,**{m+'_abs_difference':abs(float(a[m])-float(b[m]))for m in METRICS}))
write(ROOT/'quality_deltas.csv',deltas);write(ROOT/'paired_statistics.csv',paired);write(ROOT/'seed_consistency.csv',seedstats)

sets=np.load(ROOT/'selected_sets.npz');oldsets=np.load(OLD/'selected_sets.npz');overlap=[];promotion=[];dynamics=[]
tr=load(ROOT/'training_curves.csv');hh=load(ROOT/'hard_training_dynamics.csv');assert len(tr)==48000
for name in SCENES:
    data=np.load(OLD/name/'ranking_inputs.npz');ids=data['row_ids'];raw=np.load(OLD/name/'rankings.npz')['A_order'];rank=np.empty(len(ids),np.int64);rank[raw]=np.arange(1,len(ids)+1)
    assert np.all(ids[1:]>ids[:-1]);info=read(OLD/name/'input_manifest.json')
    for ratio in RATIOS:
        for seed in SEEDS:
            selected=sets[f'{name}_{ARM}_{ratio}_{seed}'];k=len(selected)
            for reference in REFS:
                key=f'{name}_{"MLP-ST"if reference=="MLP-ST-legacy"else reference}_{ratio}'+(f'_{seed}'if reference in ['PG-ST','MLP-ST-legacy']else'')
                b=oldsets[key];intersection=len(np.intersect1d(selected,b));overlap.append(dict(scene=name,ratio=ratio,seed=seed,reference=reference,K=k,intersection=intersection,Jaccard=intersection/(2*k-intersection),added=k-intersection,removed=k-intersection))
            added=np.setdiff1d(selected,ids[raw[:k]]);rr=rank[np.searchsorted(ids,added)];pct=(rr-1)/(len(ids)-1)
            promotion.append(dict(scene=name,ratio=ratio,seed=seed,added=len(added),Raw_percentile_mean=float(pct.mean())if len(pct)else'',Raw_percentile_p90=float(np.quantile(pct,.9))if len(pct)else'',max_Raw_rank=int(rr.max())if len(rr)else'',count_Raw_rank_gt_2K=int((rr>2*k).sum())))
            records=[z for z in tr if z['scene']==name and float(z['ratio'])==ratio and int(z['seed'])==seed]
            hard=[z for z in hh if z['scene']==name and float(z['ratio'])==ratio and int(z['seed'])==seed];initial=next(z for z in hard if int(z['step'])==0);final=next(z for z in hard if int(z['step'])==2000)
            first=[];last=[]
            for frame in info['bank_frames']:
                a=[float(z['sampled_photo'])for z in records if int(z['frame'])==frame];first.append(np.mean(a[:5]));last.append(np.mean(a[-5:]))
            dynamics.append(dict(scene=name,ratio=ratio,seed=seed,hard_Bank_photo_delta=float(final['photo'])-float(initial['photo']),hard_Bank_PSNR_delta=float(final['psnr'])-float(initial['psnr']),sampled_photo_first5=float(np.mean(first)),sampled_photo_last5=float(np.mean(last)),sampled_photo_delta=float(np.mean(last)-np.mean(first)),Jaccard_Raw=float(final['Jaccard_Raw']),last100_Jaccard=float(final['Jaccard_previous']),gradient_sum_ratio_max=max(abs(float(z['score_grad_sum']))/(float(z['score_grad_abs_sum'])+1e-30)for z in records),correction_mean_abs_max=max(abs(float(z['correction_mean']))for z in records),correction_std_final=float(records[-1]['correction_std']),expected_K_max_error=max(abs(float(z['expected_N'])-int(z['K']))for z in records)))
write(ROOT/'selected_set_overlap.csv',overlap);write(ROOT/'raw_rank_promotion.csv',promotion);write(ROOT/'training_dynamics_summary.csv',dynamics)
correct=dict(all_passed=True,models=24,formal_updates=48000,baselines=read(ROOT/'baseline_reproduction.json'),quality=read(ROOT/'quality_complete.json'),inputs_before_after_SHA_exact=read(ROOT/'inputs_before.json')['all_passed']and read(ROOT/'inputs_after.json')['all_passed'],all_sets_unique_same_K=True,sets_fixed_before_quality=True,initial_Raw_exact=True,only_MLP_optimized=True,frozen_tensors_exact=True,independent_weight_forward_and_rankings=True,aggregate_per_view_max_abs=maximum,no_eval_tuning=True,final_step_fixed=2000,model_checks=frozen['models'])
save(ROOT/'correctness.json',correct)
save(ROOT/'audit_summary.json',dict(counts=counts,training=dict(Bank_hard_photo_improved=sum(z['hard_Bank_photo_delta']<0 for z in dynamics),sampled_photo_improved=sum(z['sampled_photo_delta']<0 for z in dynamics),Raw_boundary_crossing_cases=sum(z['added']>0 for z in promotion)),correctness=correct,decision='PENDING_EVIDENCE_REVIEW'))
print(json.dumps(dict(counts=counts,training=read(ROOT/'audit_summary.json')['training']),indent=2))

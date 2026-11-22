from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent
def load(n):return list(csv.DictReader(open(ROOT/n)))
def write(n,rows):
    with open(ROOT/n,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
reg=json.loads((ROOT/'preregistration.json').read_text());tr=load('training_curves.csv');hard=load('hard_training_dynamics.csv');prom=load('raw_rank_promotion.csv');out=[];promotion=[];sampling=[]
for name in reg['scenes']:
    info=json.loads((ROOT/name/'input_manifest.json').read_text())
    for ratio in reg['ratios']:
        for seed in reg['seeds']:
            z=[a for a in tr if a['scene']==name and float(a['ratio'])==ratio and int(a['seed'])==seed];first=[];last=[]
            for frame in info['bank_frames']:
                zz=[a for a in z if int(a['frame'])==frame];first.append(np.mean([float(a['sampled_photo'])for a in zz[:5]]));last.append(np.mean([float(a['sampled_photo'])for a in zz[-5:]]))
            hh=[a for a in hard if a['scene']==name and float(a['ratio'])==ratio and int(a['seed'])==seed];a=next(v for v in hh if int(v['step'])==0);b=next(v for v in hh if int(v['step'])==2000)
            out.append(dict(scene=name,ratio=ratio,seed=seed,stochastic_first5=float(np.mean(first)),stochastic_last5=float(np.mean(last)),stochastic_photo_delta=float(np.mean(last)-np.mean(first)),hard_photo_delta=float(b['photo'])-float(a['photo']),hard_psnr_delta=float(b['psnr'])-float(a['psnr']),final_Jaccard_Raw=float(b['Jaccard_Raw']),last100_Jaccard=float(b['Jaccard_previous'])))
            count=[int(a['sampled_N'])for a in z];sampling.append(dict(scene=name,ratio=ratio,seed=seed,K=int(z[0]['K']),sampled_min=min(count),sampled_max=max(count),sampled_mean=float(np.mean(count)),sampled_initial=count[0],sampled_final=count[-1],calibration_max_abs_error=max(abs(float(a['expected_N'])-int(a['K']))for a in z),correction_std_final=float(z[-1]['correction_std']),correction_mean_final=float(z[-1]['correction_mean']),correction_max_abs_final=float(z[-1]['correction_max_abs']),theta_grad_norm_max=max(float(a['theta_grad_norm'])for a in z)))
            p=next(a for a in prom if a['scene']==name and float(a['ratio'])==ratio and int(a['seed'])==seed and a['arm']=='MLP-ST');s=next(a for a in prom if a['scene']==name and float(a['ratio'])==ratio and int(a['seed'])==seed and a['arm']=='PG-ST')
            promotion.append(dict(scene=name,ratio=ratio,seed=seed,MLP_added=int(p['N']),PG_added=int(s['N']),MLP_Raw_pct_mean=float(p['mean']),PG_Raw_pct_mean=float(s['mean']),MLP_Raw_pct_p90=float(p['p90']),PG_Raw_pct_p90=float(s['p90']),MLP_max_Raw_rank=int(p['max_Raw_rank']),PG_max_Raw_rank=int(s['max_Raw_rank']),MLP_gt2K=int(p['count_Raw_rank_gt_2K']),PG_gt2K=int(s['count_Raw_rank_gt_2K'])))
write('stochastic_hard_dynamics.csv',out);write('promotion_comparison.csv',promotion);write('sampling_statistics.csv',sampling)
print('DIAGNOSTIC_SUMMARY_COMPLETE',flush=True)

"""方差、目标动态与promotion补充；纯posthoc描述，不作选参。"""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent
def load(n):return list(csv.DictReader(open(ROOT/n)))
def write(n,rows):
    with open(ROOT/n,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
reg=json.loads((ROOT/'preregistration.json').read_text());tr=load('training_curves.csv');q=load('quality_summary.csv');hard=load('hard_training_dynamics.csv');prom=load('raw_rank_promotion.csv');rows=[];objective=[];seedrows=[]
for name in reg['scenes']:
    for ratio in reg['ratios']:
        for seed in reg['seeds']:
            z=[a for a in tr if a['scene']==name and float(a['ratio'])==ratio and int(a['seed'])==seed];meta=dict(scene=name,ratio=ratio,seed=seed)
            first=[];last=[]
            for frame in sorted({a['frame']for a in z}):
                a=[x for x in z if x['frame']==frame];first.append(np.mean([float(x['train_loss'])for x in a[:5]]));last.append(np.mean([float(x['train_loss'])for x in a[-5:]]))
            hs=[a for a in hard if a['scene']==name and float(a['ratio'])==ratio and int(a['seed'])==seed];initial=next(a for a in hs if int(a['step'])==0);final=next(a for a in hs if int(a['step'])==2000)
            objective.append(dict(**meta,Bank_frame_balanced_first5_loss=float(np.mean(first)),Bank_frame_balanced_last5_loss=float(np.mean(last)),stochastic_photo_delta=float(np.mean(last)-np.mean(first)),hard_photo_delta=float(final['photo'])-float(initial['photo']),hard_psnr_delta=float(final['psnr'])-float(initial['psnr']),advantage_std=float(np.std([float(a['advantage'])for a in z])),log_prob_std=float(np.std([float(a['log_prob'])for a in z])),positive_advantage_fraction=float(np.mean([float(a['advantage'])>0 for a in z])),last100_hard_changed_fraction=float(final['changed_fraction'])))
            p=next(a for a in prom if a['scene']==name and float(a['ratio'])==ratio and int(a['seed'])==seed and a['arm']=='PG-PL');s=next(a for a in prom if a['scene']==name and float(a['ratio'])==ratio and int(a['seed'])==seed and a['arm']=='PG-ST')
            rows.append(dict(**meta,PL_added=int(p['N']),ST_added=int(s['N']),PL_Raw_percentile_mean=float(p['mean']),ST_Raw_percentile_mean=float(s['mean']),PL_Raw_percentile_p90=float(p['p90']),ST_Raw_percentile_p90=float(s['p90']),PL_Raw_percentile_max=float(p['max']),ST_Raw_percentile_max=float(s['max']),PL_below2K=int(p['added_below_2K']),ST_below2K=int(s['added_below_2K'])))
        for arm in ['PG-PL','PG-ST']:
            z=[a for a in q if a['scene']==name and float(a['ratio'])==ratio and a['arm']==arm and a['split']=='eval'];assert len(z)==2
            seedrows.append(dict(scene=name,ratio=ratio,arm=arm,PSNR_seed_abs_difference=abs(float(z[0]['psnr'])-float(z[1]['psnr'])),photo_seed_abs_difference=abs(float(z[0]['photo'])-float(z[1]['photo']))))
write('stochastic_objective_dynamics.csv',objective);write('promotion_comparison.csv',rows);write('eval_seed_consistency.csv',seedrows)
print('POSTHOC_DIAGNOSTICS_WRITTEN',flush=True)

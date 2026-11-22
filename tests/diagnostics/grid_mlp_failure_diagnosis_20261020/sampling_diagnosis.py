"""只读保存score与cached uniform，量化训练sample与hard导出集合的差异。"""
from pathlib import Path
import csv,json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent;GRID=ROOT.parent/'grid-mlp-implicit-si-20261020'
torch.set_num_threads(2)
results=[]
for scene in ['fr1_room','fr1_xyz','fr2_xyz']:
    for arm in ['Grid-MLP','Independent-Cell']:
        for ratio in [.01,.05,.1,.15]:
            for seed in [2026,2027]:
                case=GRID/scene/'models'/f'{arm}_{int(100*ratio)}_seed_{seed}'
                training=[json.loads(line[3:])for line in (case/'training.log').read_text().splitlines()if line.startswith('PG ')and json.loads(line[3:])['kind']=='training']
                for step in [0,2000]:
                    state=dict(torch.jit.load(str(case/f'step_{step}.pt'),map_location='cpu').named_buffers())
                    score=state['score'];u=state['cached_uniform'];n=len(score);k=int(n*ratio+.5)
                    lo=float(score.min())-40;hi=float(score.max())+40
                    for _ in range(42):
                        mid=.5*(lo+hi)
                        if float((score-mid).sigmoid().sum())>k:lo=mid
                        else:hi=mid
                    p=(score-.5*(lo+hi)).sigmoid();sample=(u<p).numpy();order=np.argsort(-score.numpy(),kind='stable');hard=order[:k];hits=int(sample[hard].sum());count=int(sample.sum())
                    if step==0:assert count==training[0]['sampled_N']
                    results.append(dict(scene=scene,arm=arm,ratio=ratio,seed=seed,step=step,N=n,K=k,sample_count=count,
                        sample_hard_intersection=hits,sample_hard_recall=hits/k,sample_hard_Jaccard=hits/(k+count-hits),
                        expected_hard_mass_fraction=float(p[hard].sum()/k),mean_p_hard=float(p[hard].mean()),
                        expected_count=float(p.sum()),min_abs_sampling_margin=float(abs(p-u).min())))
with open(ROOT/'sampling_export_mismatch.csv','w',newline='')as f:
    writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
summary={arm:{str(step):dict(median_sample_hard_Jaccard=float(np.median([r['sample_hard_Jaccard']for r in results if r['arm']==arm and r['step']==step])),
    median_sample_hard_recall=float(np.median([r['sample_hard_recall']for r in results if r['arm']==arm and r['step']==step])),
    median_expected_hard_mass=float(np.median([r['expected_hard_mass_fraction']for r in results if r['arm']==arm and r['step']==step])))for step in [0,2000]}for arm in ['Grid-MLP','Independent-Cell']}
(ROOT/'sampling_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

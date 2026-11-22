"""独立排序/概率与逐view聚合复核，不改变任何选择。"""
from pathlib import Path
import json,csv,math
import numpy as np,torch
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
reg=read(ROOT/'preregistration.json');selected=dict(np.load(ROOT/'selected_sets.npz'));rankings=dict(np.load(ROOT/'rankings.npz'));snapshots=0;stepcount=0;max_shift=0;all_grad=True;mincov=1.;baseline_order=True
for name in reg['scenes']:
    here=ROOT/name;info=read(here/'input_manifest.json');x=np.load(here/'ranking_inputs.npz');ids=x['row_ids'];N=len(ids);base=np.log(x['raw_I'].astype(float)+1e-12)
    for ratio in reg['ratios']:
        K=int(ratio*N+.5)
        for seed in reg['seeds']:
            case=f'PL_{int(ratio*100)}_seed_{seed}';d=here/'models'/case;records=[json.loads(l[3:])for l in(d/'training.log').read_text().splitlines()if l.startswith('PG ')];baselines={z['frame']:z['photo']for z in records if z['kind']=='hard_view'and z['step']==0};noise=set();step=0
            for z in records:
                if z['kind']!='training':continue
                step+=1;stepcount+=1
                assert z['step']==step and z['sampled_count']==K and z['K']==K
                assert z['frame']in info['bank_frames']and z['frame']not in info['evaluation_frames']
                assert z['baseline_before']==baselines[z['frame']]and z['advantage']==z['train_loss']-baselines[z['frame']]
                baselines[z['frame']]=.99*baselines[z['frame']]+.01*z['train_loss'];assert z['baseline_after']==baselines[z['frame']]
                sig=tuple(z['noise_signature']);assert sig not in noise;noise.add(sig)
                assert z['selected_nonzero_grad_N']<=K and z['unselected_nonzero_grad_N']<=N-K
                all_grad&=z['nonzero_grad_N']==N;mincov=min(mincov,z['gradient_coverage']);max_shift=max(max_shift,z['gradient_shift_residual'])
                for k in ['train_loss','baseline_before','advantage','log_prob','reinforce_loss','grad_norm','grad_max','sum_gradient','sum_abs_gradient','coefficient_norm']:assert math.isfinite(z[k])
                assert z['minimum_denominator']>0 and z['minimum_shifted_weight']>0
            assert step==2000
            for step in reg['tensor_checkpoints']:
                t=dict(torch.jit.load(str(d/f'step_{step}.pt'),map_location='cpu').named_buffers());score=base+t['coefficient'].numpy();order=np.array(sorted(range(N),key=lambda i:(-float(score[i]),int(ids[i]))));rows=np.sort(ids[order[:K]]);key=f'{name}_{case}_step_{step}'
                np.testing.assert_array_equal(rows,selected[key]);np.testing.assert_array_equal(rankings[key],ids[order]);assert len(rows)==len(np.unique(rows))==K
                if step==0:np.testing.assert_array_equal(rows,selected[f'{name}_Raw_{ratio}'])
                snapshots+=1
                if step==2000:
                    for frame,value in baselines.items():assert float(t['frame_baseline'][frame])==value
q=list(csv.DictReader(open(ROOT/'quality_summary.csv')));v=list(csv.DictReader(open(ROOT/'quality_per_view.csv')));maxerror={m:0.0 for m in ['psnr','ssim','lpips','photo']}
for z in q:
    vv=[a for a in v if all(a[k]==z[k]for k in['scene','split','arm','ratio','seed','step'])];info=read(ROOT/z['scene']/'input_manifest.json');frames=info['bank_frames'if z['split']=='train'else'evaluation_frames'];assert sorted(int(a['frame_index'])for a in vv)==sorted(frames)
    for m in maxerror:
        err=abs(math.fsum(float(a[m])for a in vv)/len(vv)-float(z[m]));maxerror[m]=max(maxerror[m],err);assert err<1e-12
out=dict(all_passed=True,snapshots=snapshots,formal_steps=stepcount,all_counts_exact_K=True,all_noise_signatures_unique_within_each_run=True,baseline_sequence_exact=True,
    all_candidates_gradient_every_step=all_grad,minimum_per_step_gradient_coverage=mincov,max_gradient_shift_residual=max_shift,all_selected_sets_exact_independent_sort=True,zero_Raw_exact=True,per_view_protocol_exact=True,aggregate_max_abs_error=maxerror)
(ROOT/'independent_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

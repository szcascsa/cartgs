"""独立验证native保存权重、correction、归一化、排序、共享传播与逐view聚合。"""
from pathlib import Path
import json,csv,math
import numpy as np,torch
torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parent
reg=json.loads((ROOT/'preregistration.json').read_text());sets=np.load(ROOT/'selected_sets.npz');rankings=np.load(ROOT/'rankings.npz');snapshots=0;updates=0;max_output_error=0.;max_metric={m:0. for m in ['psnr','ssim','lpips','photo']}
for name in reg['scenes']:
    here=ROOT/name;info=json.loads((here/'input_manifest.json').read_text());x=np.load(here/'ranking_inputs.npz');ids=x['row_ids'];N=len(ids);base=np.log(x['raw_I'].astype(float)+1e-12);norm=np.load(here/'normalized_xyz.npz');buffers=dict(torch.jit.load(info['tf'],map_location='cpu').named_buffers());lo=buffers['grid.aabb_min'].numpy().astype(float);hi=buffers['grid.aabb_max'].numpy().astype(float);normalized=2*(x['xyz'].astype(float)-lo)/(hi-lo)-1;np.testing.assert_array_equal(norm['normalized_xyz'],normalized)
    for ratio in reg['ratios']:
        K=int(ratio*N+.5)
        for seed in reg['seeds']:
            case=f'MLP_{int(100*ratio)}_seed_{seed}';d=here/'models'/case
            for step in reg['tensor_checkpoints']:
                t=dict(torch.jit.load(str(d/f'step_{step}.pt'),map_location='cpu').named_buffers());a=t['correction'].numpy();score=base+a;o=np.array(sorted(range(N),key=lambda i:(-float(score[i]),int(ids[i]))));s=np.sort(ids[o[:K]]);key=f'{name}_{case}_step_{step}'
                np.testing.assert_array_equal(sets[key],s);np.testing.assert_array_equal(rankings[key],ids[o]);np.testing.assert_array_equal(t['normalized_xyz'].numpy(),normalized)
                model=torch.nn.Sequential(torch.nn.Linear(3,32),torch.nn.SiLU(),torch.nn.Linear(32,32),torch.nn.SiLU(),torch.nn.Linear(32,1)).double();saved=torch.jit.load(str(d/f'step_{step}_weights.pt'),map_location='cpu');state=dict(saved.named_parameters());model.load_state_dict(state)
                with torch.no_grad():pred=model(torch.from_numpy(normalized)).flatten().numpy()
                err=float(abs(pred-a).max());max_output_error=max(max_output_error,err);np.testing.assert_allclose(pred,a,atol=1e-9,rtol=1e-11)
                if step==0:assert np.all(a==0);assert np.all(pred==0)
                assert len(s)==len(np.unique(s))==K;snapshots+=1
            for step in reg['output_update_steps']:
                t=dict(torch.jit.load(str(d/f'output_update_{step}.pt'),map_location='cpu').named_buffers());diff=(t['correction_after']-t['correction_before']).abs();assert torch.isfinite(diff).all();assert t['selected_mask'].dtype==torch.bool;updates+=1
q=list(csv.DictReader(open(ROOT/'quality_summary.csv')));v=list(csv.DictReader(open(ROOT/'quality_per_view.csv')))
for z in q:
    vv=[a for a in v if all(a[k]==z[k]for k in ['scene','split','ratio','seed','step','arm'])];info=json.loads((ROOT/z['scene']/'input_manifest.json').read_text());expected=info['bank_frames'if z['split']=='train'else'evaluation_frames'];assert sorted(int(a['frame_index'])for a in vv)==sorted(expected)
    for m in max_metric:
        err=abs(math.fsum(float(a[m])for a in vv)/len(vv)-float(z[m]));max_metric[m]=max(max_metric[m],err);assert err<1e-12
out=dict(all_passed=True,weight_snapshots=snapshots,output_update_snapshots=updates,independent_Python_weight_forward_max_abs=max_output_error,all_rankings_exact_independent_sort=True,normalization_exact_from_frozen_AABB=True,aggregate_max_abs_error=max_metric)
(ROOT/'independent_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

"""独立检查全部saved tensor、hard ranking、same-K集合及per-view聚合。"""
from pathlib import Path
import json,csv,math
import numpy as np,torch
ROOT=Path(__file__).resolve().parent
reg=json.loads((ROOT/'preregistration.json').read_text());sets=dict(np.load(ROOT/'selected_sets.npz'));rankings=dict(np.load(ROOT/'rankings.npz'));sampled=dict(np.load(ROOT/'training_sampling_sets.npz'));snapshots=0
for scene in reg['scenes']:
    here=ROOT/scene;data=np.load(here/'ranking_inputs.npz');ids=data['row_ids'];N=len(ids)
    with open(here/'training_input.bin','rb')as f:
        assert int(np.fromfile(f,np.int64,1)[0])==N;np.testing.assert_array_equal(np.fromfile(f,np.int64,N),ids);base=np.fromfile(f,np.float64,N)
    np.testing.assert_array_equal(base,np.log(data['raw_I'].astype(float)+reg['epsilon']))
    for ratio in reg['ratios']:
        K=int(ratio*N+.5)
        for seed in reg['seeds']:
            case=f'PG_{int(100*ratio)}_seed_{seed}';d=here/'models'/case
            initial_records=json.loads((d/'completed.json').read_text())['records'];first=next(x for x in initial_records if x.get('step')==1)
            assert len(sampled[f'{scene}_{case}_step_0'])==first['sampled_N']
            for step in reg['checkpoints']:
                t=dict(torch.jit.load(str(d/f'step_{step}.pt'),map_location='cpu').named_buffers());a=t['coefficient'].numpy();score=base+a
                order=np.array(sorted(range(N),key=lambda i:(-float(score[i]),int(ids[i]))))
                z=np.load(d/f'step_{step}.npz');np.testing.assert_array_equal(z['score'],score);np.testing.assert_array_equal(z['order'],order)
                selected=np.sort(ids[order[:K]]);key=f'{scene}_{case}_step_{step}'
                np.testing.assert_array_equal(sets[key],selected);np.testing.assert_array_equal(rankings[key],ids[order]);np.testing.assert_array_equal(z['selected'],selected)
                assert len(selected)==K and len(np.unique(selected))==K and np.isin(selected,ids).all();snapshots+=1
quality=list(csv.DictReader(open(ROOT/'quality_summary.csv')));views=list(csv.DictReader(open(ROOT/'quality_per_view.csv')));maxerror={m:0.0 for m in ['psnr','ssim','lpips','photo']}
for z in quality:
    v=[a for a in views if all(a[k]==z[k]for k in['scene','split','arm','ratio','seed','step'])]
    info=json.loads((ROOT/z['scene']/'input_manifest.json').read_text());expected=info['bank_frames'if z['split']=='train'else'evaluation_frames']
    assert sorted(int(a['frame_index'])for a in v)==sorted(expected)
    for m in maxerror:
        error=abs(math.fsum(float(a[m])for a in v)/len(v)-float(z[m]));maxerror[m]=max(maxerror[m],error);assert error<1e-12
out=dict(all_passed=True,snapshots=snapshots,independent_python_tuple_sort_exact=True,same_K_unique_candidate_rows=True,raw_binary_base_exact=True,initial_sampling_count_exact=True,per_view_protocol_exact=True,aggregate_max_abs_error=maxerror)
(ROOT/'independent_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

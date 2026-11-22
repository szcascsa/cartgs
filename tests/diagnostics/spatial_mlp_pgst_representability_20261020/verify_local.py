"""本地独立复算MSE、含tie相关性、排序、质量聚合及交付SHA，无GPU。"""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,math,re,subprocess
import numpy as np

ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'spatial_mlp_implicit_si_20261020'
load=lambda p:list(csv.DictReader(open(ROOT/p)))
def ranks(values):
    order=np.argsort(values,kind='stable');sorted_values=values[order]
    starts=np.flatnonzero(np.r_[True,sorted_values[1:]!=sorted_values[:-1]]);ends=np.r_[starts[1:],len(values)]
    result=np.empty(len(values),float)
    for start,end in zip(starts,ends):result[order[start:end]]=(start+1+end)/2
    return result
manifest=json.loads((ROOT/'artifact_manifest.json').read_text())['files']
for name,meta in manifest.items():
    path=ROOT/name;assert path.stat().st_size==meta['bytes']and hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
reg=load('regression_metrics.csv');assert len(reg)==24
sets=np.load(ROOT/'selected_sets.npz');rankings=np.load(ROOT/'rankings.npz');overlap=load('topk_overlap.csv')
maximum=0.
for row in reg:
    scene=row['scene'];ratio=float(row['ratio']);seed=int(row['seed']);case=ROOT/scene/'models'/f'fit_{int(100*ratio)}_seed_{seed}'
    field=np.load(case/'field.npz');ids=field['row_ids'];target=field['target'];pred=field['prediction'];meta=json.loads((case/'metadata.json').read_text())
    assert hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest()==meta['target_SHA256']
    np.testing.assert_array_equal(field['normalized_xyz'],np.load(OLD/scene/'normalized_xyz.npz')['normalized_xyz'])
    computed=dict(MSE=float(np.mean((pred-target)**2)),RMSE=float(np.sqrt(np.mean((pred-target)**2))),Pearson=float(np.corrcoef(pred,target)[0,1]),Spearman=float(np.corrcoef(ranks(pred),ranks(target))[0,1]),target_std=float(target.std()),prediction_std=float(pred.std()))
    for name,value in computed.items():
        error=abs(value-float(row[name]));maximum=max(maximum,error);assert error<1e-12
    k=int(row['K']);data=np.load(OLD/scene/'ranking_inputs.npz');base=np.log(data['raw_I'].astype(float)+1e-12)
    for arm,coefficient in [('PG-ST',target),('MLP-fit',pred)]:
        score=base+coefficient;order=np.array(sorted(range(len(ids)),key=lambda i:(-float(score[i]),int(ids[i]))))
        key=f'{scene}_{ratio}_{seed}_{arm}'
        np.testing.assert_array_equal(ids[order],rankings[key]);np.testing.assert_array_equal(np.sort(ids[order[:k]]),sets[key])
        assert len(sets[key])==len(np.unique(sets[key]))==k
    loss=np.load(case/'full_MSE_curve.npy');assert len(loss)==5001 and np.isfinite(loss).all()
    assert abs(loss[-1]-computed['MSE'])<1e-12
    a=sets[f'{scene}_{ratio}_{seed}_MLP-fit'];b=sets[f'{scene}_{ratio}_{seed}_PG-ST'];intersection=len(np.intersect1d(a,b))
    reference=next(z for z in overlap if z['scene']==scene and float(z['ratio'])==ratio and int(z['seed'])==seed and z['arm']=='MLP-fit')
    assert int(reference['intersection'])==intersection and float(reference['recall'])==intersection/k
q=load('quality_summary.csv');v=load('quality_per_view.csv');assert len(q)==96 and len(v)==5680
groups=defaultdict(list)
for z in v:groups[z['scene'],z['ratio'],z['seed'],z['split'],z['arm']].append(z)
for z in q:
    rows=groups[z['scene'],z['ratio'],z['seed'],z['split'],z['arm']];assert len(rows)==int(z['views'])
    for metric in ['psnr','ssim','lpips','photo']:
        assert abs(math.fsum(float(a[metric])for a in rows)/len(rows)-float(z[metric]))<1e-12
for target in re.findall(r'\]\((H:/[^)]+)\)',(ROOT/'SPATIAL_MLP_PGST_REPRESENTABILITY_AUDIT.md').read_text(encoding='utf-8')):assert Path(target).exists()
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True,cwd=OLD.parents[2]).strip()
checks=dict(all_passed=True,models=24,manifest_files_verified=len(manifest),independent_regression_max_abs_error=maximum,
            tie_aware_Spearman_verified=True,independent_final_rankings_and_sets_exact=True,quality_view_records=5680,
            independent_quality_aggregate_verified=True,production_tracked_tree_index_unchanged=True)
(ROOT/'local_delivery_checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))

"""离线独立复算交付包：不需要torch或GPU。"""
from collections import defaultdict
from pathlib import Path
import csv, hashlib, json, math, re, subprocess
import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'spatial_mlp_implicit_si_20261020'
load=lambda name:list(csv.DictReader(open(ROOT/name)))
manifest=json.loads((ROOT/'artifact_manifest.json').read_text())['files']
for rel,meta in manifest.items():
    path=ROOT/rel
    assert path.stat().st_size==meta['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256'],rel
q=load('quality_summary.csv');v=load('quality_per_view.csv');pairs=load('paired_statistics.csv');d=load('quality_deltas.csv')
summary=json.loads((ROOT/'audit_summary.json').read_text());groups=defaultdict(list)
keys=['scene','split','arm','ratio','seed']
for row in v:groups[tuple(row[k]for k in keys)].append(row)
assert len(q)==216 and len(v)==12780
maximum=0.
for row in q:
    rows=groups[tuple(row[k]for k in keys)]
    assert len(rows)==int(row['views'])and len({x['frame_index']for x in rows})==len(rows)
    for metric in ['psnr','ssim','lpips','photo']:
        error=abs(math.fsum(float(x[metric])for x in rows)/len(rows)-float(row[metric]))
        maximum=max(maximum,error);assert error<1e-12
for reference in ['MLP-ST-legacy','PG-ST','Raw','Fixed','C']:
    for split in ['train','eval']:
        rows=[row for row in d if row['reference']==reference and row['split']==split];assert len(rows)==24
        for metric in ['psnr','ssim','lpips','photo']:
            sign=1 if metric in ['psnr','ssim']else-1
            assert sum(sign*float(row[metric])>0 for row in rows)==summary['counts'][reference][split][metric]['improved']
sets=np.load(ROOT/'selected_sets.npz');rankings=np.load(ROOT/'rankings.npz')
assert len(load('training_curves.csv'))==48000
for scene in ['fr1_room','fr1_xyz','fr2_xyz']:
    data=np.load(OLD/scene/'ranking_inputs.npz');ids=data['row_ids'];base=np.log(data['raw_I'].astype(float)+1e-12)
    raw=ids[np.lexsort((ids,-base))]
    for ratio in [.01,.05,.1,.15]:
        k=int(len(ids)*ratio+.5)
        for seed in [2026,2027]:
            key=f'{scene}_MLP-ST-fixed_{ratio}_{seed}'
            np.testing.assert_array_equal(rankings[key+'_step_0'],raw)
            field=np.load(ROOT/scene/'models'/f'fixed_{int(100*ratio)}_seed_{seed}'/'final_field.npz')
            correction=field['correction'];score=base+correction
            order=np.array(sorted(range(len(ids)),key=lambda i:(-float(score[i]),int(ids[i]))))
            np.testing.assert_array_equal(rankings[key+'_step_2000'],ids[order])
            np.testing.assert_array_equal(sets[key],np.sort(ids[order[:k]]))
            assert len(sets[key])==len(np.unique(sets[key]))==k
for row in pairs:
    a={int(x['frame_index']):x for x in groups[row['scene'],row['split'],'MLP-ST-fixed',row['ratio'],row['seed']]}
    refseed=row['seed']if row['reference']in ['MLP-ST-legacy','PG-ST']else'-1'
    b={int(x['frame_index']):x for x in groups[row['scene'],row['split'],row['reference'],row['ratio'],refseed]}
    values=sorted(float(a[frame][row['metric']])-float(b[frame][row['metric']])for frame in a)
    assert abs(math.fsum(values)/len(values)-float(row['mean']))<1e-12
    for key,value in [('median',np.median(values)),('p10',np.quantile(values,.1)),('p90',np.quantile(values,.9))]:
        assert abs(float(value)-float(row[key]))<1e-12
for target in re.findall(r'\]\((H:/[^)]+)\)',(ROOT/'SPATIAL_MLP_CALIBRATED_QUALITY_AUDIT.md').read_text(encoding='utf-8')):
    assert Path(target).exists(),target
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True,cwd=OLD.parents[2]).strip()
checks=dict(all_passed=True,manifest_files_SHA_verified=len(manifest),models=24,training_updates=48000,
            quality_summaries=len(q),quality_per_view_rows=len(v),paired_rows=len(pairs),
            independent_aggregate_max_abs_error=maximum,independent_final_rankings_and_selected_sets_exact=True,
            independent_initial_Raw_rankings_exact=True,paired_statistics_independently_recomputed=True,
            tracked_tree_index_unchanged=True)
(ROOT/'local_delivery_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))

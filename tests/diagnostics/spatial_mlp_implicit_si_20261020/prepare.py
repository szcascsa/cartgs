"""保留旧PG-ST与全部冻结输入；只预注册新selection estimator。"""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parent;OLD=Path('/home/prmi/zwj/per-gaussian-implicit-si-20261020')
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
from protocol import reg
import torch
manifest=read(OLD/'artifact_manifest.json')['files'];prior=read(OLD/'representability_check.json')['scenes'];checks={}
for name in reg['scenes']:
    src=OLD/name;dst=ROOT/name;dst.mkdir(exist_ok=True)
    info=read(src/'input_manifest.json');p=prior[name]
    for path,h in p['source_sha'].items():assert sha(path)==h['sha256']
    assert sha(info['cameras'])==p['cameras_sha']
    assert not(set(info['bank_frames'])&set(info['evaluation_frames']))
    names=['ranking_inputs.npz','rankings.npz','frozen_arrays.npz','hierarchy_input.bin','population.json','input_manifest.json','prior_pool_rankings.npz','prior_selected_sets.npz','prior_quality_summary.csv','prior_quality_per_view.csv','audit_worker','common.py','evaluate.py','fixed_marginal_reference.py','baseline.py','baseline_quality.csv','baseline_per_view.csv','baseline_reproduction.json','training_input.bin','training_manifest.json']
    sources={}
    for n in names:
        if name+'/'+n in manifest:assert sha(src/n)==manifest[name+'/'+n]['sha256']
        elif str(src/n)in p['input_sha']:assert sha(src/n)==p['input_sha'][str(src/n)]
        elif n=='training_input.bin':pass
        else:assert n in ['baseline.py','training_manifest.json']
        shutil.copy2(src/n,dst/n);sources[str(dst/n)]=sha(dst/n)
    shutil.copytree(src/'old_tree',dst/'old_tree',dirs_exist_ok=True);shutil.copytree(src/'runtime',dst/'runtime',dirs_exist_ok=True);shutil.copytree(src/'models',dst/'legacy_models',dirs_exist_ok=True)
    for n in ['quality_summary.csv','quality_per_view.csv','selected_sets.npz','gradient_statistics.csv','selected_set_overlap.csv','coefficient_statistics.csv']:
        assert sha(OLD/n)==manifest[n]['sha256'];shutil.copy2(OLD/n,ROOT/('legacy_'+n))
    x=np.load(dst/'ranking_inputs.npz');b=np.load(dst/'rankings.npz');ids=x['row_ids'];base=np.log(x['raw_I'].astype(float)+1e-12)
    np.testing.assert_array_equal(np.lexsort((ids,-base)),b['A_order'])
    save(dst/'protocol.json',dict(ratios=reg['ratios']))
    buffers=dict(torch.jit.load(info['tf'],map_location='cpu').named_buffers())
    lo=buffers['grid.aabb_min'].numpy().astype(np.float64);hi=buffers['grid.aabb_max'].numpy().astype(np.float64)
    xyz=x['xyz'].astype(np.float64);normalized=2*(xyz-lo)/(hi-lo)-1
    assert np.isfinite(normalized).all()and(hi>lo).all()
    np.savez_compressed(dst/'normalized_xyz.npz',row_ids=ids,xyz=xyz,normalized_xyz=normalized,aabb_min=lo,aabb_max=hi)
    save(dst/'normalization.json',dict(aabb_source='frozen TF grid buffers',aabb_min=lo.tolist(),aabb_max=hi.tolist(),
        normalized_sha=hashlib.sha256(normalized.tobytes()).hexdigest(),min=normalized.min(0).tolist(),max=normalized.max(0).tolist(),
        mean=normalized.mean(0).tolist(),std=normalized.std(0).tolist(),outside_rows=int(((normalized<-1)|(normalized>1)).any(1).sum()),clamp=False,refit=False))
    checks[name]=dict(N=len(ids),zero_full_Raw_ranking_exact=True,input_sha=sources,source_sha=p['source_sha'],cameras_sha=p['cameras_sha'],
        runtime_sha={str(f):sha(f)for f in(dst/'runtime/lib').iterdir()},legacy_model_sha={str(f):sha(f)for f in(dst/'legacy_models').glob('*/step_2000.pt')})
save(ROOT/'input_checks.json',dict(all_passed=True,scenes=checks));print('PREPARATION_EXACT_PASS',flush=True)

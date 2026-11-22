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
reg=dict(selection_estimator='pl_reinforce_exact_k',scenes=['fr1_room','fr1_xyz','fr2_xyz'],ratios=[.01,.05,.1,.15],seeds=[2026,2027],steps=2000,
    tensor_checkpoints=[0,250,1000,2000],hard_interval=100,hard_steps=[0]+list(range(100,2001,100))+[250],
    lr=.001,betas=[.9,.999],adam_eps=1e-8,weight_decay=0,temperature=1.,baseline_beta=.99,
    score='log(RawSI+1e-12)+a, FP64',initialization='exact zero',candidate='same full valid AND observed population',
    baseline='per Bank frame EMA, initial Raw hard TopK photo; advantage uses old baseline; update AFTER Adam step',
    loss='.8 L1 + .2 (1-fast_ssim), original frozen renderer',surrogate='detach(photo-old_baseline)*ordered_PL_log_probability; gradient descent',
    rng='torch manual_seed(seed) AFTER FrozenScene initialization; one new FP64 CUDA rand_like(N) per update; no other CUDA random draws; save CUDA RNG states',
    views='Bank only, same std::mt19937(seed) shuffled epochs as PG-ST; eval never used for optimization or checkpoint selection',
    sampling='fresh Gumbel TopK, exact K, no replacement; renderer source rows sorted ascending; log_prob keeps sampled order',
    log_prob='FP64 max-shift weights + prefix selected mass, O(N+K), strict finite/positive checks, no clamp',
    gradient='pure score function, no opacity mask/autograd renderer path to a; all unselected weights in denominator',
    runtime='CUDA synchronize wall timings sampling/log_prob/render/backward+Adam and complete step; diagnostic only',
    formal_runs=24,formal_updates=48000,no_tuning=True,no_grid=True,no_split=True,no_teacher=True,no_new_loss=True,
    toy_samples=200000,toy_seed=41020,toy_statistical_gate='frequency and gradient errors within 6 Monte Carlo SE + 1e-12; normalization/manual probability 1e-12; exact gradient finite difference 1e-8',
    commit=False,stage=False,push=False)
save(ROOT/'preregistration.json',reg)
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
    checks[name]=dict(N=len(ids),zero_full_Raw_ranking_exact=True,input_sha=sources,source_sha=p['source_sha'],cameras_sha=p['cameras_sha'],
        runtime_sha={str(f):sha(f)for f in(dst/'runtime/lib').iterdir()},legacy_model_sha={str(f):sha(f)for f in(dst/'legacy_models').glob('*/step_2000.pt')})
save(ROOT/'input_checks.json',dict(all_passed=True,scenes=checks));print('PREPARATION_EXACT_PASS',flush=True)

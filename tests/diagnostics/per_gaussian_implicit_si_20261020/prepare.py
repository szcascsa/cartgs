"""冻结输入复制、训练协议预注册；不生成 Bank。"""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
ROOT=Path(__file__).resolve().parent
OLD=Path('/home/prmi/zwj/final-rep-residual-fallback-20261020')
def read(p): return json.loads(Path(p).read_text())
def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
reg=dict(scenes=['fr1_room','fr1_xyz','fr2_xyz'],ratios=[.01,.05,.1,.15],seeds=[2026,2027],
    steps=2000,checkpoints=[0,250,1000,2000],learning_rate=.001,optimizer='Adam defaults',
    epsilon=1e-12,score_dtype='float64',coefficient_dtype='float64',temperature=1.,
    sampling='existing sampleSelectorST Bernoulli with cached per-model uniform; detached common offset recalibrates sum(sigmoid(score-offset))=K each step; user authorized',
    calibration='monotone bisection, 42 iterations; offset detached; not a learned parameter or loss',
    evaluation='hard score descending / source row ascending, same historical K',
    loss='.8 L1 + .2 (1-fast_ssim), exact existing production loss',
    training_views='Bank only; seeded shuffled complete epochs; no non-Bank optimization',
    training_inputs='source rows / Raw SI / Bank frame IDs / frozen map / frozen TF only; no C/B/tree labels',
    parameters='only one a per full valid AND observed candidate; all Gaussian/TF/Raw/contribution/Bank frozen',
    lr_source='cfg/selector/fast.yaml Selector.learning_rate=.001; no lr sweep or eval tuning',
    stopping='fixed final step, no early stopping or evaluation-based selection',
    gradient_ema_decay=.99,ratio_models='independent zero initialized vectors',
    representability='separate aC= -rankC/N-base; NEVER used to initialize/train PG',
    diagnostic_grid='frozen target2048 root IDs only AFTER training; never used for scoring',
    no_regularization=True,no_new_loss=True,no_grid_training=True,commit=False,stage=False,push=False)
save(ROOT/'preregistration.json',reg)
manifest=read(OLD/'artifact_manifest.json')['files'];checks={}
for name in reg['scenes']:
    src=OLD/name;dst=ROOT/name;dst.mkdir(exist_ok=True)
    prior=read(src/'prior_checks.json');info=read(src/'input_manifest.json')
    for p,v in prior['source_sha'].items(): assert sha(p)==v['sha256']
    assert sha(info['cameras'])==prior['camera_sha256']
    assert not(set(info['bank_frames'])&set(info['evaluation_frames']))
    names=['ranking_inputs.npz','rankings.npz','frozen_arrays.npz','hierarchy_input.bin','population.json','input_manifest.json','prior_pool_rankings.npz','prior_selected_sets.npz','prior_quality_summary.csv','prior_quality_per_view.csv','audit_worker','common.py','evaluate.py','fixed_marginal_reference.py']
    for n in names:
        if name+'/'+n in manifest: assert sha(src/n)==manifest[name+'/'+n]['sha256'],n
        elif n=='audit_worker': assert sha(src/n)==prior['source_math']['audit_worker']
        shutil.copy2(src/n,dst/n)
    shutil.copytree(src/'old_tree',dst/'old_tree',dirs_exist_ok=True);shutil.copytree(src/'runtime',dst/'runtime',dirs_exist_ok=True)
    # A/B/C/F baseline gate is exactly the preceding audit's reproduction program.
    shutil.copy2(src/'baseline.py',dst/'baseline.py')
    save(dst/'protocol.json',reg)
    x=dict(np.load(dst/'ranking_inputs.npz'));ids=x['row_ids'];I=x['raw_I'].astype(np.float64);N=len(ids)
    assert np.isfinite(I).all()and(I>=0).all()
    base=np.log(I+reg['epsilon']); raw=np.lexsort((ids,-I));np.testing.assert_array_equal(np.lexsort((ids,-base)),raw)
    b=dict(np.load(dst/'rankings.npz'));np.testing.assert_array_equal(raw,b['A_order'])
    rank=np.empty(N,np.int64);rank[b['C0_order']]=np.arange(N)
    target=-rank/N;ac=target-base;order=np.lexsort((ids,-(base+ac)))
    np.testing.assert_array_equal(order,b['C0_order'])
    for r in reg['ratios']:np.testing.assert_array_equal(ids[order[:int(r*N+.5)]],ids[b['C0_order'][:int(r*N+.5)]])
    np.savez_compressed(dst/'representability_only.npz',aC=ac,target=target,order=order)
    # Training loads this small, label-free binary, not ranking/tree archives.
    with open(dst/'training_input.bin','wb')as f:
        np.array([N],np.int64).tofile(f);ids.astype(np.int64).tofile(f);base.tofile(f)
    save(dst/'training_manifest.json',{k:info[k]for k in ['master','tf','config','cameras','bank_frames']})
    checks[name]=dict(N=N,zero_coefficient_full_Raw_ranking_exact=True,C_full_ranking_representable_exact=True,all_four_prefixes_exact=True,
        no_C_initialized_training=True,train_eval_disjoint=True,input_sha={str(dst/n):sha(dst/n)for n in names},
        source_sha=prior['source_sha'],cameras_sha=sha(info['cameras']),runtime_sha={str(p):sha(p)for p in(dst/'runtime/lib').iterdir()})
save(ROOT/'representability_check.json',dict(all_passed=True,scenes=checks,learned_result=False))
print('PREPARATION_AND_REPRESENTABILITY_PASS',flush=True)

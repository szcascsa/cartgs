"""复用上一轮冻结数据和F-C结果；不生成Bank、不读取质量选点。"""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parent;OLD=Path('/home/prmi/zwj/hierarchy-candidate-pool-20261020')
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
reg=dict(scenes=['fr1_room','fr1_xyz','fr2_xyz'],baseline=['A','B','C','F'],new_arms=['RF-C','RF-Raw','RF-Gap'],ratios=[.01,.05,.1,.15],max_existing_test_ratio=.15,
    backend='target-2048 ONLY',candidate='valid AND observed',maturity=1,
    F='exact frozen final leaf structural representative mask; no pure internal-only union',U='candidate minus F',
    residual='R=Raw*(1-final_leaf_cover); export exact original LibTorch CUDA selectCell and residual arithmetic',
    demand='D_r=C TopK intersect U; D_r must equal R-desc/source-row-asc residual prefix of length m_r and be nested',
    RF_C='F union D15 = F union top_m15(U,R)',RF_Raw='F union top_m15(U,Raw)',RF_Gap='F union top_m15(U,1-cover)',
    scorer='ALL fallback arms existing Current C score descending; source row ascending tie',
    static_pool='one fixed m15-size fallback pool per proxy for all ratios; no count tuning, no ratio-specific generation',
    baseline_gate='all3 scene A/B/C/F complete ranking, same-K selected sets, frozen tree exact; aggregate and per-view metrics EXACT before new rendering',
    exact_gate='R equals C nonrep score row-by-row EXACT; D15_RES equals D15; RF-C selected sets equal C at all4 ratios; RF-C real rendered metrics EXACT',
    hypotheses='compact fallback SUPPORTED if C can be exactly preserved with retained fraction well below100%; Raw/Gap supported only if stable across3scenes4ratios. No new arbitrary dB gate. R required only relative to tested equal-size proxies, not universal necessity.',
    metrics=['psnr','ssim','lpips','photo'],same_K='floor(ratio*N+.5) over full candidates',
    freeze='Master/TF/AABB/contributions/Raw/Bank/masks/tree/finalleaves/target2048/hsearch/origin/3reps/Q=.80/L3/scalar kappa/renderer/cameras/GT/metrics',
    forbidden='new proxies/count tuning/ratio tuning/multilevel/progressive/d50x32/Selector training/online',
    raw_contribution_renders=0,training_updates=0,performance_claim=False,stage=False,commit=False,push=False)
save(ROOT/'preregistration.json',reg);manifest=read(OLD/'artifact_manifest.json')['files'];checks=[]
for scene in reg['scenes']:
    old=OLD/scene;out=ROOT/scene;out.mkdir(exist_ok=False);prior=read(old/'prior_checks.json');info=read(old/'input_manifest.json')
    for p,a in prior['source_sha'].items():assert sha(p)==a['sha256'],('STOP_SOURCE_SHA',p)
    assert sha(info['cameras'])==prior['camera_sha256']
    names=['ranking_inputs.npz','rankings.npz','frozen_arrays.npz','hierarchy_input.bin','population.json','input_manifest.json','pool_rankings.npz','selected_sets.npz','quality_summary.csv','quality_per_view.csv','old_tree/hierarchy_tree.csv','old_tree/importance.bin','old_tree/hierarchy_telemetry.json','redundancy_statistics.csv','candidate_pool_summary.csv']
    for name in names:
        p=old/name;assert sha(p)==manifest[scene+'/'+name]['sha256'],('STOP_ARTIFACT_SHA',p)
    for name in ['ranking_inputs.npz','rankings.npz','frozen_arrays.npz','hierarchy_input.bin','population.json','input_manifest.json']:shutil.copy2(old/name,out/name)
    shutil.copytree(old/'old_tree',out/'old_tree')
    for a,b in [('pool_rankings.npz','prior_pool_rankings.npz'),('selected_sets.npz','prior_selected_sets.npz'),('quality_summary.csv','prior_quality_summary.csv'),('quality_per_view.csv','prior_quality_per_view.csv'),('redundancy_statistics.csv','prior_redundancy.csv')]:shutil.copy2(old/a,out/b)
    original=Path('/home/prmi/zwj/hierarchical-progressive-marginal-20261019')if scene=='fr1_room'else Path('/home/prmi/zwj/hierarchical-xyz-f1f2-mechanism-20261019')/scene
    if scene=='fr1_room':runtime_expected=read(original/'artifact_manifest.json')
    else:runtime_expected={str(Path(k).relative_to(scene)):v['sha256']for k,v in read(original.parent/'bundle_manifest.json')['files'].items()if k.startswith(scene+'/')}
    shutil.copytree(old/'runtime',out/'runtime')
    for lib in(out/'runtime/lib').glob('*'):assert sha(lib)==runtime_expected[str(lib.relative_to(out))]
    for name in ['common.py','evaluate.py','fixed_marginal_reference.py']:
        assert sha(old/name)==manifest[scene+'/'+name]['sha256'];shutil.copy2(old/name,out/name)
    shutil.copy2(old/'audit_worker',out/'audit_worker');assert sha(old/'audit_worker')==prior['source_math']['audit_worker']
    for name in ['baseline.py','coverage.py','fallback.py','quality.py','diagnostics.py']:shutil.copy2(ROOT/name,out/name)
    save(out/'protocol.json',reg);save(out/'prior_checks.json',dict(all_passed=True,source_sha=prior['source_sha'],camera_sha256=prior['camera_sha256'],previous_artifacts={str(old/name):sha(old/name)for name in names},runtime_sha={p.name:sha(p)for p in(out/'runtime/lib').glob('*')},source_math=prior['source_math'],preregistration_sha256=sha(ROOT/'preregistration.json')))
    checks.append(dict(scene=scene,source=str(old),N_candidate=read(out/'population.json')['N_candidate'],all_passed=True))
save(ROOT/'frozen_preparation.json',dict(all_passed=True,checks=checks))
print('PREPARE_FROZEN_PASS',checks,flush=True)

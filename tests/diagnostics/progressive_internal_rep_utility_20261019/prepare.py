"""核对三scene旧冻结SHA，统一命名A/B/C0/P0；不导出贡献或修改相机。"""
from pathlib import Path
import json,hashlib,shutil,csv
import numpy as np
ROOT=Path(__file__).resolve().parent;ROOM=Path('/home/prmi/zwj/hierarchical-progressive-marginal-20261019');XYZ=Path('/home/prmi/zwj/hierarchical-xyz-f1f2-mechanism-20261019')
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
reg=dict(scenes=['fr1_room','fr1_xyz','fr2_xyz'],arms=['A','B','C','P0','S','L','G'],ratios=[.01,.05,.1,.15],
    backend='sparse_hash_target_2048 ONLY',candidate='valid AND observed',maturity=1,K='floor(ratio*N+.5)',
    frozen='grid/h search/origin/3reps/Q=.80/maxL3/radius/kappa/Raw/Bank/Master/TF/cameras/GT/metrics/renderer',
    baseline_gate='full A/B/C/P0 rankings and selected sets exact; rebuilt C binary/tree exact; all baseline per-view and aggregate metrics EXACT; all scenes pass before diagnostic rendering',
    S='stable filter ALL internal_rep entries from complete P0; compact non-internal ranking; prefixes K',
    L='per ratio sequential P0 rank; replace internal not in C TopK by best C-ranked currently unselected and never-used replacement in SAME node.members subtree; otherwise retain original; no GT access',
    G='per ratio remove internal not in C TopK, then global missing C-rank refill to K',
    secondary_Q_probes='L_Q0..3 apply L only to bad internal rows in each fixed Q bin: <.25, .25-.50, .50-.65, .65-.80. Joint rendering effect, not additive per-row utility. No tuning.',
    S_nested=True,L_G_ratio_specific_not_nested=True,
    verdict='SUPPORTED if S and/or same-subtree L consistently restore main failed cases toward C without opposite systematic evidence in successful cases; PARTIAL for only some scenes/ratios; NOT SUPPORTED if no stable restoration. No new dB threshold.',
    no_algorithm_change=True,no_d50_backend=True,raw_contribution_renders=0,optimizer_updates=0,commit=False,stage=False,push=False)
save(ROOT/'preregistration.json',reg)
allchecks={}
for scene in reg['scenes']:
    old=ROOM if scene=='fr1_room'else XYZ/scene;out=ROOT/scene;out.mkdir(exist_ok=False)
    info=read(old/'input_manifest.json');pop=read(old/'population.json');expected={}
    if scene=='fr1_room':
        art=read(ROOM/'artifact_manifest.json');expected={str(ROOM/n):v for n,v in art.items()}
        sources=read(ROOM/'prior_input_manifest.json')['frozen_sources']
        freeze={p:v for p,v in sources.items()if p in [info[k]for k in ['master','tf','config','cameras']]or'/rgb/'in p or p.endswith('/cameras.json')or p.endswith('/view_bank_orb.json')}
        inputbin=ROOM/'fr1_room_input.bin';inputarray=ROOM/'fr1_room_inputs.npz';qfile=ROOM/'progressive_marginal_quality_summary.csv';pvfile=ROOM/'progressive_marginal_per_view.csv'
    else:
        art=read(XYZ/'bundle_manifest.json')['files'];expected={str(XYZ/n):v['sha256']for n,v in art.items()}
        freeze=read(old/'frozen_sources.json');inputbin=old/'hierarchy_input.bin';inputarray=old/'frozen_input_arrays.npz';qfile=old/'quality_summary.csv';pvfile=old/'quality_per_view.csv'
    for path,v in freeze.items():assert sha(path)==v['sha256'],('STOP_FROZEN_SOURCE_SHA',path)
    artifacts=[old/'ranking_inputs.npz',inputbin,inputarray,qfile,pvfile,old/'arm_rankings.npz',old/'new_arm_rankings.npz',old/'structure_run1/sparse_hash_target_2048_tree.csv',old/'structure_run1/sparse_hash_target_2048.bin',old/'structure_run1/sparse_hash_target_2048_telemetry.json']
    for p in artifacts:assert sha(p)==expected[str(p)],('STOP_PRIOR_ARTIFACT_SHA',p)
    for n in ['common.py','evaluate.py','progressive_ranker.py','fixed_marginal_reference.py','audit_worker']:
        assert sha(old/n)==sha(ROOM/n),n
        if str(old/n)in expected:assert sha(old/n)==expected[str(old/n)]
        shutil.copy2(old/n,out/n)
    shutil.copytree(old/'runtime',out/'runtime')
    for source,name in [(inputbin,'hierarchy_input.bin'),(inputarray,'frozen_arrays.npz'),(qfile,'prior_quality_summary.csv'),(pvfile,'prior_quality_per_view.csv')]:shutil.copy2(source,out/name)
    for n in ['ranking_inputs.npz','input_manifest.json','population.json']:shutil.copy2(old/n,out/n)
    (out/'old_tree').mkdir()
    for source,name in [('sparse_hash_target_2048_tree.csv','hierarchy_tree.csv'),('sparse_hash_target_2048.bin','importance.bin'),('sparse_hash_target_2048_telemetry.json','hierarchy_telemetry.json')]:shutil.copy2(old/'structure_run1'/source,out/'old_tree'/name)
    a=dict(np.load(old/'arm_rankings.npz'));b=dict(np.load(old/'new_arm_rankings.npz'));np.savez_compressed(out/'rankings.npz',row_ids=a['row_ids'],A_order=a['A_Raw_order'],A_score=a['A_Raw_score'],B_order=a['B_FixedMarginal_order'],B_score=a['B_FixedMarginal_score'],C0_order=a['C_Hierarchical_order'],C0_score=a['C_Hierarchical_score'],P0_order=b['F2_ProgressiveResidual_order'],P0_score=b['F2_ProgressiveResidual_score'],P0_priority=b['F2_ProgressiveResidual_priority'],old_root=a['root'],old_leaf=a['leaf'],old_rep_mask=a['rep_mask'],old_rep_level=a['rep_level'],fixed_cell=a['fixed_cell'],fixed_keys=a['fixed_keys'])
    for n in ['tree_io.py','baseline.py','construct.py','diagnostics.py','quality.py']:shutil.copy2(ROOT/n,out/n)
    save(out/'protocol.json',reg);save(out/'prior_checks.json',dict(all_passed=True,source_sha=freeze,artifact_sha={str(p):sha(p)for p in artifacts},source_math={n:sha(old/n)for n in ['progressive_ranker.py','fixed_marginal_reference.py','common.py','evaluate.py','audit_worker']},camera_sha256=sha(info['cameras']),preregistration_sha256=sha(ROOT/'preregistration.json')))
    oldQ={(z['arm'],float(z['ratio'])):float(z['psnr'])for z in csv.DictReader(open(qfile))};deltas=[oldQ['F2_ProgressiveResidual',r]-oldQ['C_Hierarchical',r]for r in reg['ratios']]
    allchecks[scene]=dict(N_candidate=pop['N_candidate'],Bank=len(info['bank_frames']),evaluation=len(info['evaluation_frames']),old_progressive_delta_PSNR=deltas,old_C_sha=sha(out/'old_tree/importance.bin'),input_sha=sha(inputbin),source=str(old))
    print('PREPARED_FROZEN',scene,allchecks[scene],flush=True)
save(ROOT/'frozen_preparation.json',dict(all_passed=True,scenes=allchecks))

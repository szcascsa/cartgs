"""只复制并核对既有冻结fixture；不重渲染Raw contribution、不更新Bank或tree。"""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parent
PREV=Path('/home/prmi/zwj/hierarchical-progressive-marginal-20261019')
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
prior=read(PREV/'preregistration.json')
for scene in ['fr1_xyz','fr2_xyz']:
    old=Path('/home/prmi/zwj')/('hierarchical-'+scene.replace('_','-')+'-same-k-20261019')
    out=ROOT/scene;out.mkdir(exist_ok=False)
    for path,v in read(old/'frozen_sources.json').items():assert sha(path)==v['sha256'],('STOP_FROZEN_SOURCE_SHA',path)
    info=read(old/'input_manifest.json');assert sha(info['cameras'])==read(old/'camera_checks.json')['camera_archive_sha256']
    frozen={}
    names=['input_manifest.json','population.json','ranking_inputs.npz','frozen_input_arrays.npz','hierarchy_input.bin','contributions.pt','arm_rankings.npz','frozen_sources.json','camera_checks.json','region_definition.json']
    frozen_manifest=read(old/'bundle_manifest.json')['files']
    for name in names+['structure_run1/sparse_hash_target_2048.bin','structure_run1/sparse_hash_target_2048_tree.csv','structure_run1/sparse_hash_target_2048_telemetry.json','quality_summary.csv','quality_per_view.csv','selected_redundancy.csv','new_arm_rankings.npz']:
        p=old/name;assert sha(p)==frozen_manifest[name]['sha256'],('STOP_PRIOR_ARTIFACT_SHA',name)
        frozen[str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size)
    for name in names:shutil.copy2(old/name,out/name)
    shutil.copytree(old/'structure_run1',out/'structure_run1');shutil.copytree(old/'runtime',out/'runtime')
    for name in ['common.py','evaluate.py','progressive_ranker.py','experiment_data.py','fixed_marginal_reference.py','test_progressive.py']:
        assert sha(old/name)==sha(PREV/name),name
        shutil.copy2(old/name,out/name)
    shutil.copy2(old/'audit_worker',out/'audit_worker')
    for source,dest in [('quality_summary.csv','prior_quality_summary.csv'),('quality_per_view.csv','prior_quality_per_view.csv'),('selected_redundancy.csv','prior_selected_redundancy.csv'),('new_arm_rankings.npz','prior_F3_rankings.npz')]:shutil.copy2(old/source,out/dest)
    for name in ['baseline.py','new_rankings.py','replay_checks.py','diagnostics.py','quality.py','analyze.py']:shutil.copy2(ROOT/name,out/name)
    p=dict(prior);p.update(scene=scene,N_expected=read(old/'population.json')['N_candidate'],baseline=['A_Raw','B_FixedMarginal','C_Hierarchical'],new_arms=['F1_MarginalTail','F2_ProgressiveResidual','F3_ProgressiveMarginal'],render_new_arms=['F1_MarginalTail','F2_ProgressiveResidual'],baseline_reproduction_arms=['A_Raw','B_FixedMarginal','C_Hierarchical','F3_ProgressiveMarginal'],
        baseline_gate='all A/B/C/F3 complete rankings exact; rerender all old arm/ratio/view cases; per-view and aggregate max abs<=1e-6; then generate F1/F2',
        evaluation=f"same {len(info['evaluation_frames'])} fixed non-Bank archived views; same {len(info['bank_frames'])} Bank frozen contribution; no export/update",
        previous_experiment=str(old),new_raw_contribution_renders=0)
    save(out/'preregistration.json',p);save(out/'protocol.json',p);save(out/'prior_artifact_checks.json',dict(all_passed=True,files=frozen))
    save(out/'preparation.json',dict(all_passed=True,source_sha_exact=True,prior_artifact_sha_exact=True,camera_archive_exact=True,prior=str(old),baseline_C_executable=str(old/'build/hierarchy_fixture'),preregistration_sha256=sha(out/'preregistration.json'),quality_started=False))
    print('PREPARED',scene,p['N_expected'],len(info['bank_frames']),len(info['evaluation_frames']),flush=True)

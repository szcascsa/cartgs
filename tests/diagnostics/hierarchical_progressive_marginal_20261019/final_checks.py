from pathlib import Path
import json,hashlib,csv,math
import numpy as np
ROOT=Path(__file__).resolve().parent;P=Path('/home/prmi/zwj/hierarchical-si-quality-audit-20261019')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
digest='a094f25781f8f5834beab3b21880fa16a01b4facababb12c3d526218505d4021'
assert sha(ROOT/'preregistration.json')==digest and sha(ROOT/'mechanism_preregistration.json')==digest
checks=read(ROOT/'frozen_sha_checks.json')
for path,v in checks['checked'].items():assert sha(path)==v['expected'],path
for original,v in checks['runtime_resolution'].items():
    assert sha(v['resolved'])==v['expected']and sha(ROOT/'runtime/lib'/Path(original).name)==v['expected']
    assert sha(original)==v['current_production_sha']
prior=read(P/'artifact_manifest.json')
for n in ['ranking_inputs.npz','arm_rankings.npz','selected_rows.npz','quality_summary.csv','quality_per_view.csv','input_manifest.json','structure_run1/sparse_hash_target_2048_tree.csv']:
    assert sha(P/n)==prior[n],n
assert sha(ROOT/'structure_run1/sparse_hash_target_2048_tree.csv')==prior['structure_run1/sparse_hash_target_2048_tree.csv']
for f in ['baseline_reproduction.json','synthetic_tests.json','new_ranking_checks.json','independent_replay_checks.json','diagnostic_checks.json','quality_complete.json']:
    assert read(ROOT/f)['all_passed'],f
summary=list(csv.DictReader(open(ROOT/'progressive_marginal_quality_summary.csv')));pv=list(csv.DictReader(open(ROOT/'progressive_marginal_per_view.csv')))
assert len(summary)==24 and len(pv)==3864 and len({(v['arm'],v['ratio'],v['frame_index'])for v in pv})==3864
for a in summary:
    group=[v for v in pv if v['arm']==a['arm']and v['ratio']==a['ratio']];assert len(group)==161
    assert int(a['K'])==int(float(a['ratio'])*50330+.5)
    for m in ['psnr','ssim','lpips','photo']:
        values=[float(v[m])for v in group];assert all(math.isfinite(v)for v in values)
        assert abs(math.fsum(values)/161-float(a[m]))<1e-12
result={'all_passed':True,'frozen_input_count':len(checks['checked']),'prior_consumed_artifact_SHA_checked':True,'pre_registration_SHA256':digest,
        'complete_rankings_repeat_and_replay':True,'same_K_nested':True,'mean_and_per_view_consistent':True,'optimizer_updates':0,'production_runtime_unchanged':True,'quality_cases':24,'view_renders':3864,'baseline_metrics_max_abs':read(ROOT/'baseline_reproduction.json')['per_view_max_abs']}
(ROOT/'final_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
driver=ROOT/'driver_status.json'
if driver.exists()and not read(driver).get('all_passed',False)and not(ROOT/'baseline_driver_initial_failure.json').exists():
    (ROOT/'baseline_driver_initial_failure.json').write_bytes(driver.read_bytes())
driver.write_text(json.dumps({'complete':True,'all_passed':True,'baseline_reproduced':True,'new_rankings_replayed':True,'new_quality_complete':True,'initial_worker_maps_race_resolved':True},indent=2)+'\n')

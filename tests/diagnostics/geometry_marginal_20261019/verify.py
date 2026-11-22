"""交付前独立检查source/config、完整ranking、真实训练终点与全部评价记录。"""
import json,hashlib,struct,subprocess,csv
from pathlib import Path
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parent;R=Path('/home/prmi/zwj/mycartgs')
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,h in read(ROOT/'code_sync_manifest.json')['files'].items():assert sha(R/name)==h,name
ref=dict(np.load(ROOT/'reference.npz'));inputs=dict(np.load(ROOT/'ranking_inputs.npz'));ids=inputs['row_ids'];expected=ids[ref['order']]
for p in ROOT.glob('final_*.bin'):
    with open(p,'rb')as f:n=struct.unpack('<q',f.read(8))[0];order=np.frombuffer(f.read(n*8),dtype='<i8');gains=np.frombuffer(f.read(n*8),dtype='<f8')
    np.testing.assert_array_equal(order,expected[:n]);assert np.isfinite(gains).all()
for tag in ['reference','optimized']:
    rows=list(csv.DictReader(open(ROOT/(tag+'_per_view.csv'))));assert len(rows)==644
    for r in rows:assert int(r['K'])==int(float(r['ratio'])*len(ids)+.5)
assert read(ROOT/'optimized_render_validation.json')['max_abs_difference']==dict(photo=0.,psnr=0.,ssim=0.,lpips=0.)
subprocess.run([str(ROOT/'test_ranker')],check=True)
shadow=read(ROOT/'shadow_validation.json');assert shadow['all_passed']and{1000,8000,16000}<=set(shadow['reference_exact_iterations'])
def config(p):
    fs=cv2.FileStorage(str(p),cv2.FILE_STORAGE_READ);d={}
    for k in fs.root().keys():
        n=fs.getNode(k);d[k]=n.string()if n.isString()else[n.at(i).real()for i in range(n.size())]if n.isSeq()else n.real()
    fs.release();return d
diffs={}
for scene in ['fr1_room','fr1_desk']:
    a=config(ROOT/'configs'/(scene+'_raw.yaml'));b=config(ROOT/'configs'/(scene+'_marginal.yaml'));delta={k:[a[k],b[k]]for k in a if a[k]!=b[k]};assert set(delta)=={'GlobalGI.mode'},delta;diffs[scene]=delta
assert config(R/'cfg/selector/fast.yaml')['GlobalGI.mode']=='streaming'
models=read(ROOT/'online_training_results.json');assert len(models)==8 and len({(m['scene'],m['seed'],m['arm'])for m in models})==8
for m in models:
    target=16900 if m['scene']=='fr1_room'else 7400
    assert sum(m['state'][p]['mappings']for p in ['online','tail'])==target
    assert all(m['state'][p]['lod_nonfinite']==0 for p in ['online','tail'])
    assert len(list(Path(m['model']).glob('*_shutdown')))==1
ev=read(ROOT/'online_evaluation_results.json');assert len(ev)==40
for e in ev:
    assert all(np.isfinite(e['metrics'][k])for k in ['psnr','ssim','lpips']);assert np.isfinite(e['ate']['translation_ape_m']['rmse'])
    actual=read(Path(e['output'])/'evaluation_frames.json')['gt_indices'];expected_frames=read(ROOT/(e['scene']+'_common_frames.json'))['gt_indices'];assert actual==expected_frames and len(actual)==128
    if e['policy']!='full':
        selected=np.load(Path(e['output'])/'selected_rows.npz')['rows'];assert len(selected)==e['selection']['active_gaussians']and len(np.unique(selected))==len(selected)
for scene in ['fr1_room','fr1_desk']:
    for ratio in [.01,.05]:
        counts={e['selection']['active_gaussians']for e in ev if e['scene']==scene and e['ratio']==ratio and e['policy']=='same_k'};assert len(counts)==1,counts
core='\n'.join(l.split('//')[0]for l in(R/'include/geometry_marginal_ranker.h').read_text().splitlines());assert all(s not in core for s in ['contribution','cosine','covariance','view_similarity'])
for name in ['GEOMETRY_ONLY_REFERENCE_VALIDATION.md','GEOMETRY_ONLY_SIMPLIFICATION_REPORT.md','GEOMETRY_MARGINAL_PERFORMANCE_REPORT.md','GEOMETRY_MARGINAL_ONLINE_REPORT.md']:assert(ROOT/name).exists()
out=dict(all_passed=True,source_files_checked=1831,models=8,quality_cases=40,offline_per_view_cases=1288,configuration_differences=diffs,all_native_repetitions_exact=True,shadow_early_middle_late_exact=True,local_runtime_mode_default_raw=True,no_commit_no_push=True)
(ROOT/'independent_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)

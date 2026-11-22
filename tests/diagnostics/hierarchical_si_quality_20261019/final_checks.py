"""结束时复核冻结输入、原始CSV、ranking和最终判读；不渲染。"""
from pathlib import Path
import hashlib,json,csv,platform
import numpy as np
ROOT=Path(__file__).resolve().parent
S=Path('/home/prmi/zwj/redundancy-si-ranking-audit-20261019')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert sha(ROOT/'preregistration.json')=='ebb36b8fdb16b0583e59ffd27c92b78127c7565127e24d3d23c3a2e374bfc6cb'
frozen=read(ROOT/'frozen_sha_checks.json')
for path,v in frozen['checked'].items():assert sha(path)==v['expected'],path
for original,v in frozen['runtime_resolution'].items():
    assert sha(v['resolved'])==v['expected']
    assert sha(ROOT/'runtime/lib'/Path(original).name)==v['expected']
    assert sha(original)==v['current_production_sha'],'Production library changed during audit'
old=read(S/'audit_manifest.json');assert sha(S/'rankings.npz')==old['rankings_sha']
assert read(ROOT/'input_manifest.json')==old['inputs']
raw=list(csv.DictReader(open(ROOT/'quality_per_view.csv')));means=list(csv.DictReader(open(ROOT/'quality_summary.csv')))
assert len(raw)==3220 and len(means)==20
assert len({(a['arm'],a['ratio'],a['frame_index'])for a in raw})==3220
frames=read(ROOT/'input_manifest.json')['evaluation_frames'];N=50330;metric=['psnr','ssim','lpips','photo']
for summary in means:
    group=[a for a in raw if a['arm']==summary['arm']and a['ratio']==summary['ratio']]
    assert len(group)==161 and sorted(int(a['frame_index'])for a in group)==sorted(frames)
    assert int(summary['K'])==int(float(summary['ratio'])*N+.5)
    for m in metric:
        values=np.array([float(a[m])for a in group]);assert np.isfinite(values).all()
        assert abs(values.mean()-float(summary[m]))<1e-12
r=np.load(ROOT/'arm_rankings.npz');selected=np.load(ROOT/'selected_rows.npz');arms=read(ROOT/'preregistration.json')['primary']+read(ROOT/'preregistration.json')['diagnostic']
for a in arms:
    order=r[a+'_order'];np.testing.assert_array_equal(np.sort(order),np.arange(N));previous=set()
    for ratio in [.01,.05,.1,.15]:
        K=int(ratio*N+.5);ids=r['row_ids'][order[:K]];np.testing.assert_array_equal(np.sort(ids),selected[a+'_'+str(ratio)])
        assert len(ids)==K and previous<=set(ids);previous=set(ids)
assert read(ROOT/'baseline_reproduction.json')['all_passed'];assert read(ROOT/'frozen_tensor_checks.json')['ok']
decision=read(ROOT/'mechanism_cases.json');assert decision['primary_decision']=='QUALITY INCONCLUSIVE'and not decision['Strong_GO']
assert decision['mixed_ratios']==[.1,.15]
result={'all_passed':True,'pre_registration_unchanged':True,'frozen_sources_unchanged':len(frozen['checked']),'historical_rankings_SHA_exact':True,'input_manifest_exact':True,
    'mean_per_view_consistency':True,'K_and_nested':True,'frozen_worker_tensors':True,'production_libraries_unchanged':True,
    'verdict_matches_original_user_mixed_direction_clause':True,'decision':decision['primary_decision'],'python':platform.python_version(),'numpy':np.__version__}
(ROOT/'final_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)

import hashlib,json,struct,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent/'mycartgs';read=lambda p:json.loads(Path(p).read_text())
for name,h in read(ROOT/'code_sync_manifest.json')['files'].items():assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==h,name
x=dict(np.load(ROOT/'ranking_inputs.npz'));ref=dict(np.load(ROOT/'reference.npz'));expected=x['row_ids'][ref['order']];cases=0
for path in ROOT.glob('final_*.bin'):
    with open(path,'rb')as f:n=struct.unpack('<q',f.read(8))[0];rows=np.frombuffer(f.read(n*8),dtype='<i8')
    np.testing.assert_array_equal(rows,expected[:n]);cases+=1
quality=read(ROOT/'online_evaluation_results.json');assert len(quality)==40
for e in quality:
    # Remote paths are resolved through their audit-root suffix, never accessed locally.
    out=ROOT/e['output'].split('/geometry-marginal-online-20261019/',1)[1]
    for name in ['psnr','ssim','lpips']:
        values=np.loadtxt(out/(name+'.txt'));assert len(values)==128 and np.isfinite(values).all();assert abs(float(np.mean(values))-e['metrics'][name])<1e-10
    assert read(out/'evaluation_frames.json')['gt_indices']==read(ROOT/(e['scene']+'_common_frames.json'))['gt_indices']
    if e['policy']!='full':assert len(np.load(out/'selected_rows.npz')['rows'])==e['selection']['active_gaussians']
assert read(ROOT/'reference_validation.json')['quality_max_abs']==dict(photo=0.,psnr=0.,ssim=0.,lpips=0.)
assert read(ROOT/'optimized_render_validation.json')['max_abs_difference']==dict(photo=0.,psnr=0.,ssim=0.,lpips=0.)
assert read(ROOT/'independent_verification.json')['all_passed']
out=dict(all_passed=True,source_hashes=1831,native_repeated_rankings=cases,online_quality_cases=40,raw_per_view_metrics_reaggregated=15360,common_frames_exact=True,selected_count_exact=True,offline_render_exact=True)
(ROOT/'independent_local_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

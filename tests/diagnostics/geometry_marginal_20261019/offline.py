"""先复现，再profile/编译exact CPU算法；原renderer不重新编译。"""
import os,subprocess,csv,struct
from common import *
from evaluate import evaluate_arms,write_csv
S=Path('/home/prmi/zwj/redundancy-si-ranking-audit-20261019');C=Path('/home/prmi/zwj/covariance-redundancy-audit-20261019')
for path,v in read(S/'audit_manifest.json')['frozen_sources'].items():assert sha(path)==v['sha256'],path
import shutil
for n in ['input_manifest.json','population.json','ranking_inputs.npz','audit_worker']:shutil.copy2(S/n,ROOT/n)
save(ROOT/'protocol.json',dict(ratios=[.01,.05,.1,.15],optimizer_updates=0))
subprocess.run([os.sys.executable,str(ROOT/'reference.py')],check=True)
x=dict(np.load(ROOT/'ranking_inputs.npz'));rr=dict(np.load(ROOT/'reference.npz'));torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
scene=Scene()
try:
    metrics=Metrics(tensors(read(ROOT/'input_manifest.json')['cameras'])['gt']);records,quality=evaluate_arms(scene,metrics,{'GeometryOnlyReference':rr['order']},'reference')
finally:scene.close()
old={(float(v['ratio']),int(v['frame_index'])):v for v in csv.DictReader(open(C/'metric_per_view.csv'))if v['arm']=='IsoOnly'}
diff={m:max(abs(float(v[m])-float(old[(float(v['ratio']),int(v['frame_index']))][m]))for v in records)for m in ['photo','psnr','ssim','lpips']};assert all(t==0 for t in diff.values()),diff
save(ROOT/'reference_validation.json',dict(all_passed=True,ranking_exact=True,quality_max_abs=diff,candidates=len(x['row_ids'])))
(ROOT/'GEOMETRY_ONLY_REFERENCE_VALIDATION.md').write_text('# Geometry-only Reference复现\n\n完整ranking、selected set与旧IsoOnly exact相同；644个view-ratio四项quality最大差均为0。候选50330，maturity=1、原h8、原TF、原161个eval views，未重算contribution。\n')
print('QUALITY_REFERENCE_EXACT',diff,flush=True)
import time
print('AWAIT_PROFILE_REVIEW',flush=True)
while not(ROOT/'profile_review.json').exists():time.sleep(2)
with open(ROOT/'ranking_input.bin','wb')as f:
    f.write(struct.pack('<qd',len(x['row_ids']),read(ROOT/'population.json')['h']))
    for i,row in enumerate(x['row_ids']):f.write(struct.pack('<5dq',*x['xyz'][i],x['radius'][i],x['raw_I'][i],int(row)))
subprocess.run(['/usr/bin/g++-9','-std=c++17','-O3','-DNDEBUG','-o',str(ROOT/'rank_cli'),str(ROOT/'rank_cli.cpp')],check=True)
N=len(x['row_ids']);expected=x['row_ids'][rr['order']];profiles=[]
for mode,limit in [('reference',N),('global',int(.15*N+.5)),('optimized',int(.15*N+.5)),('optimized_full',N)]:
    for repeat in range(5):
        tag=mode+'_'+str(repeat);subprocess.run([str(ROOT/'rank_cli'),str(ROOT/'ranking_input.bin'),str(ROOT/tag),mode if mode!='optimized_full'else'optimized',str(limit)],check=True)
        with open(ROOT/(tag+'.bin'),'rb')as f:
            size=struct.unpack('<q',f.read(8))[0];order=np.frombuffer(f.read(size*8),dtype='<i8');gains=np.frombuffer(f.read(size*8),dtype='<f8')
        np.testing.assert_array_equal(order,expected[:len(order)]);assert np.isfinite(gains).all()
        profile=read(ROOT/(tag+'.json'));profile.update(mode=mode,repeat=repeat);profiles.append(profile)
    print('EXACT_OPTIMIZATION_PASS',mode,flush=True)
save(ROOT/'offline_profiles.json',profiles)
save(ROOT/'optimized_exact_validation.json',dict(all_passed=True,repeats=5,full_order_exact=True,prefix_1_5_10_15_exact=True,quality_identical_by_exact_rows_and_frozen_renderer=True,approximation=False))

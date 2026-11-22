"""先重渲染A/B复现旧质量，gate通过才渲染C/D/E；全程固定TF。"""
from common import *
from evaluate import evaluate_arms,write_csv
import csv,time
assert read(ROOT/'preparation.json')['all_passed'] and read(ROOT/'ranking_sanity.json')['all_passed']
os.environ['LD_LIBRARY_PATH']=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
rank=dict(np.load(ROOT/'arm_rankings.npz'));info=read(ROOT/'input_manifest.json')
S=Path('/home/prmi/zwj/redundancy-si-ranking-audit-20261019');G=Path('/home/prmi/zwj/geometry-marginal-online-20261019')
scene=Scene();start=time.perf_counter()
try:
    assert scene.call({'method':'verify'})['maturity_all_one']
    maps=Path(f'/proc/{scene.proc.pid}/maps').read_text()
    for name in ['libgaussian_mapper.so','libcuda_rasterizer.so']:
        assert str(ROOT/'runtime/lib'/name)in maps,name
        expected=read(S/'audit_manifest.json')['frozen_sources']['/home/prmi/zwj/mycartgs/lib/'+name]['sha256']
        assert sha(ROOT/'runtime/lib'/name)==expected
    (ROOT/'actual_worker_maps.txt').write_text(maps)
    metrics=Metrics(tensors(info['cameras'])['gt'])
    records,summary=evaluate_arms(scene,metrics,{a:rank[a+'_order']for a in ['A_Raw','B_FixedMarginal']},'baseline')
    old_raw={(float(r['ratio']),int(r['frame_index'])):r for r in csv.DictReader(open(S/'per_view_quality.csv'))if r['arm']=='Raw'and r['TF']=='True'}
    old_b={(float(r['ratio']),int(r['frame_index'])):r for r in csv.DictReader(open(G/'reference_per_view.csv'))}
    differences={}
    for arm,old in [('A_Raw',old_raw),('B_FixedMarginal',old_b)]:
        selected=[r for r in records if r['arm']==arm];assert len(selected)==644
        differences[arm]={m:max(abs(float(r[m])-float(old[(float(r['ratio']),int(r['frame_index']))][m]))for r in selected)for m in ['psnr','ssim','lpips','photo']}
    ok=all(v<=1e-6 for d in differences.values()for v in d.values())
    save(ROOT/'baseline_reproduction.json',{'all_passed':ok,'max_abs_per_view':differences,'tolerance':1e-6,'views_per_ratio':161,'K_exact':True,'B_complete_ranking_exact':True,'actual_frozen_libraries_verified':True})
    if not ok:raise RuntimeError('STOP_BASELINE_QUALITY_REPRODUCTION_MISMATCH '+repr(differences))
    print('BASELINE_REPRODUCTION_PASS',differences,flush=True)
    new_records,new_summary=evaluate_arms(scene,metrics,{a:rank[a+'_order']for a in ['C_Hierarchical','D_RepresentativeOnly','E_L1Only']},'hierarchy')
    records+=new_records;summary+=new_summary
    write_csv('quality_per_view.csv',records);write_csv('quality_summary.csv',summary)
    assert len(summary)==20 and len(records)==3220
    save(ROOT/'quality_complete.json',{'all_passed':True,'arms':5,'cases':20,'view_renders':3220,'real_render':True,'TF':True,'same_K':True,'optimizer_updates':0,'runtime_seconds':time.perf_counter()-start})
finally:
    scene.close()

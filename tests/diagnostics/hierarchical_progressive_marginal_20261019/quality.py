from common import *
from evaluate import evaluate_arms,write_csv
import csv,time
assert read(ROOT/'baseline_reproduction.json')['all_passed']
assert read(ROOT/'new_ranking_checks.json')['all_passed'] and read(ROOT/'synthetic_tests.json')['all_passed']
os.environ['LD_LIBRARY_PATH']=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
r=np.load(ROOT/'new_arm_rankings.npz');scene=Scene();start=time.perf_counter()
try:
    assert scene.call({'method':'verify'})['maturity_all_one']
    metrics=Metrics(tensors(read(ROOT/'input_manifest.json')['cameras'])['gt'])
    records,summary=evaluate_arms(scene,metrics,{a:r[a+'_order']for a in read(ROOT/'preregistration.json')['new_arms']},'new')
finally:scene.close()
base=list(csv.DictReader(open(ROOT/'baseline_per_view.csv')));baseq=list(csv.DictReader(open(ROOT/'baseline_quality.csv')))
write_csv('progressive_marginal_per_view.csv',base+records);write_csv('progressive_marginal_quality_summary.csv',baseq+summary)
assert len(base+records)==3864 and len(baseq+summary)==24
save(ROOT/'quality_complete.json',{'all_passed':True,'arm_ratio_cases':24,'view_renders':3864,'baseline_cases':12,'new_cases':12,'new_arm_runtime_seconds':time.perf_counter()-start,'optimizer_updates':0,'quality_frozen':True})
print('QUALITY_ALL_COMPLETE',flush=True)

from common import *
from evaluate import evaluate_arms,write_csv
import csv,time
assert read(ROOT/'baseline_reproduction.json')['all_passed']and read(ROOT/'new_ranking_checks.json')['all_passed']and read(ROOT/'independent_replay_checks.json')['all_passed']
os.environ['LD_LIBRARY_PATH']=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
r=np.load(ROOT/'new_arm_rankings.npz');scene=Scene();start=time.perf_counter()
try:
    assert scene.call(dict(method='verify'))['maturity_all_one']
    metrics=Metrics(tensors(read(ROOT/'input_manifest.json')['cameras'])['gt'])
    records,summary=evaluate_arms(scene,metrics,{a:r[a+'_order']for a in read(ROOT/'preregistration.json')['render_new_arms']},'new')
finally:scene.close()
base=list(csv.DictReader(open(ROOT/'baseline_per_view.csv')));baseq=list(csv.DictReader(open(ROOT/'baseline_quality.csv')))
write_csv('quality_per_view.csv',base+records);write_csv('quality_summary.csv',baseq+summary)
views=len(read(ROOT/'input_manifest.json')['evaluation_frames']);assert len(base+records)==24*views and len(baseq+summary)==24
save(ROOT/'quality_complete.json',dict(all_passed=True,arm_ratio_cases=24,view_renders=24*views,baseline_rerenders=len(base),new_F1_F2_renders=len(records),new_runtime_seconds=time.perf_counter()-start,optimizer_updates=0))
print('ALL_SIX_ARMS_QUALITY_COMPLETE',flush=True)

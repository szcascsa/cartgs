from common import *
from evaluate import evaluate_arms,write_csv
import csv
assert read(ROOT.parent/'baseline_all_passed.json')['all_passed']and read(ROOT/'new_ranking_checks.json')['all_passed']and read(ROOT/'independent_replay_checks.json')['all_passed']
os.environ['LD_LIBRARY_PATH']=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
b=np.load(ROOT/'rankings.npz');scene=Scene()
try:
    assert scene.call(dict(method='verify'))['maturity_all_one'];m=Metrics(tensors(read(ROOT/'input_manifest.json')['cameras'])['gt'])
    rows,q=evaluate_arms(scene,m,{a:b[a+'_order']for a in ['C32','P32']},'new')
finally:scene.close()
old=list(csv.DictReader(open(ROOT/'baseline_per_view.csv')));oldq=list(csv.DictReader(open(ROOT/'baseline_quality.csv')))
write_csv('quality_per_view.csv',old+rows);write_csv('quality_summary.csv',oldq+q)
save(ROOT/'quality_complete.json',dict(all_passed=True,baseline_rerenders=len(old),new_C32_P32_renders=len(rows),total_renders=len(old)+len(rows),cases=24,optimizer_updates=0,no_F1_F3_renders=True))

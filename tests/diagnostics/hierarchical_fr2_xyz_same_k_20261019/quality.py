from common import *
from evaluate import evaluate_arms
import time
assert read(ROOT/'ranking_sanity.json')['all_passed']and read(ROOT/'independent_replay_checks.json')['all_passed']
os.environ['LD_LIBRARY_PATH']=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
ranking=np.load(ROOT/'arm_rankings.npz');scene=Scene();start=time.perf_counter()
try:
    assert scene.call(dict(method='verify'))['maturity_all_one']
    metrics=Metrics(tensors(read(ROOT/'input_manifest.json')['cameras'])['gt'])
    rows,summary=evaluate_arms(scene,metrics,{a:ranking[a+'_order']for a in read(ROOT/'protocol.json')['arms']},'quality')
finally:scene.close()
views=len(read(ROOT/'input_manifest.json')['evaluation_frames']);assert len(rows)==16*views and len(summary)==16
save(ROOT/'quality_complete.json',dict(all_passed=True,cases=16,views_per_case=views,view_renders=len(rows),runtime_seconds=time.perf_counter()-start,optimizer_updates=0,TF_frozen=True))
print('QUALITY_ALL_COMPLETE',flush=True)

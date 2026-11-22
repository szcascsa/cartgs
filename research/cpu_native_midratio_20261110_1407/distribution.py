"""计时外记录精确边界桶规模和饱和回退，解释不同地图收益。"""
from common import *
import gc
from selector import slot_eligibility
torch.set_num_threads(1);torch.set_num_interop_threads(1)
rows=[]
with torch.no_grad():
    for spec in SCENES:
        source=load_scene(spec['slug']);engine=native(source,'combined');eligible=slot_eligibility(source['model'],source['pc'].get_opacity).numpy();records=source['records']
        for frame in [int(f*(len(records)-1)) for f in [.19,.51,.83]]:
            cam=camera(records[frame])
            for ratio in [.1,.2,.3,.4,.5]:
                _,ids,info=request(engine,cam,'local'+str(ratio));diagnostics=engine.native.diagnostics();all_scores=diagnostics[2].numpy();indices=diagnostics[1].numpy();scores=all_scores[eligible[indices]];M=len(indices);K=len(ids)
                hist=np.bincount(np.minimum(255,(scores*256).astype(np.int32)),minlength=256);bucket=-1;above=0
                if K:
                    for b in range(255,-1,-1):
                        if above+hist[b]>=K:bucket=b;break
                        above+=int(hist[b])
                size=int(hist[bucket]) if bucket>=0 else 0
                rows.append(dict(scene=spec['id'],frame=frame,ratio=ratio,N=engine.total,M=M,K=K,eligible=len(scores),bucket=bucket,boundary_size=size,boundary_fraction=size/max(1,M),above=above,
                    histogram_active=bool(M>=2048 and K),saturation_fallback=bool(M>=2048 and K and size>M//4),score_zero=int((scores==0).sum()),score_one=int((scores==1).sum())))
        del engine,source;gc.collect();print('SELECTION_DISTRIBUTION',spec['id'],len(rows),flush=True)
write_csv(HERE/'reports/CPU_SELECTION_DISTRIBUTIONS.csv',rows);save(HERE/'reports/DISTRIBUTION_SUMMARY.json',dict(passed=True,rows=len(rows),diagnostic_only=True,cuda_initialized=torch.cuda.is_initialized()))

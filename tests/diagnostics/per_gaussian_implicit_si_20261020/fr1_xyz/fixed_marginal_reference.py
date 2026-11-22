"""GeometryOnlyReference：仅xyz/radius/Raw I，不接收view vector。"""
import time,resource,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def build(x,radius,w,ids,h):
    t=time.perf_counter();keys,cell=np.unique(np.floor(x.astype(float)/h).astype(np.int64),axis=0,return_inverse=True);grouping=time.perf_counter()-t
    t=time.perf_counter();groups=[np.flatnonzero(cell==c)for c in range(len(keys))];membership=time.perf_counter()-t
    t=time.perf_counter();gain=np.zeros(len(w));local=np.zeros(len(w),np.int64);stats=[];allocation=time.perf_counter()-t
    for c,g in enumerate(groups):
        start=time.perf_counter();cpu=time.process_time();z=x[g].astype(float);rr=radius[g].astype(float);d=((z[:,None]-z[None,:])**2).sum(2);sigma=.5*(rr[:,None]+rr[None,:]);k=np.exp(-d/(2*sigma*sigma+1e-12));np.fill_diagonal(k,1);sim_ms=1000*(time.perf_counter()-start)
        cover=np.zeros(len(g));taken=np.zeros(len(g),bool);score_time=0.;cover_time=0.;init=0.
        for step in range(len(g)):
            s=time.perf_counter();values=(np.maximum(k-cover[None,:],0)*w[g][None,:]).sum(1);values[taken]=-np.inf;j=np.lexsort((ids[g],-values))[0];dt=time.perf_counter()-s
            if step==0:init=dt
            else:score_time+=dt
            gain[g[j]]=values[j];local[g[j]]=step+1;s=time.perf_counter();cover=np.maximum(cover,k[j]);taken[j]=True;cover_time+=time.perf_counter()-s
        stats.append(dict(cell=c,n=len(g),similarity_ms=sim_ms,initial_gain_ms=init*1000,rescore_ms=score_time*1000,coverage_ms=cover_time*1000,total_ms=(time.perf_counter()-start)*1000,cpu_ms=(time.process_time()-cpu)*1000,pair_bytes=k.nbytes,gain_recomputations=len(g)**2))
    s=time.perf_counter();order=np.lexsort((ids,cell,-gain));merge=time.perf_counter()-s
    return dict(order=order,gain=gain,rank_local=local,cell=cell,keys=keys),dict(grid_ms=grouping*1000,membership_ms=membership*1000,output_allocation_ms=allocation*1000,merge_ms=merge*1000,cells=stats)
if __name__=='__main__':
    import csv
    S=Path('/home/prmi/zwj/redundancy-si-ranking-audit-20261019');x=dict(np.load(S/'ranking_inputs.npz'));p=json.loads((S/'population.json').read_text());old=dict(np.load(S/'rankings.npz'));start=time.perf_counter();cpu=time.process_time();r,profile=build(x['xyz'],x['radius'],x['raw_I'],x['row_ids'],p['h'])
    for k in r:np.testing.assert_array_equal(r[k],old['GeometryOnly8_'+k])
    profile.update(wall_ms=(time.perf_counter()-start)*1000,cpu_ms=(time.process_time()-cpu)*1000,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu_ms=0,gpu_kernels=0,cpu_gpu_copies=0,arrays_allocated_known=sum(10+5*z['n']for z in profile['cells']),allocation_note='explicit NumPy array sites/count; allocator-level count is not claimed')
    (ROOT/'reference_profile.json').write_text(json.dumps(profile,indent=2));np.savez_compressed(ROOT/'reference.npz',**r)
    with open(ROOT/'reference_cells.csv','w',newline='')as f:v=csv.DictWriter(f,fieldnames=list(profile['cells'][0]));v.writeheader();v.writerows(profile['cells'])
    print('REFERENCE_EXACT',len(x['row_ids']),profile['wall_ms'],flush=True)

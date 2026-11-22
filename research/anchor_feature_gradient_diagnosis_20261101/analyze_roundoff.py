"""用FP64求和和FP32误差界区分原子累加舍入、映射错误及遗漏梯度。"""
from common import *
import csv,json,itertools,collections,time
import torch,numpy as np

def write(name,rows):
    with (R/name).open('w',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
def load(path):
    module=torch.jit.load(str(path),map_location='cpu')
    return {n:t.detach().numpy() for n,t in module.named_buffers()}
def attr(path,key):
    return getattr(torch.jit.load(str(path),map_location='cpu'),key).detach().numpy()
def ulps(a,b):
    def order(x):
        u=x.astype(np.float32,copy=False).view(np.uint32).astype(np.int64)
        return np.where(u>=0x80000000,0xFFFFFFFF-u,u+0x80000000)
    return abs(order(a)-order(b))
torch.set_num_threads(2)
ratio=[.01,.05,.1,.2,.3,.4,.025,.075,.15,.25,.35]
occupancy_rows=[];bounds=[];permutations=[];bias=[]
u=2.**-24
for j,r in enumerate(ratio):
    path=R/f'snapshots/d0_r{j}_fused.pt';module=torch.jit.load(str(path),map_location='cpu')
    owners=module.owners.numpy().astype(np.int64);incoming=getattr(module,'incoming_a_0').numpy();a=getattr(module,'scatter_a_0').numpy();b=getattr(module,'scatter_b_0').numpy()
    count=np.bincount(owners,minlength=len(a));ref=np.zeros_like(a,dtype=np.float64);absolute=np.zeros_like(ref)
    np.add.at(ref,owners,incoming.astype(np.float64));np.add.at(absolute,owners,np.abs(incoming.astype(np.float64)))
    n=count[:,None];bound=(n*u/(1-n*u))*absolute;bound=np.maximum(bound,np.finfo(np.float32).tiny)
    da=a.astype(np.float64)-ref;db=b.astype(np.float64)-ref;different=a.view(np.int32)!=b.view(np.int32)
    finite=np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(incoming).all()
    bounds.append(dict(ratio=r,K=len(owners),anchors=len(a),max_occupancy=int(count.max()),finite=finite,
        baseline_bound_violations=int((abs(da)>bound+1e-15).sum()),candidate_bound_violations=int((abs(db)>bound+1e-15).sum()),
        baseline_max_error_over_bound=float((abs(da)/bound).max()),candidate_max_error_over_bound=float((abs(db)/bound).max()),
        baseline_relative_l2=float(np.linalg.norm(da)/max(1e-30,np.linalg.norm(ref))),candidate_relative_l2=float(np.linalg.norm(db)/max(1e-30,np.linalg.norm(ref))),
        baseline_max_abs_vs_fp64=float(abs(da).max()),candidate_max_abs_vs_fp64=float(abs(db).max()),
        baseline_candidate_max_abs=float(abs(a-b).max()),gather_input_bits_equal=np.array_equal(getattr(module,'incoming_a_0').numpy().view(np.int32),getattr(module,'incoming_b_0').numpy().view(np.int32))))
    for group,mask in [('0',count==0),('1',count==1),('2',count==2),('3',count==3),('4-10',(count>=4)&(count<=10)),('gt10',count>10)]:
        values=different[mask];ad=abs(a[mask].astype(np.float64)-b[mask].astype(np.float64));uu=ulps(a[mask],b[mask])
        occupancy_rows.append(dict(ratio=r,occupancy=group,anchor_rows=int(mask.sum()),elements=values.size,different_elements=int(values.sum()),
            different_pct=100*float(values.mean()) if values.size else 0.,max_abs=float(ad.max()) if values.size else 0.,
            max_ulp=int(uu.max()) if values.size else 0,p99_ulp_changed=float(np.percentile(uu[values],99)) if values.any() else 0.))
    chosen=[]
    for row,col in np.argwhere(different):
        if 3<=count[row]<=7 and len(chosen)<4:chosen.append((int(row),int(col)))
        if len(chosen)==4:break
    for row,col in chosen:
        values=incoming[owners==row,col];possible={};match_a=match_b=None
        for order in itertools.permutations(range(len(values))):
            value=np.float32(0.)
            for index in order:value=np.float32(value+values[index])
            code=int(value.view(np.int32));possible[code]=order
            if code==int(a[row,col].view(np.int32)):match_a=order
            if code==int(b[row,col].view(np.int32)):match_b=order
        permutations.append(dict(ratio=r,anchor=row,channel=col,occupancy=int(count[row]),
            incoming_values=json.dumps(values.tolist()),baseline=float(a[row,col]),candidate=float(b[row,col]),fp64=float(ref[row,col]),
            baseline_valid_FP32_order=match_a is not None,candidate_valid_FP32_order=match_b is not None,
            baseline_order=json.dumps(match_a),candidate_order=json.dumps(match_b),distinct_FP32_results=len(possible)))
    # Same fixed incoming tensor, 24 pure Scatter replays: min/max envelope and occupancy attribution.
    for deterministic in [0,1]:
        state=torch.jit.load(str(R/f'snapshots/scatter_d{deterministic}_l0_r{j}.pt'),map_location='cpu')
        low=state.minimum.numpy().astype(np.float64);high=state.maximum.numpy().astype(np.float64)
        bias.append(dict(ratio=r,deterministic=deterministic,variable_elements=int((low!=high).sum()),max_repeat_range=float((high-low).max()),
            max_error_vs_fp64=float(np.maximum(abs(low-ref),abs(high-ref)).max()),
            violates_FP32_sum_bound=int(((abs(low-ref)>bound+1e-15)|(abs(high-ref)>bound+1e-15)).sum())))
    print('ROUNDING_ANALYZED',r,flush=True)
write('OCCUPANCY_ERRORS.csv',occupancy_rows);write('FP32_ROUNDOFF_BOUNDS.csv',bounds);write('VALID_ACCUMULATION_ORDERS.csv',permutations);write('SCATTER_REPEAT_ENVELOPE.csv',bias)
assert all(x['finite'] and x['gather_input_bits_equal'] and x['baseline_bound_violations']==0 and x['candidate_bound_violations']==0 for x in bounds)
assert all(x['baseline_valid_FP32_order'] and x['candidate_valid_FP32_order'] for x in permutations)
assert all(x['different_elements']==0 for x in occupancy_rows if x['occupancy'] in ['0','1','2'])
save('roundoff_analysis_complete.json',dict(passed=True,ratios=len(ratio),exact_permutation_examples=len(permutations),
    violating_FP32_bounds=0,different_elements_for_occupancy_le2=0))

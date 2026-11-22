from common import *
import statistics,subprocess

@torch.no_grad()
def main():
 s=scene();reports=[];rep=int(sys.argv[1]) if len(sys.argv)>1 else 1
 engines={'baseline':reference_renderer(s),'boundary':FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda'),
          'fused':FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda')}
 engines['boundary'].inference.quaternion_fusion=False
 engines['boundary'].inference.narrow_sort=False
 for e in engines.values():
  for i in range(24):
   data,_,_=e.prepare(camera(s,396+i),budget=2000,view_conditioned=True,measure_time=False);render_prepared(camera(s,396+i),data,s.background)
 torch.cuda.synchronize()
 for case in ['ratio1','ratio70','points2000','changing_ratio','changing_points']:
  samples={k:[] for k in engines};ks=[];ns=[]
  for frame in range(792):
   cam=camera(s,frame)
   request=dict(ratio_budget=.01 if case=='ratio1' else .7) if case.startswith('ratio') else dict(budget=2000)
   if case=='changing_ratio':request=dict(ratio_budget=RATIOS[(frame*7)%len(RATIOS)])
   if case=='changing_points':request=dict(budget=[1000,2000,3000,4000,5000,10000][(frame*5)%6])
   order=list(engines);order=order[(frame+rep)%3:]+order[:(frame+rep)%3]
   for tag in order:
    torch.cuda.synchronize();t=time.perf_counter()
    data,ids,stats=engines[tag].prepare(cam,**request,view_conditioned=True,measure_time=False)
    render_prepared(cam,data,s.background);torch.cuda.synchronize()
    samples[tag].append((time.perf_counter()-t)*1000)
   ks.append(len(ids));ns.append(stats['candidate_gaussians'])
  row=dict(case=case,frames=792,unique_K=len(set(ks)),K_min=min(ks),K_max=max(ks),candidate_min=min(ns),candidate_max=max(ns),
    summary={k:dict(mean_ms=statistics.mean(v),median_ms=statistics.median(v),p90_ms=float(np.percentile(v,90)),p99_ms=float(np.percentile(v,99)),total_ms=sum(v)) for k,v in samples.items()},raw_ms=samples)
  reports.append(row);save(ROOT/'reports'/f'DYNAMIC_BENCHMARK_{rep}.json',dict(results=reports,graph=False,camera_and_budget_cache=False))
  print(json.dumps({k:v for k,v in row.items() if k!='raw_ms'}),flush=True)
 save(ROOT/'reports'/f'GPU_STATE_{rep}.json',dict(processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],text=True)))
if __name__=='__main__':main()

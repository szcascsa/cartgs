from common import *
import statistics,subprocess

def summary(values):return dict(mean_ms=statistics.mean(values),median_ms=statistics.median(values),p90_ms=float(np.percentile(values,90)),p99_ms=float(np.percentile(values,99)),under_2ms=sum(v<2 for v in values)/len(values))
@torch.no_grad()
def main():
 s=scene();rep=int(sys.argv[1]) if len(sys.argv)>1 else 1
 engines={'v3':reference_renderer(s),'epilogue':FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda'),'parallel':FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda')}
 engines['parallel'].inference.parallel_heads=True
 results=[]
 # Complete live pipeline: new camera, candidate N and ratio every request. No Graph or output cache.
 for case in ['ratio1','ratio70','changing_ratio','changing_points']:
  samples={tag:dict(prepare=[],total=[]) for tag in engines};ks=[];ns=[]
  for i in range(220):
   frame=(i*7)%792;cam=camera(s,frame)
   req=dict(ratio_budget=.01 if case=='ratio1' else .7)
   if case=='changing_ratio':req=dict(ratio_budget=RATIOS[i%len(RATIOS)])
   if case=='changing_points':req=dict(budget=[1000,2000,3000,4000,5000,10000][i%6])
   order=list(engines);order=order[(i+rep)%3:]+order[:(i+rep)%3]
   for tag in order:
    torch.cuda.synchronize();start=time.perf_counter()
    data,ids,stats=engines[tag].prepare(cam,**req,view_conditioned=True,measure_time=False)
    prepared=time.perf_counter();render_prepared(cam,data,s.background);torch.cuda.synchronize();end=time.perf_counter()
    if i>=20:samples[tag]['prepare'].append((prepared-start)*1000);samples[tag]['total'].append((end-start)*1000)
   ks.append(len(ids));ns.append(stats['candidate_gaussians'])
  row=dict(case=case,frames=200,K_min=min(ks),K_max=max(ks),N_min=min(ns),N_max=max(ns),
    summary={tag:{name:summary(values) for name,values in stages.items()} for tag,stages in samples.items()},raw_ms=samples)
  results.append(row);save(ROOT/'reports'/f'DYNAMIC_BENCHMARK_{rep}.json',dict(results=results,graph=False,inputs_change_every_frame=True))
  print(json.dumps({k:v for k,v in row.items() if k!='raw_ms'}),flush=True)
 # Two MLPs only: original-shaped physical/view input and selected rows are prepared outside timing.
 nets=[];cam=camera(s,396)
 base=engines['v3'];a=base.inference;visible=a.frustum(cam.full_proj_transform,base.occupied);rows,_,available=a.candidates(visible);code=a.view_code(cam)
 for ratio in [.01,.1,.7]:
  controls=a.controls(ratio);x=a.selector_input(rows,ratio,code,True);k=min(int(len(rows)*ratio),available);ids=rows[:k]
  packed=a.compensation_input(ids,controls);condition=a.ext.condition_pack(ids,a.owners,code,True);ref=a.reference[ids]
  values={tag:dict(host=[],event=[]) for tag in engines}
  def invoke(tag):
   b=engines[tag].inference
   if tag=='v3':return b.ext.selector_logits(x,b.selector_parameters),b.ext.compensation_network(packed,controls[len(a.tables):].view(1,3),ref,b.compensation_parameters,condition,b.view_weight)
   streams,events=b._head_resources() if tag=='parallel' else ((),())
   return b.ext.selector_logits(x,b.selector_parameters,True),b.ext.compensation_network(packed,controls[len(a.tables):].view(1,3),ref,b.compensation_parameters,condition,b.view_weight,True,streams,events)
  for tag in engines:
   for _ in range(20):invoke(tag)
  for repeat in range(120):
   order=list(engines);order=order[(repeat+rep)%3:]+order[:(repeat+rep)%3]
   for tag in order:
    torch.cuda.synchronize();start=time.perf_counter();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
    invoke(tag);end.record();end.synchronize()
    values[tag]['host'].append((time.perf_counter()-start)*1000);values[tag]['event'].append(begin.elapsed_time(end))
  row=dict(ratio=ratio,candidate_N=len(rows),K=k,summary={tag:{n:summary(v) for n,v in stages.items()} for tag,stages in values.items()},raw_ms=values)
  nets.append(row);print('NETWORK',json.dumps({k:v for k,v in row.items() if k!='raw_ms'}),flush=True)
 save(ROOT/'reports'/f'NETWORK_BENCHMARK_{rep}.json',dict(results=nets,scope='两MLP，包括Compensation投影和Head打包；不含候选、排序、物理解码和有限值检查；相同输入形状；Event包含发射间隙和GPU争用'))
 save(ROOT/'reports'/f'GPU_STATE_{rep}.json',dict(processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],text=True)))
if __name__=='__main__':main()

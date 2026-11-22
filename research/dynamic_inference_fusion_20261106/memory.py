from common import *
@torch.no_grad()
def main():
 s=scene();tag=sys.argv[1];torch.cuda.synchronize();base=torch.cuda.memory_allocated()
 e=reference_renderer(s) if tag=='baseline' else FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda')
 for i in range(12):e.prepare(camera(s,396+i),budget=2000,view_conditioned=True,measure_time=False)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();retained=torch.cuda.memory_allocated()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True) as prof:
  for i in range(10):
   data,_,_=e.prepare(camera(s,396+i),budget=2000,view_conditioned=True,measure_time=False);render_prepared(camera(s,396+i),data,s.background)
 torch.cuda.synchronize()
 ops=[dict(name=v.key,count=v.count) for v in prof.key_averages()]
 save(ROOT/'reports'/f'MEMORY_{tag}.json',dict(base_bytes=base,retained_bytes=retained-base,peak_additional_bytes=torch.cuda.max_memory_allocated()-base,iterations=10,operations=ops))
 print(tag,'peak',torch.cuda.max_memory_allocated()-base,flush=True)
if __name__=='__main__':main()

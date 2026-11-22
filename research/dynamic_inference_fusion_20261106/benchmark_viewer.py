from common import *
import statistics,struct

def parse(packet):
 n=struct.unpack('<I',packet[:4])[0];return json.loads(packet[4:4+n]),packet[4+n:]
@torch.no_grad()
def main():
 s=scene();engines={'baseline':reference_renderer(s),'fused':FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda')}
 results=[];checks=0
 # Same scene instance and response work. Each frame/request recomputes visibility; no per-frame mask reuse.
 for case in ['changing_ratio','changing_points']:
  values={k:[] for k in engines};summary={k:[] for k in engines}
  for index,frame in enumerate(range(0,792,7)):
   req=dict(frame=frame,view_conditioned=True,reference_full=False,measure_inference_stages=False)
   if case=='changing_ratio':req.update(budget_unit='ratio',ratio=RATIOS[index%len(RATIOS)])
   else:req.update(budget=[1000,2000,3000,4000,5000,10000][index%6])
   packets={};order=list(engines)
   if index%2:order.reverse()
   for tag in order:
    s.renderer=engines[tag];torch.cuda.synchronize();start=time.perf_counter();packet=s.render(req,native=True)
    values[tag].append((time.perf_counter()-start)*1000);meta,payload=parse(packet);packets[tag]=(meta,payload)
    summary[tag].append(meta['stats']['anchor_summary_ms'])
   a,b=packets['baseline'],packets['fused'];assert a[1]==b[1],(case,frame,'payload')
   for k in ['active_gaussians','projected_gaussians','ground_truth_psnr','enabled_anchors','enabled_visible_gaussians','submitted_anchors','projected_anchors','anchor_levels']:assert a[0]['stats'][k]==b[0]['stats'][k],(case,frame,k)
   checks+=1
  row=dict(case=case,samples=len(values['baseline']),summary={tag:dict(median_ms=statistics.median(v),p90_ms=float(np.percentile(v,90)),p99_ms=float(np.percentile(v,99)),mean_ms=statistics.mean(v),summary_median_ms=statistics.median(summary[tag])) for tag,v in values.items()},raw_ms=values)
  results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='raw_ms'}),flush=True)
 save(ROOT/'reports/VIEWER_BENCHMARK.json',dict(passed=True,comparisons=checks,results=results,bytes_and_metrics_exact=True,graph=False,excludes='HTTP client and display'))
if __name__=='__main__':main()

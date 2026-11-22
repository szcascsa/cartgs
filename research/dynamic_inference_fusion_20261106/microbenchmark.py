from common import *
from selector_sampling import gumbel_selection
from transform_field import _normalize_quaternion,_multiply_quaternion
from anchor_budget import exact_budget_mask
import statistics

def measure(f):
 for _ in range(20):f()
 values=[]
 for _ in range(50):
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  a.record()
  for j in range(10):f()
  b.record();b.synchronize();values.append(a.elapsed_time(b)*1000/10)
 return dict(median_us=statistics.median(values),p90_us=float(np.percentile(values,90)),raw_us=values)
@torch.no_grad()
def main():
 s=scene();a=reference_renderer(s).inference;b=FrozenResidualInference(s.model,s.gaussians);cam=camera(s,396)
 occ=torch.bincount(b.owners,minlength=len(b.anchors.centers))>0
 rows,_,available=b.candidates(b.frustum(cam.full_proj_transform,occ));code=b.view_code(cam)
 reports=[]
 for ratio in [.01,.1,.7]:
  k=int(len(rows)*ratio);logits=b.logits(rows,ratio,code,True);scores=gumbel_selection(logits,hard=False,noise=torch.zeros_like(logits)).contiguous();eligible=b.eligible[rows]
  ids=rows[b.select(scores,eligible,min(k,available))];controls=b.controls(ratio);res=b.residual(ids,controls,code,True)
  def old_rotation():
   identity=torch.zeros_like(res[:,6:10]);identity[:,0]=1
   return s.gaussians.rotation_activation(_normalize_quaternion(_multiply_quaternion(_normalize_quaternion(identity+res[:,6:10]),a.rotation[ids])))
  functions={
   'frustum':(lambda:anchor_frustum_mask(b.anchors.centers,b.anchors.offset_scales,cam.full_proj_transform,occ),lambda:b.frustum(cam.full_proj_transform,occ)),
   'score':(lambda:gumbel_selection(logits,hard=False,noise=s.renderer.noise[rows]),lambda:b.scores(logits,s.renderer.noise,rows)),
   'index_sort':(lambda:b.ext.exact_select(scores,eligible,min(k,available),b.sort_workspace,False),lambda:b.ext.exact_select(scores,eligible,min(k,available),b.sort_workspace,True)),
   'quaternion':(old_rotation,lambda:b.ext.quaternion_decode(res,ids,b.rotation)),
   'compensation':(lambda:a.render_data(ids,ratio,True,code,True),lambda:b.render_data(ids,ratio,True,code,True))}
  for name,(old,new) in functions.items():
   torch.cuda.synchronize();x=measure(old);y=measure(new);row=dict(ratio=ratio,K=k,module=name,baseline=x,fused=y,speedup=x['median_us']/y['median_us']);reports.append(row);print(name,ratio,round(row['speedup'],3),flush=True)
 save(ROOT/'reports/MICROBENCHMARK.json',dict(results=reports,cuda_event=True,includes_other_gpu_interference=True))
if __name__=='__main__':main()

from common import *
from contextlib import contextmanager

def instrument(engine,tag):
 for name in ['frustum','candidates','view_code','logits','scores','select','render_data','compensation_input','residual','_compute']:
  if not hasattr(engine.inference,name):continue
  f=getattr(engine.inference,name)
  def wrapped(*args,_f=f,_name=name,**kwargs):
   with torch.cuda.nvtx.range('node/'+tag+'/'+_name):return _f(*args,**kwargs)
  setattr(engine.inference,name,wrapped)

@torch.no_grad()
def main():
 s=scene();engines={'baseline':reference_renderer(s),'fused':FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda')}
 for tag,e in engines.items():
  for j in range(20):e.prepare(camera(s,396+j),ratio_budget=.1,view_conditioned=True,measure_time=False)
  instrument(e,tag)
 for ratio in [.01,.1,.7]:
  for frame in range(396,406):
   cam=camera(s,frame);order=list(engines)
   if frame%2:order.reverse()
   for tag in order:
    e=engines[tag]
    with torch.cuda.nvtx.range(f'prepare/{tag}/{ratio}'):
     data,ids,stats=e.prepare(cam,ratio_budget=ratio,view_conditioned=True,measure_time=False)
    with torch.cuda.nvtx.range(f'render/{tag}/{ratio}'):rendered=render_prepared(cam,data,s.background,return_radii=True)
    with torch.cuda.nvtx.range(f'node/{tag}/summary'):
     s.anchor_visibility.resolve([0,1,2])
     s.anchor_visibility.frame_stats(s.model.anchors,cam,None,ids,rendered['radii'],e.inference)
    torch.cuda.synchronize()
if __name__=='__main__':main()

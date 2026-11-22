from common import *
@torch.no_grad()
def main():
 s=scene();a=reference_renderer(s);b=FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda')
 t=Comparisons('DYNAMIC_VALIDATION.json');cases=0
 for frame in range(792):
  cam=camera(s,frame)
  for request in [dict(ratio_budget=r) for r in RATIOS]+[dict(budget=k) for k in [0,1,1000,2000,3000,4000,5000,10000]]:
   x,ix,sx=a.prepare(cam,**request,view_conditioned=True,measure_time=False)
   y,iy,sy=b.prepare(cam,**request,view_conditioned=True,measure_time=False)
   t.same(f'{frame}/{request}/ids',ix,iy);t.packet(f'{frame}/{request}',x,y)
   for k in ['candidate_gaussians','visible_anchors','eligible_candidates','active_gaussians','ratio','budget_shortfall']:assert sx[k]==sy[k],(frame,request,k)
   rx=render_prepared(cam,x,s.background,return_radii=True);ry=render_prepared(cam,y,s.background,return_radii=True)
   t.same(f'{frame}/{request}/image',rx['render'],ry['render']);t.same(f'{frame}/{request}/radii',rx['radii'],ry['radii']);cases+=1
  if frame%25==0 or frame==791:
   print('VALIDATED',frame+1,cases,len(t.checks),flush=True);save(ROOT/'reports/DYNAMIC_STATUS.json',dict(frame=frame+1,cases=cases,comparisons=len(t.checks)))
 t.finish(frames=792,cases=cases,images_saved=0)
if __name__=='__main__':main()

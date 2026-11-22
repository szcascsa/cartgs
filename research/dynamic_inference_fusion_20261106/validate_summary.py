from common import *
from anchor_visibility import AnchorVisibility
@torch.no_grad()
def main():
 s=scene();e=FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda');vis=AnchorVisibility(s.model.anchors)
 checks=0
 for frame in range(792):
  cam=camera(s,frame);levels=[[0,1,2],[],[0],[1,2]][frame%4];mask=vis.resolve(levels)
  request=dict(ratio_budget=RATIOS[frame%len(RATIOS)])
  data,ids,_=e.prepare(cam,**request,anchor_mask=mask,view_conditioned=True,measure_time=False)
  rad=render_prepared(cam,data,s.background,return_radii=True)['radii']
  a=vis.frame_stats(s.model.anchors,cam,mask,ids,rad)
  b=vis.frame_stats(s.model.anchors,cam,mask,ids,rad,e.inference)
  assert a==b,(frame,a,b);checks+=1
  if frame%100==0:print('summary',frame,flush=True)
 # Synthetic level distribution and repeated submitted IDs: preserve unique-Anchor and per-point semantics.
 vis.levels=torch.arange(len(vis.levels),device='cuda')%3;vis.level_masks=torch.arange(3,device='cuda')[:,None]==vis.levels[None,:]
 for j in range(24):
  mask=vis.resolve([[0,1,2],[],[0],[1,2]][j%4]);cam=camera(s,j*31)
  ids=torch.arange(j*17,device='cuda')%len(e.inference.owners);ids=ids.repeat(2);rad=torch.arange(len(ids),device='cuda',dtype=torch.int32)%3
  a=vis.frame_stats(s.model.anchors,cam,mask,ids,rad);b=vis.frame_stats(s.model.anchors,cam,mask,ids,rad,e.inference)
  assert a==b,(j,a,b);checks+=1
 save(ROOT/'reports/SUMMARY_VALIDATION.json',dict(passed=True,comparisons=checks,frames=792,integer_exact=True))
 print('SUMMARY_PASS',checks,flush=True)
if __name__=='__main__':main()

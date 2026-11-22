from common import *
from selector_sampling import gumbel_selection
@torch.no_grad()
def main():
 s=scene();e=FrozenResidualInference(s.model,s.gaussians);test=Comparisons('PROBE.json')
 print('BUILT',flush=True)
 g=torch.Generator(device='cuda').manual_seed(123)
 a=s.model.anchors;occupied=torch.bincount(a.rows,minlength=len(a.centers))>0
 for frame in range(792):
  c=camera(s,frame)
  test.same(f'frustum/{frame}',e.frustum(c.full_proj_transform,occupied),anchor_frustum_mask(a.centers,a.offset_scales,c.full_proj_transform,occupied))
 for j in range(100):
  m=torch.randn(4,4,device='cuda',generator=g).t()
  test.same(f'frustum/random{j}',e.frustum(m,occupied),anchor_frustum_mask(a.centers,a.offset_scales,m,occupied))
 print('FRUSTUM_PASS',flush=True)
 for n in [0,1,7,32,129,1000,13298,100000]:
  rows=torch.randperm(n,device='cuda',generator=g);noise=torch.randn(n,2,device='cuda',generator=g)
  for scale in [0.,.01,1.,10.,100.,1e6]:
   logits=torch.randn(n,2,device='cuda',generator=g)*scale
   test.same(f'scores/{n}/{scale}',e.scores(logits,noise,rows),gumbel_selection(logits,hard=False,noise=noise[rows]))
 print('SCORES_PASS',flush=True)
 test.finish();print('PROBE_PASS',len(test.checks),flush=True)
if __name__=='__main__':main()

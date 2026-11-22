from common import *
from selector_sampling import gumbel_selection
from transform_field import _normalize_quaternion,_multiply_quaternion
@torch.no_grad()
def main():
 s=scene();e=FrozenResidualInference(s.model,s.gaussians);base=reference_renderer(s).inference;t=Comparisons('EDGE_VALIDATION.json');g=torch.Generator(device='cuda').manual_seed(824)
 for n in [0,1,2,3,7,31,32,33,127,128,129,1024,13298,65537]:
  rows=torch.arange(n,device='cuda');noise=torch.zeros(n,2,device='cuda')
  for typ in ['zero','signed_zero','saturated','extreme','random']:
   logits=torch.zeros(n,2,device='cuda')
   if typ=='signed_zero':logits[:,0]=-0.
   if typ=='saturated':logits[:,0]=100.;logits[:,1]=-100.
   if typ=='extreme':logits[:,0]=1e38;logits[:,1]=-1e38
   if typ=='random':logits.normal_(generator=g)
   t.same(f'score/{n}/{typ}',e.scores(logits,noise,rows),gumbel_selection(logits,hard=False,noise=noise))
 for n in [0,1,2,7,31,129,4096,len(e.owners)]:
  scores=torch.randint(0,4,(n,),device='cuda',generator=g).float();eligible=torch.rand(n,device='cuda',generator=g)>.3
  available=int(eligible.sum())
  for k in sorted(set([0,min(available,1),min(available,17),available//2,available])):
   t.same(f'sort/{n}/{k}',e.select(scores,eligible,k),base.select(scores,eligible,k))
 for n in [1,7,129,4096]:
  rows=torch.arange(n,device='cuda')
  for scale in [0.,-0.,1e-30,1e20,1e38]:
   res=torch.full((n,14),scale,device='cuda');ident=torch.zeros_like(res[:,6:10]);ident[:,0]=1.
   a=s.gaussians.rotation_activation(_normalize_quaternion(_multiply_quaternion(_normalize_quaternion(ident+res[:,6:10]),e.rotation[rows])))
   t.same(f'quat/{n}/{scale}',e.ext.quaternion_decode(res,rows,e.rotation),a)
 for n in [0,1,7,129,13298]:
  centers=torch.randn(n,3,device='cuda',generator=g);scales=torch.rand(n,1,device='cuda',generator=g);occupied=torch.rand(n,device='cuda',generator=g)>.2
  for i in range(30):
   m=torch.randn(4,4,device='cuda',generator=g);m=m.t() if i%2 else m
   t.same(f'frustum/{n}/{i}',e.ext.frustum(centers,scales,m,occupied),anchor_frustum_mask(centers,scales,m,occupied))
 # Non-default stream must consume the producer on the same stream.
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for frame in [0,396,791]:
   cam=camera(s,frame);occ=torch.bincount(e.owners,minlength=len(e.anchors.centers))>0
   t.same(f'stream/{frame}',e.frustum(cam.full_proj_transform,occ),anchor_frustum_mask(e.anchors.centers,e.anchors.offset_scales,cam.full_proj_transform,occ))
 torch.cuda.current_stream().wait_stream(stream)
 t.finish();print('EDGES_PASS',len(t.checks),flush=True)
if __name__=='__main__':main()

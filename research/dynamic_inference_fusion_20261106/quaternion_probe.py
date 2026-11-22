from common import *
from transform_field import _normalize_quaternion,_multiply_quaternion
@torch.no_grad()
def main():
 s=scene();e=FrozenResidualInference(s.model,s.gaussians);g=torch.Generator(device='cuda').manual_seed(73);t=Comparisons('QUATERNION_PROBE.json')
 for n in [0,1,7,32,129,1000,13298]:
  rows=torch.randperm(len(s.gaussians.get_xyz),device='cuda',generator=g)[:n]
  for scale in [0.,.01,1.,100.,1e6]:
   res=torch.randn(n,14,device='cuda',generator=g)*scale
   ident=torch.zeros_like(res[:,6:10]);ident[:,0]=1
   a=ident+res[:,6:10];b=e.ext.quaternion_identity(res);t.same(f'{n}/{scale}/id',a,b)
   a=_normalize_quaternion(a);b=e.ext.quaternion_divide(b,b.norm(dim=-1,keepdim=True));t.same(f'{n}/{scale}/norm1',a,b)
   a=_multiply_quaternion(a,e.rotation[rows]);b=e.ext.quaternion_multiply(b,rows,e.rotation);t.same(f'{n}/{scale}/multiply',a,b)
   a=_normalize_quaternion(a);b=e.ext.quaternion_divide(b,b.norm(dim=-1,keepdim=True));t.same(f'{n}/{scale}/norm2',a,b)
   a=s.gaussians.rotation_activation(a);t.same(f'{n}/{scale}/decode',a,e.ext.quaternion_decode(res,rows,e.rotation))
 t.finish();print('QUATERNION_PASS',len(t.checks),flush=True)
if __name__=='__main__':main()

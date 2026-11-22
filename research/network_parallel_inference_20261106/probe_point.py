from common import *
from torch.utils.cpp_extension import load
@torch.no_grad()
def main():
 s=scene();base=reference_renderer(s).inference
 ext=load(name='point_mlp_probe',sources=[str(HERE/'point_mlp.cpp'),str(HERE/'point_mlp.cu')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)
 g=torch.Generator(device='cuda').manual_seed(936);checks=[]
 for n in [0,1,7,32,129,1000,5000,13298]:
  rows=torch.arange(n,device='cuda');ratio=.3;code=base.view_code(camera(s,396));controls=base.controls(ratio)
  x=base.selector_input(rows,ratio,code,True)
  a=base.ext.selector_logits(x,base.selector_parameters);b=ext.selector(x,base.selector_parameters)
  tests=[('selector',a,b)]
  packed=base.compensation_input(rows,controls);condition=base.ext.condition_pack(rows,base.owners,code,True)
  rp=torch.nn.functional.linear(controls[len(base.tables):].view(1,3),base.spatial.ratio_projection.weight)
  vp=torch.nn.functional.linear(condition,base.view_weight)
  a=base.residual(rows,controls,code,True);b=ext.compensator(packed,rp,vp,base.reference[rows],base.compensation_parameters)
  tests.append(('compensator',a,b))
  for name,a,b in tests:
   same=torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32));row=dict(module=name,N=n,bitwise=same)
   if not same:row.update(max_abs=float((a-b).abs().max()),different=int((a.view(torch.int32)!=b.view(torch.int32)).sum()))
   checks.append(row);print(row,flush=True)
 save(ROOT/'reports/POINT_MLP_PROBE.json',dict(passed=all(v['bitwise'] for v in checks),checks=checks,production_enabled=False))
if __name__=='__main__':main()

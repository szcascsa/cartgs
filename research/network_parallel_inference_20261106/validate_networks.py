from common import *
@torch.no_grad()
def main():
 t=Comparisons('NETWORK_VALIDATION.json');g=torch.Generator(device='cuda').manual_seed(729)
 for group in ['both_li','selector_li','compensator_li','mode','legacy']:
  s=scene(group);a=reference_renderer(s).inference;b=FrozenResidualInference(s.model,s.gaussians)
  cam=camera(s,396);code=a.view_code(cam) if a.joint_schema else None
  for n in [0,1,7,32,129,1000,len(s.gaussians.get_xyz)]:
   rows=torch.randperm(len(s.gaussians.get_xyz),device='cuda',generator=g)[:n].sort().values
   for ratio in [.01,.03,.05,.08,.1,.3,.7,1.]:
    for local in ([False,True] if b.joint_schema else [False]):
     controls=b.controls(ratio);expect=a.residual(rows,controls,code,local);logits=a.logits(rows,ratio,code,local)
     packet=a.render_data(rows,ratio,True,code,local)
     t.same(f'{group}/{n}/{ratio}/{local}/logits',b.logits(rows,ratio,code,local),logits)
     for parallel in [False,True]:
      b.parallel_heads=parallel
      t.same(f'{group}/{n}/{ratio}/{local}/parallel{parallel}/residual',b.residual(rows,controls,code,local),expect)
      t.packet(f'{group}/{n}/{ratio}/{local}/parallel{parallel}',b.render_data(rows,ratio,True,code,local),packet)
   print('network',group,n,len(t.checks),flush=True)
  t.finish(group_completed=group)
 t.finish(groups=5,head_streams_exact=True,epilogue_exact=True)
if __name__=='__main__':main()

from common import *
from deployment_runtime import deployment_extension,DeploymentRenderer
from baseline import previous_extension,previous_runtime
from large_scene import LARGE
import itertools


@torch.no_grad()
def main():
    ext=deployment_extension();ref=previous_extension();test=Comparisons('PREFIX_EDGES.json')
    generator=torch.Generator(device='cuda').manual_seed(781)
    for n,width in itertools.product([0,1,3,15,16,17,257],[11,19]):
        physical=torch.randn((n,10),device='cuda',generator=generator)
        owners=torch.arange(n,device='cuda')%7;noise=torch.randn((n,2),device='cuda',generator=generator)
        view=torch.randn((7,7),device='cuda',generator=generator)
        weights=[torch.randn(shape,device='cuda',generator=generator)*.1 for shape in [(width,32),(32,),(32,32),(32,),(32,2),(2,)]]
        rows=torch.randperm(n,device='cuda',generator=generator)
        for case in ['ordinary','signed_zero','large','nonfinite']:
            if case=='signed_zero':physical.zero_();physical.view(torch.int32)[::2].fill_(-2147483648)
            elif case=='large':physical.fill_(1e30)
            elif case=='nonfinite' and n:physical[0,0]=float('nan')
            prefix=ext._selector_prefix(physical,weights[0])
            for tiled,local,ratio in itertools.product([False,True],[False,True],[.01,.08,.7,1.]):
                args=(physical,rows,owners,view,noise,weights,ratio,local,2 if width==19 else 0)
                a=(ref._selector_tiled if tiled else ref.selector)(*args)
                b=ext._selector_prefixed(*args,tiled,prefix)
                for j in range(3):test.same(str((n,width,case,tiled,local,ratio,j)),a[j],b[j])
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        a=previous_runtime(s);b=DeploymentRenderer(s.model,s.gaussians,s.config)
        for prefix,visible,small in itertools.product([False,True],[False,True],[False,True]):
            b._pipeline._static_options(prefix,visible,small)
            for frame,k in [(0,1),(len(s.views)-1,1000),(len(s.views)//2,5000)]:
                cam=camera(s,frame);x,ix,_=a.prepare(cam,budget=k);y,iy,stats=b.prepare(cam,budget=k)
                name=str((group,prefix,visible,small,frame,k));test.same(name,ix,iy);test.packet(name,x,y)
                assert b._pipeline._prefix_bytes()==(b.total*128 if prefix else 0)
                assert stats['graph_count']<=3
        test.finish(group=group)
        del a,b,s;torch.cuda.empty_cache()
    test.finish()


if __name__=='__main__':main()

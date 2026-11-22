from common import *
from baseline import previous_extension,previous_runtime
from deployment_runtime import deployment_extension
import math


@torch.no_grad()
def main():
    torch.manual_seed(2917);base=previous_extension();new=deployment_extension();test=Comparisons('SYNTHETIC_VALIDATION.json')
    for width in [11,19]:
        for n in [0,1,15,16,17,127,511,1025,4097,16385,65537]:
            physical=torch.randn((n,10),device='cuda')*10;rows=torch.randperm(n,device='cuda');owners=torch.randint(0,67,(n,),device='cuda')
            view=torch.randn((67,7),device='cuda');noise=torch.randn((n,2),device='cuda')
            weights=[torch.randn(shape,device='cuda')*.1 for shape in [(width,32),(32,),(32,32),(32,),(32,2),(2,)]]
            for local in [False,True] if width==19 else [False]:
                for magnitude in [1e-8,1.,1e6]:
                    args=(physical*magnitude,rows,owners,view,noise,weights,.08,local,2 if width==19 else 0)
                    a=base.selector(*args);b=new._selector_tiled(*args)
                    for j in range(3):test.same(str(('selector',width,n,local,magnitude,j)),a[j],b[j])
    s=scene();runtime=previous_runtime(s);b=runtime._base
    for n in [0,1,15,16,17,127,511,1025,4097]:
        rows=torch.randperm(runtime.total,device='cuda')[:n]
        for magnitude in [0.,1e-6,1.,1e3]:
            features=[torch.randn_like(t)*magnitude for t in b.tables]
            view=torch.randn((len(b.anchors.centers),7),device='cuda')*magnitude
            args=(b.offset,rows,b.owners,b.reference,features,runtime._compensator,b.compensated_master,
                view,runtime._view_weight,[1.,.4,1.,-.1,-.1,-.1][:len(features)],[.3,math.sin(.15*math.pi),math.cos(.15*math.pi)],True,b.joint_schema,hasattr(b.spatial,'softplus_scale_schema'))
            a=base.compensate(*args);result=new._compensate_tiled(*args,16)
            for j in range(3):test.same(str(('comp',n,magnitude,j)),a[j],result[j])
    test.finish();print('SYNTHETIC_PASS',len(test.checks),flush=True)


if __name__=='__main__':main()

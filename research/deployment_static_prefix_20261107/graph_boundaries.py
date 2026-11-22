from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from large_scene import LARGE


@torch.no_grad()
def main():
    checks=Comparisons('GRAPH_BOUNDARIES.json')
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        a=previous_runtime(s);b=DeploymentRenderer(s.model,s.gaussians,s.config)
        # Change the input camera layout, field of view metadata and near plane.
        for i in range(36):
            cam=camera(s,(i*73)%len(s.views))
            cam.world_view_transform=cam.world_view_transform.clone().t().contiguous().t()
            cam.full_proj_transform=cam.full_proj_transform.clone().contiguous()
            cam.camera_center=cam.camera_center.repeat_interleave(2)[::2]
            cam.image_width+=i%3;cam.znear*=1+i*.02
            request=dict(budget=[0,1,511,512,767,768,769,1023,1024,1025,2048,2049,4096,4097,8192,8193,100000][i%17])
            x,ix,_=a.prepare(cam,**request);y,iy,stats=b.prepare(cam,**request)
            checks.same(f'{group}/{i}/ids',ix,iy);checks.packet(f'{group}/{i}',x,y)
            assert stats['graph_count']<=32
        # A flag change must drop every captured execution plan before reuse.
        for radix,tile,selector in [(True,0,True),(True,4,True),(True,8,False),(False,16,True),(True,16,True)]:
            b._pipeline._tuning(radix,tile,selector)
            for local in [False,True]:
                for comp in [False,True]:
                    cam=camera(s,7)
                    x,ix,_=a.prepare(cam,budget=2000,view_conditioned=local,compensation=comp)
                    y,iy,stats=b.prepare(cam,budget=2000,view_conditioned=local,compensation=comp)
                    checks.same(f'{group}/flags/{radix}/{tile}/{selector}/{local}/{comp}',ix,iy)
                    checks.packet(f'{group}/flags/{tile}/{local}/{comp}',x,y)
        b._pipeline._graph_enabled(False)
        x,ix,_=a.prepare(camera(s,12),budget=5000)
        y,iy,stats=b.prepare(camera(s,12),budget=5000)
        assert not stats['graph_replay']
        checks.same(f'{group}/disabled',ix,iy);checks.packet(group+'/disabled',x,y)
        b._pipeline._graph_enabled(True)
        y,iy,stats=b.prepare(camera(s,12),budget=5000)
        assert stats['graph_replay'] and stats['graph_count']==1
        checks.same(f'{group}/reenabled',ix,iy);checks.packet(group+'/reenabled',x,y)
        for enabled in [False,True]:
            b._pipeline._optimization_controls(enabled)
            x,ix,_=a.prepare(camera(s,12),budget=100000)
            y,iy,stats=b.prepare(camera(s,12),budget=100000)
            assert stats['graph_count']<=1
            checks.same(f'{group}/saturation/{enabled}',ix,iy)
            checks.packet(f'{group}/saturation/{enabled}',x,y)
        for unroll in [False,True]:
            for select in [False,True]:
                for candidates in [False,True]:
                    b._pipeline._kernel_options(unroll,select,candidates)
                    for request in [dict(budget=1000),dict(ratio_budget=.7)]:
                        x,ix,_=a.prepare(camera(s,17),**request)
                        y,iy,_=b.prepare(camera(s,17),**request)
                        label=f'{group}/kernel_options/{unroll}/{select}/{candidates}/{request}'
                        checks.same(label,ix,iy);checks.packet(label,x,y)
        checks.finish(group=group)
        del a,b,s
        torch.cuda.empty_cache()
    checks.finish(groups=2)


if __name__=='__main__':main()

from common import *
from deployment_runtime import DeploymentRenderer,deployment_extension
from baseline import previous_runtime,previous_extension
from large_scene import LARGE
from components import sample
import statistics


@torch.no_grad()
def gpu_time(fn):
    for _ in range(4):value=fn()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(16):value=fn()
    for _ in range(8):graph.replay()
    torch.cuda.synchronize()
    rows=[]
    for _ in range(9):
        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(12):graph.replay()
        end.record();end.synchronize()
        rows.append(start.elapsed_time(end)*1000/(16*12))
    return dict(median_us=statistics.median(rows),samples_us=rows)


@torch.no_grad()
def main():
    ext=deployment_extension();ref=previous_extension();check=Comparisons('COMPONENT_BITS.json');rows=[]
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        run=previous_runtime(s);b=run._base
        prefix=ext._selector_prefix(b.selector_static,run._selector[0])
        for frame in [0,len(s.views)//2,len(s.views)-1]:
            cam=camera(s,frame);planes=ref._frustum_planes(cam.full_proj_transform)
            distance=b.anchors.centers@planes[:,:3].t()
            visible=ref._frustum_test(distance,planes,b.anchors.offset_scales,run._reference.occupied)
            candidates=torch.nonzero(visible[b.owners],as_tuple=False).flatten()
            vargs=(b.anchors.centers,b.view_centers4,b.view_corners4,cam.camera_center.contiguous(),cam.world_view_transform,
                cam.full_proj_transform,b.model.selector_view_scene_scale,cam.image_width/cam.image_height,cam.znear)
            full=ref.view(*vargs);masked=ext._view_masked(*vargs,visible)
            check.same(str((group,frame,'view')),full[visible],masked[visible])
            vfns=[lambda:ref.view(*vargs),lambda:ext._view_masked(*vargs,visible)]
            for version,fn in zip(['v27','visible'],vfns):
                rows.append(dict(group=group,module='view',frame=frame,anchors=len(visible),visible=int(visible.sum()),version=version,**gpu_time(fn)))
            for ratio in [.01,.1,.7]:
                args=(b.selector_static,candidates,b.owners,full,run._reference.noise,run._selector,ratio,True,b.joint_schema)
                for tiled in [True,False]:
                    a=(ref._selector_tiled if tiled else ref.selector)(*args)
                    f=lambda:ext._selector_prefixed(*args,tiled,prefix)
                    c=f()
                    for j in range(3):check.same(str((group,frame,ratio,tiled,'selector',j)),a[j],c[j])
                    if tiled:
                        funcs=[lambda:ref._selector_tiled(*args),f]
                        for version,fn in zip(['v27','prefix'],funcs):
                            rows.append(dict(group=group,module='selector',frame=frame,ratio=ratio,n=len(candidates),version=version,**gpu_time(fn)))
            if frame==len(s.views)//2:
                for ratio in [.01,.1,.7]:
                    ids,args=sample(s,run,frame,ratio)
                    # Components use the same frozen input and ratio, with a controlled selected count.
                    pool=run.prepare(cam,ratio_budget=.7)[1]
                    for k in [1,16,64,128,256,512,768,1000,1024,1500,2000]:
                        arguments=(args[0],pool[:k],*args[2:]);baseline=ref._compensate_tiled(*arguments,0)
                        funcs=[('v27',lambda:ref._compensate_tiled(*arguments,0))]
                        for tile in [6416,8208]:
                            f=lambda tile=tile:ext._compensate_tiled(*arguments,tile)
                            c=f()
                            for j in range(3):check.same(str((group,ratio,k,tile,'comp',j)),baseline[j],c[j])
                            funcs.append((str(tile),f))
                        # Alternate order to reduce clock and thermal bias.
                        if k%128:funcs.reverse()
                        for version,fn in funcs:
                            rows.append(dict(group=group,module='compensation',ratio=ratio,k=min(k,len(pool)),version=version,**gpu_time(fn)))
            save(ROOT/'reports/MICRO.json',dict(rows=rows))
            check.finish(group=group,frame=frame)
            print('MICRO',group,frame,len(rows),flush=True)
        del run,s,b;torch.cuda.empty_cache()
    save(ROOT/'reports/MICRO.json',dict(passed=True,rows=rows));check.finish()


if __name__=='__main__':main()

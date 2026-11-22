from common import *
from baseline import previous_runtime,previous_extension
from deployment_runtime import deployment_extension
from large_scene import LARGE
from radix_test import scratch,timer
import statistics
import math


@torch.no_grad()
def sample(s,a,frame,ratio):
    b=a._base;cam=camera(s,frame)
    visible=b.frustum(cam.full_proj_transform,a._reference.occupied)
    rows,_,available=b.candidates(visible)
    view=previous_extension().view(b.anchors.centers,b.view_centers4,b.view_corners4,cam.camera_center.contiguous(),
        cam.world_view_transform,cam.full_proj_transform,b.model.selector_view_scene_scale,
        cam.image_width/cam.image_height,cam.znear) if b.joint_schema!=1 and b.joint_schema else b.empty
    scores=previous_extension().selector(b.selector_static,rows,b.owners,view,a._reference.noise,a._selector,ratio,bool(b.joint_schema),b.joint_schema)
    k=min(int(ratio*len(rows)),available)
    small=torch.empty(2+2*((len(rows)+255)//256),device='cuda',dtype=torch.int32)
    from radix_test import sorting
    old=sorting(len(rows))
    ids=previous_extension().select(scores[0],rows,b.eligible,k,small,old)
    weights=[1.]+[-1.]*(len(a.knots)-1)
    for j in range(1,len(weights)):
        if ratio<=a.knots[j-1]+1e-8:break
        weights[j]=min(1.,(ratio-a.knots[j-1])/(a.knots[j]-a.knots[j-1]))
    lo,hi=b.spatial.ratio_encoding_values;t=2*(math.log(ratio)-math.log(lo))/(math.log(hi)-math.log(lo))-1
    code=[t,math.sin(math.pi*t/2),math.cos(math.pi*t/2)]
    arguments=(b.offset,ids,b.owners,b.reference,b.tables,a._compensator,b.compensated_master,view,a._view_weight,
        weights,code,bool(b.joint_schema),b.joint_schema,hasattr(b.spatial,'softplus_scale_schema'))
    return rows,scores,ids,arguments,small,old


@torch.no_grad()
def main():
    ext=deployment_extension();base=previous_extension();checks=Comparisons('COMPONENT_VALIDATION.json');rows_out=[]
    groups=['both_li','selector_li','compensator_li','mode','legacy','replica']
    for group in groups:
        s=scene(group) if group!='replica' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        a=previous_runtime(s)
        for frame in [0,len(s.views)//2,len(s.views)-1]:
            for ratio in [.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7,1.]:
                rows,scores,ids,args,small,old=sample(s,a,frame,ratio)
                work=scratch(len(rows));k=len(ids)
                checks.same(str((group,frame,ratio,'ids')),ids,ext._radix_select(scores[0],rows,a._base.eligible,k,work,scores[2]))
                expected=ext.compensate(*args)
                for tile in [4,8]:
                    actual=ext._compensate_tiled(*args,tile)
                    for j in range(3):checks.same(str((group,frame,ratio,tile,j)),expected[j],actual[j])
                if frame==0 and ratio==.7:
                    for edge in [0,1,3,4,5,7,8,9,15,16,31,32,33,127,128,129,511,512,1023,1024]:
                        partial=list(args);partial[1]=ids[:edge];target=ext.compensate(*partial)
                        for tile in [4,8]:
                            result=ext._compensate_tiled(*partial,tile)
                            for j in range(3):checks.same(str((group,'edge',edge,tile,j)),target[j],result[j])
                if frame==len(s.views)//2 and ratio in [.01,.1,.7]:
                    functions={'v7_comp':lambda:ext.compensate(*args),
                        'tile4':lambda:ext._compensate_tiled(*args,4),'tile8':lambda:ext._compensate_tiled(*args,8),
                        'v7_select':lambda:base.select(scores[0],rows,a._base.eligible,k,small,old),
                        'radix':lambda:ext._radix_select(scores[0],rows,a._base.eligible,k,work,scores[2])}
                    samples={tag:[] for tag in functions}
                    for fn in functions.values():
                        for _ in range(5):fn()
                    for repeat in range(8):
                        for tag in (list(functions) if repeat%2 else list(functions)[::-1]):
                            with torch.cuda.nvtx.range('node/'+group+'/'+str(ratio)+'/'+tag):samples[tag].append(timer(functions[tag],20))
                    rows_out.append(dict(group=group,frame=frame,ratio=ratio,N=len(rows),K=k,
                        median_us={t:statistics.median(v) for t,v in samples.items()},samples=samples))
                checks.finish(group_completed=group,frame_completed=frame)
        print('COMPONENTS',group,len(checks.checks),flush=True)
        save(ROOT/'reports/COMPONENT_BENCHMARK.json',dict(results=rows_out))
        del a,s;torch.cuda.empty_cache()
    checks.finish(groups=groups)


if __name__=='__main__':main()

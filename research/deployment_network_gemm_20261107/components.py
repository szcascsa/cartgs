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



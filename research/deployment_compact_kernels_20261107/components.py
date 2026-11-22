from common import *
from baseline import previous_runtime,previous_extension
from large_scene import LARGE
import math


@torch.no_grad()
def sample(s,runtime,frame,ratio):
    b=runtime._base;cam=camera(s,frame)
    packet,ids,stats=runtime.prepare(cam,ratio_budget=ratio)
    view=previous_extension().view(b.anchors.centers,b.view_centers4,b.view_corners4,cam.camera_center.contiguous(),
        cam.world_view_transform,cam.full_proj_transform,b.model.selector_view_scene_scale,
        cam.image_width/cam.image_height,cam.znear) if b.joint_schema!=1 and b.joint_schema else b.empty
    weights=[1.]+[-1.]*(len(runtime.knots)-1)
    for j in range(1,len(weights)):
        if ratio<=runtime.knots[j-1]+1e-8:break
        weights[j]=min(1.,(ratio-runtime.knots[j-1])/(runtime.knots[j]-runtime.knots[j-1]))
    lo,hi=b.spatial.ratio_encoding_values
    t=2*(math.log(ratio)-math.log(lo))/(math.log(hi)-math.log(lo))-1
    code=[t,math.sin(math.pi*t/2),math.cos(math.pi*t/2)]
    arguments=(b.offset,ids,b.owners,b.reference,b.tables,runtime._compensator,b.compensated_master,view,runtime._view_weight,
        weights,code,bool(b.joint_schema),b.joint_schema,hasattr(b.spatial,'softplus_scale_schema'))
    return ids,arguments

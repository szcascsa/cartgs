""">65536容量只验证保留路径；重复地图不是性能证据。"""
from common import *
from baseline import previous_runtime,previous_extension
from deployment_runtime import deployment_extension
from large_scene import LARGE


@torch.no_grad()
def main():
    s=BudgetScene(LARGE,None,5000,0,None,'pytorch');a=previous_runtime(s);b=a._base
    repeat=lambda t:t.repeat((2,)+(1,)*(t.dim()-1))
    state=[b.anchors.centers,b.anchors.offset_scales,a._reference.occupied,repeat(b.owners),b.eligible_counts*2,
        repeat(b.eligible),repeat(b.selector_static),repeat(b.offset),repeat(b.reference),repeat(a._reference.noise),
        b.view_centers4,b.view_corners4,b.model.selector_view_scene_scale]
    arguments=(state,a._selector,a._compensator,b.tables,[repeat(t) for t in b.compensated_master],
        [repeat(t) for t in b.base_master],a._view_weight,list(a.knots),list(b.spatial.ratio_encoding_values),
        b.joint_schema,b.pc.active_sh_degree,hasattr(b.spatial,'softplus_scale_schema'))
    old=previous_extension().Pipeline(*arguments);new=deployment_extension().Pipeline(*arguments)
    checks=Comparisons('HOST_CAPACITY_VALIDATION.json');cases=0
    for frame in [0,68,136]:
        cam=camera(s,frame)
        for ratio in [0.,*RATIOS,1.]:
            values=(cam.world_view_transform,cam.full_proj_transform,cam.camera_center,cam.image_width,cam.image_height,
                cam.znear,0,ratio,True,True,True)
            x,ix,sx=old.prepare(*values);y,iy,sy=new.prepare(*values)
            assert not sy['device_counts']
            checks.same(str((frame,ratio,'ids')),ix,iy);checks.packet(str((frame,ratio)),x,y);cases+=1
    checks.finish(cases=cases,N=2*a.total,synthetic=True,performance_claim=False)
    print('HOST_CAPACITY_PASS',2*a.total,cases,flush=True)


if __name__=='__main__':main()

from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer


@torch.no_grad()
def main():
    s=scene();old=previous_runtime(s);new=DeploymentRenderer(s.model,s.gaussians,s.config)
    b=new._base;test=Comparisons('LARGE_PIPELINE.json');cases=0
    for factor in [2,5]:
        repeated=lambda t:torch.cat([t]*factor,0)
        state=[b.anchors.centers,b.anchors.offset_scales,new._reference.occupied,repeated(b.owners),
            b.eligible_counts*factor,repeated(b.eligible),repeated(b.selector_static),repeated(b.offset),
            repeated(b.reference),repeated(new._reference.noise),b.view_centers4,b.view_corners4,b.model.selector_view_scene_scale]
        arguments=[state,new._selector,new._compensator,b.tables,[repeated(t) for t in b.compensated_master],
            [repeated(t) for t in b.base_master],new._view_weight,list(new.knots),list(b.spatial.ratio_encoding_values),
            b.joint_schema,b.pc.active_sh_degree,hasattr(b.spatial,'softplus_scale_schema')]
        a=old._ext.Pipeline(*arguments);c=new._ext.Pipeline(*arguments)
        for frame in [0,396,791]:
            cam=camera(s,frame)
            for ratio,budget,use in [(.01,0,True),(.1,0,True),(.7,0,True),(1.,0,True),(0.,0,False),(0.,5000,False)]:
                for comp in [False,True]:
                    args=[cam.world_view_transform,cam.full_proj_transform,cam.camera_center,
                        cam.image_width,cam.image_height,cam.znear,budget,ratio,use,True,comp]
                    x,ix,sx=a.prepare(*args);y,iy,sy=c.prepare(*args)
                    assert sy['device_counts'] is False
                    label=str((factor,frame,ratio,budget,comp));test.same(label+'/ids',ix,iy);test.packet(label,x,y)
                    assert all(sx[k]==sy[k] for k in ['candidate_gaussians','active_gaussians','ratio','budget_shortfall'])
                    # Render representative cases; all attributes are already checked for every case.
                    if ratio in [.01,.7] and comp:
                        rx=render_prepared(cam,x,s.background,return_radii=True);ry=render_prepared(cam,y,s.background,return_radii=True)
                        test.same(label+'/pixels',rx['render'],ry['render']);test.same(label+'/radii',rx['radii'],ry['radii'])
                    cases+=1
        test.finish(cases=cases,capacity=factor*new.total,synthetic_duplicate_map=True)
        print('LARGE_PIPELINE',factor*new.total,cases,flush=True)
    test.finish(cases=cases,capacities=[2*new.total,5*new.total],synthetic_duplicate_map=True)


if __name__=='__main__':main()

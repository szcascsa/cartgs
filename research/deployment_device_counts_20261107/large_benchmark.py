from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from benchmark import stats


@torch.no_grad()
def main():
    s=scene();old=previous_runtime(s);new=DeploymentRenderer(s.model,s.gaussians,s.config)
    b=new._base;results=[]
    for factor in [2,5]:
        rep=lambda t:torch.cat([t]*factor,0)
        state=[b.anchors.centers,b.anchors.offset_scales,new._reference.occupied,rep(b.owners),b.eligible_counts*factor,
            rep(b.eligible),rep(b.selector_static),rep(b.offset),rep(b.reference),rep(new._reference.noise),
            b.view_centers4,b.view_corners4,b.model.selector_view_scene_scale]
        args=[state,new._selector,new._compensator,b.tables,[rep(t) for t in b.compensated_master],
            [rep(t) for t in b.base_master],new._view_weight,list(new.knots),list(b.spatial.ratio_encoding_values),
            b.joint_schema,b.pc.active_sh_degree,hasattr(b.spatial,'softplus_scale_schema')]
        engines={'previous_cpp':old._ext.Pipeline(*args),'device_counts':new._ext.Pipeline(*args)}
        for repeat in [1,2]:
            for ratio in [.01,.7]:
                values={tag:[] for tag in engines}
                for i in range(120):
                    cam=camera(s,((i-20)*137)%792);tags=list(engines)
                    if (i+repeat)%2:tags.reverse()
                    for tag in tags:
                        torch.cuda.synchronize();start=time.perf_counter()
                        result=engines[tag].prepare(cam.world_view_transform,cam.full_proj_transform,cam.camera_center,
                            cam.image_width,cam.image_height,cam.znear,0,ratio,True,True,True)
                        if tag=='device_counts':assert result[2]['device_counts'] is False
                        if i>=20:values[tag].append((time.perf_counter()-start)*1000)
                row=dict(capacity=factor*new.total,repeat=repeat,ratio=ratio,frames=100,
                    summary={tag:stats(v) for tag,v in values.items()},raw_ms=values)
                results.append(row);save(ROOT/'reports/LARGE_BENCHMARK.json',dict(results=results,
                    synthetic_duplicate_map=True,includes_rasterizer=False))
                print(json.dumps({k:v for k,v in row.items() if k!='raw_ms'}),flush=True)


if __name__=='__main__':main()

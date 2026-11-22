from common import *
from deployment_runtime import DeploymentRenderer
import math


@torch.no_grad()
def main():
    s=scene();ref=reference_renderer(s);fast=DeploymentRenderer(s.model,s.gaussians,s.config)
    rows=[]
    for frame in range(0,792,12):
        cam=camera(s,frame)
        for ratio in [.01,.05,.1,.3,.5,.7]:
            a,ia,_=ref.prepare(cam,ratio_budget=ratio,view_conditioned=True,measure_time=False)
            b,ib,_=fast.prepare(cam,ratio_budget=ratio)
            assert torch.equal(ia,ib)
            output=dict(frame=frame,ratio=ratio,K=len(ib))
            for tag,packet in [('base',a),('fast',b)]:
                xyz=packet['inputs']['means3D'];scale=packet['inputs']['scales'];q=packet['inputs']['rotations']
                d=(xyz-s.gaussians.get_xyz[ia]).norm(dim=1)/ref.inference.reference[ia].flatten()
                s_ratio=scale/s.gaussians.get_scaling[ia]
                rotation=2*torch.acos((q*s.gaussians.get_rotation[ia]).sum(1).abs().clamp_max(1))*180/math.pi
                for name,v in [('xyz_reference_units',d),('scale_factor',s_ratio.flatten()),('rotation_degrees',rotation)]:
                    values=torch.quantile(v,v.new_tensor([.5,.9,.99])).cpu().tolist()
                    for p,value in zip(['p50','p90','p99'],values):output[tag+'_'+name+'_'+p]=value
                    output[tag+'_'+name+'_max']=float(v.max())
            output['xyz_max_abs_difference']=float((a['inputs']['means3D']-b['inputs']['means3D']).abs().max())
            output['scale_max_abs_difference']=float((a['inputs']['scales']-b['inputs']['scales']).abs().max())
            rows.append(output)
    comparisons={}
    for key in ['xyz_reference_units_p99','xyz_reference_units_max','scale_factor_p99','scale_factor_max',
                'rotation_degrees_p99','rotation_degrees_max']:
        comparisons[key]=max(abs(v['fast_'+key]-v['base_'+key]) for v in rows)
    save(ROOT/'reports/GEOMETRY.json',dict(passed=True,frames=66,cases=len(rows),rows=rows,
        max_paired_absolute_change=comparisons,finite=True,
        scope='检查优化是否放大补偿；原模型既有几何长尾未改动，不把长尾小当作渲染质量证明'))
    print('GEOMETRY_PASS',len(rows),comparisons,flush=True)


if __name__=='__main__':
    import math
    main()

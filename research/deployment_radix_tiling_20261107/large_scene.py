from common import *
from deployment_runtime import DeploymentRenderer
from baseline import previous_runtime


LARGE=Path('/home/prmi/zwj/realLOD/saved_experiments/dual_view_li_best_full_20261107/runs/Replica/office3/rgbd/attempt_02/model/41149_shutdown/ply')


@torch.no_grad()
def main():
    before={str(p):sha(p) for p in LARGE.rglob('*') if p.is_file()}
    save(ROOT/'reports/LARGE_INPUTS_BEFORE.json',before)
    s=BudgetScene(LARGE,None,5000,0,None,'pytorch')
    a=previous_runtime(s);b=DeploymentRenderer(s.model,s.gaussians,s.config)
    metadata=dict(checkpoint=str(LARGE),N=b.total,anchors=len(s.model.anchors.centers),frames=len(s.views),schema=b._base.joint_schema,views=[])
    for frame in np.linspace(0,len(s.views)-1,9,dtype=int):
        cam=camera(s,int(frame))
        for ratio in [.01,.1,.7]:
            data,ids,stats=a.prepare(cam,ratio_budget=ratio)
            metadata['views'].append(dict(frame=int(frame),ratio=ratio,**stats))
    save(ROOT/'reports/LARGE_SCENE.json',metadata)
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()

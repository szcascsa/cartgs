from common import *
from compensation_student import StudentCompensator, teacher_fingerprint
from deployment_runtime import DeploymentRenderer, deployment_extension
from distill_compensation import ratio_controls


def old_runtime(s):
    sys.path.insert(0,'/home/prmi/zwj/experiments/deployment_static_prefix_20261107/build/mycartgs_deployment_fp32_v30')
    import mycartgs_deployment_fp32_v30
    spec=importlib.util.spec_from_file_location('teacher_v30',ROOT/'reference/python/deployment_runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.deployment_extension=lambda:mycartgs_deployment_fp32_v30
    return module.DeploymentRenderer(s.model,s.gaussians,s.config)


@torch.no_grad()
def main():
    ext=deployment_extension();s=scene();teacher=DeploymentRenderer(s.model,s.gaussians,s.config)
    old=old_runtime(s);b=teacher._base;check=Comparisons('TEACHER_REGRESSION.json')
    for frame in [0,396,791]:
        cam=camera(s,frame)
        for k in [0,1,128,767,768,1000,1024,1025,5000]:
            a,ia,_=old.prepare(cam,budget=k);v,iv,_=teacher.prepare(cam,budget=k)
            label=str((frame,k));check.same(label,ia,iv);check.packet(label,a,v)
    check.finish()
    student=StudentCompensator.from_teacher(b.spatial);student.teacher_fingerprint=teacher_fingerprint(s.model,s.gaussians)
    with torch.no_grad():
        for head in student.heads.values():head.weight.normal_(0,.01);head.bias.normal_(0,.005)
    student.eval();student.requires_grad_(False)
    run=DeploymentRenderer(s.model,s.gaussians,s.config,student=student)
    errors=[];cases=0
    for frame in [0,396,791]:
        cam=camera(s,frame)
        for ratio in [.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7,1.]:
            _,pool,_=teacher.prepare(cam,ratio_budget=ratio)
            view=ext.view(b.anchors.centers,b.view_centers4,b.view_corners4,cam.camera_center.contiguous(),
                cam.world_view_transform,cam.full_proj_transform,b.model.selector_view_scene_scale,
                cam.image_width/cam.image_height,cam.znear)
            weights,code=ratio_controls(ratio,teacher.knots,b.spatial.ratio_encoding_values)
            control=b.offset.new_tensor(weights+code)
            for local in [False,True]:
                for k in [0,1,3,4,7,8,15,16,17,129,1000]:
                    ids=pool[:k];packed=b.compensation_input(ids,control)
                    condition=torch.cat((view[b.owners[ids]],view.new_ones((len(ids),1))),dim=1) if local else None
                    truth=student(packed,packed.new_tensor([code]),b.reference[ids],condition)
                    args=(b.offset,ids,b.owners,b.reference,b.tables,run._compensator,b.compensated_master,view,
                        run._view_weight,weights,code,local,b.joint_schema,hasattr(b.spatial,'softplus_scale_schema'))
                    for tile in [0,8,16,8208]:
                        flat,delta,status=ext._compensate_tiled(*args,tile)
                        assert int(status)==0
                        error=float((delta-truth).abs().max()) if len(ids) else 0
                        assert torch.allclose(delta,truth,rtol=3e-5,atol=3e-6),(frame,ratio,k,local,tile,error)
                        errors.append(error);cases+=1
    graph_checks=Comparisons('STUDENT_GRAPH_VALIDATION.json')
    for i in range(24):
        cam=camera(s,(i*73+19)%len(s.views));k=[0,1,767,768,1000,1024,1025,5000][i%8]
        _,ids,_=teacher.prepare(cam,budget=k)
        a,ia,_=run.prepare(cam,budget=k)
        graph_checks.same(str(i)+'/selector',ids,ia)
        run._pipeline._graph_enabled(False)
        c,ic,_=run.prepare(cam,budget=k)
        graph_checks.same(str(i)+'/ids',ia,ic);graph_checks.packet(str(i),a,c)
        run._pipeline._graph_enabled(True)
    graph_checks.finish()
    state=student.state_dict();name='heads.xyz.weight';saved=state[name].clone()
    baseline=run.prepare(camera(s,0),budget=1000)
    student.heads['xyz'].weight.add_(1)
    again=run.prepare(camera(s,0),budget=1000)
    graph_checks.same('private_student_snapshot',baseline[0]['inputs']['means3D'],again[0]['inputs']['means3D'])
    graph_checks.finish()
    save(ROOT/'reports/GPU_STUDENT_TEST.json',dict(passed=True,cases=cases,max_residual_error=max(errors),
        rtol=3e-5,atol=3e-6,teacher_regression_bitwise=True,selection_unchanged=True,
        graph_native_bitwise=True,private_snapshot=True,performance_measured=False))
    print('GPU_STUDENT_CORRECTNESS_PASS',cases,max(errors),flush=True)


if __name__=='__main__':main()

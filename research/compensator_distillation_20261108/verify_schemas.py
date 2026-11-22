from common import *
from compensation_student import StudentCompensator,teacher_fingerprint
from deployment_runtime import DeploymentRenderer
from distill_compensation import ratio_controls


@torch.no_grad()
def main():
    check=Comparisons('STUDENT_SCHEMAS.json');cases=0;maximum=0.
    for group in ['both_li','selector_li','compensator_li','mode','legacy']:
        s=scene(group);teacher=DeploymentRenderer(s.model,s.gaussians,s.config);b=teacher._base
        student=StudentCompensator.from_teacher(b.spatial);student.teacher_fingerprint=teacher_fingerprint(s.model,s.gaussians)
        for name in ['xyz','rotation','other']:
            student.heads[name].weight.normal_(0,.005)
        run=DeploymentRenderer(s.model,s.gaussians,s.config,student=student)
        for local in ([False,True] if b.joint_schema else [False]):
            for frame,k in [(0,0),(396,767),(791,768),(0,1000),(396,5000)]:
                cam=camera(s,frame)
                _,it,_=teacher.prepare(cam,budget=k,view_conditioned=local)
                a,ia,stats=run.prepare(cam,budget=k,view_conditioned=local)
                check.same(str((group,frame,k,local)),it,ia)
                run._pipeline._graph_enabled(False)
                native,ni,_=run.prepare(cam,budget=k,view_conditioned=local)
                check.same(str((group,k,local,'native')),ia,ni);check.packet(str((group,k,local)),a,native)
                run._pipeline._graph_enabled(True)
                cases+=1
        # Standalone inference with NaN must publish finite failure status.
        cam=camera(s,0);_,ids,stats=teacher.prepare(cam,budget=1,view_conditioned=bool(b.joint_schema))
        ratio=stats['ratio'];weights,code=ratio_controls(ratio,teacher.knots,b.spatial.ratio_encoding_values)
        view=teacher._ext.view(b.anchors.centers,b.view_centers4,b.view_corners4,cam.camera_center.contiguous(),
            cam.world_view_transform,cam.full_proj_transform,b.model.selector_view_scene_scale,
            cam.image_width/cam.image_height,cam.znear) if b.joint_schema and b.joint_schema!=1 else b.empty
        old=run._compensator[0][0,0].clone();run._compensator[0][0,0]=float('nan')
        _,_,status=run._ext.compensate(b.offset,ids,b.owners,b.reference,b.tables,run._compensator,
            b.compensated_master,view,run._view_weight,weights,code,bool(b.joint_schema),b.joint_schema,
            hasattr(b.spatial,'softplus_scale_schema'))
        assert int(status)==2
        run._compensator[0][0,0].copy_(old)
        for enabled in [True,False]:
            a,ia,_=teacher.prepare(cam,budget=1000,view_conditioned=bool(b.joint_schema),compensation=enabled)
            v,iv,_=run.prepare(cam,budget=1000,view_conditioned=bool(b.joint_schema),compensation=enabled)
            check.same(group+'/comp_switch/'+str(enabled),ia,iv)
            if not enabled:check.packet(group+'/uncompensated',a,v)
        check.finish(cases=cases,group_completed=group)
        print('STUDENT_SCHEMA',group,cases,flush=True)
        del teacher,run,s,b,student;torch.cuda.empty_cache()
    check.finish(cases=cases,groups=5,nonfinite_status=True,compensation_off_bitwise=True)


if __name__=='__main__':main()

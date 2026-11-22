from common import *
from compensation_student import load_student,decode_student
from deployment_runtime import DeploymentRenderer
from distill_compensation import TeacherSamples,student_residual,ratio_controls


@torch.no_grad()
def main():
    source=TeacherSamples(CHECKPOINTS['both_li'],DATASET)
    name='fr1_student32_image_calibration' if (ROOT/'models/fr1_student32_image_calibration/best.pt').exists() else 'fr1_student32'
    student,_=load_student(ROOT/'models'/name/'best.pt','cuda',source.fingerprint)
    run=DeploymentRenderer(source.scene.model,source.scene.gaussians,source.scene.config,student=student)
    b=run._base;rows=[]
    for frame in [0,395,790]:
        for ratio in [.01,.05,.1,.3,.5,.7,.03,.08,.2,.4,.6]:
            case=source.sample(frame,ratio,True)
            weights,code=ratio_controls(ratio,run.knots,b.spatial.ratio_encoding_values)
            view=run._ext.view(b.anchors.centers,b.view_centers4,b.view_corners4,case['camera'].camera_center.contiguous(),
                case['camera'].world_view_transform,case['camera'].full_proj_transform,b.model.selector_view_scene_scale,
                case['camera'].image_width/case['camera'].image_height,case['camera'].znear)
            flat,delta,status=run._ext.compensate(b.offset,case['ids'],b.owners,b.reference,b.tables,run._compensator,
                b.compensated_master,view,run._view_weight,weights,code,True,b.joint_schema,case['softplus'])
            truth=student_residual(student,case)
            assert int(status)==0
            error=float((delta-truth).abs().max())
            assert torch.allclose(delta,truth,rtol=1e-4,atol=1e-5),(frame,ratio,error)
            reference=dict(sh_degree=b.pc.active_sh_degree,inputs=decode_student(truth,case['master'],case['softplus']))
            fused=run._packet(flat,len(case['ids']))
            a=render_prepared(case['camera'],reference,source.scene.background)['render']
            v=render_prepared(case['camera'],fused,source.scene.background)['render']
            mse=float((a-v).square().mean());psnr=-10*math.log10(max(mse,1e-20))
            assert psnr>=70,(frame,ratio,psnr)
            rows.append(dict(frame=frame,ratio=ratio,K=len(case['ids']),residual_max_abs=error,
                fused_pytorch_image_psnr=psnr,image_max_abs=float((a-v).abs().max())))
    save(ROOT/'reports/TRAINED_CUDA_PARITY.json',dict(passed=True,cases=len(rows),
        residual_rtol=1e-4,residual_atol=1e-5,min_fused_image_psnr=min(r['fused_pytorch_image_psnr'] for r in rows),
        max_residual_error=max(r['residual_max_abs'] for r in rows),rows=rows,performance_measured=False,student=name))
    print('TRAINED_CUDA_PARITY_PASS',len(rows),flush=True)


if __name__=='__main__':
    import math
    main()

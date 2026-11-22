"""各场景学生的普通前向、CUDA 和 Graph 部署一致性。"""
from scene_runtime import *
import math
from compensation_student import load_student,decode_student
from deployment_runtime import DeploymentRenderer
from distill_compensation import student_residual,ratio_controls


@torch.no_grad()
def main():
    scene,source = load_scene(sys.argv[1])
    folder = ROOT/'reports'/scene['slug']
    preflight = json.loads((folder/'PREFLIGHT.json').read_text())
    path = ROOT/'models'/scene['slug']/'continuous_calibration/best.pt'
    student,_ = load_student(path,'cuda',source.fingerprint)
    run = DeploymentRenderer(source.scene.model,source.scene.gaussians,source.scene.config,student=student)
    b = run._base
    checks = Comparisons(scene['slug']+'/GRAPH_PARITY.json')
    rows = []
    for frame in preflight['validation_views']:
        for ratio in [.01,.05,.1,.3,.5,.7,.03,.08,.2,.4,.6]:
            for local in source.local_modes:
                case = source.sample(frame,ratio,local)
                encoded = b.spatial.encode(case['inputs']) + b.spatial.ratio_projection(case['ratio_code'])
                if case['condition'] is not None:
                    encoded = encoded + b.spatial.view_projection(case['condition'])
                hidden = torch.relu(b.spatial.body(torch.relu(encoded)))
                raw = torch.cat([head(hidden) for head in b.spatial.heads.values()],dim=1)
                teacher = torch.cat((raw[:,:3]*case['reference'],raw[:,3:]),dim=1)
                assert torch.allclose(teacher,case['target'],rtol=1e-4,atol=1e-5)
                weights,code = ratio_controls(ratio,run.knots,b.spatial.ratio_encoding_values)
                view = b.empty
                cam = case['camera']
                if local:
                    view = run._ext.view(b.anchors.centers,b.view_centers4,b.view_corners4,
                        cam.camera_center.contiguous(),cam.world_view_transform,cam.full_proj_transform,
                        b.model.selector_view_scene_scale,cam.image_width/cam.image_height,cam.znear)
                flat,delta,status = run._ext.compensate(b.offset,case['ids'],b.owners,b.reference,b.tables,
                    run._compensator,b.compensated_master,view,run._view_weight,weights,code,
                    local,b.joint_schema,case['softplus'])
                target = student_residual(student,case)
                assert int(status)==0
                if not torch.allclose(delta,target,rtol=1e-4,atol=1e-5):
                    raise AssertionError((scene['id'],frame,ratio,local,float((delta-target).abs().max())))
                packet = dict(sh_degree=b.pc.active_sh_degree,inputs=decode_student(target,case['master'],case['softplus']))
                original = render_prepared(cam,packet,source.scene.background)['render']
                fused = render_prepared(cam,run._packet(flat,len(case['ids'])),source.scene.background)['render']
                mse = float((original-fused).square().mean())
                psnr = -10*math.log10(max(mse,1e-20))
                assert psnr>=70,(scene['id'],frame,ratio,local,psnr)
                if local:
                    first,ids,_ = run.prepare(cam,ratio_budget=ratio)
                    checks.same(f'{frame}/{ratio}/teacher_ids',ids,case['ids'])
                    run._pipeline._graph_enabled(False)
                    second,other_ids,_ = run.prepare(cam,ratio_budget=ratio)
                    run._pipeline._graph_enabled(True)
                    checks.same(f'{frame}/{ratio}/ids',ids,other_ids)
                    checks.packet(f'{frame}/{ratio}',first,second)
                    image = render_prepared(cam,first,source.scene.background)['render']
                    deployed_mse = float((original-image).square().mean())
                    deployed_psnr = -10*math.log10(max(deployed_mse,1e-20))
                    assert deployed_psnr>=70,(scene['id'],frame,ratio,deployed_psnr)
                rows.append(dict(frame=frame,ratio=ratio,local=local,K=len(case['ids']),
                    max_residual_abs=float((delta-target).abs().max()),pytorch_cuda_psnr=psnr))
                rows[-1]['teacher_cuda_max_abs'] = float((teacher-case['target']).abs().max())
    checks.finish()
    save(folder/'DEPLOYMENT_VALIDATION.json',dict(passed=True,cases=len(rows),rows=rows,
        parameters=3534,student_sha256=sha(path),teacher_fingerprint=source.fingerprint,
        min_pytorch_cuda_psnr=min(r['pytorch_cuda_psnr'] for r in rows),
        max_residual_abs=max(r['max_residual_abs'] for r in rows),speed_measured=False))
    print('DEPLOYMENT_VALIDATION_PASS',scene['id'],len(rows),flush=True)


if __name__ == '__main__':
    main()

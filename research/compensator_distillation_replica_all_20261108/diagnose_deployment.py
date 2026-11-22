"""保留部署门限，定位普通学生与融合部署的数值差异，不修改模型。"""
from scene_runtime import *
import math
from compensation_student import load_student, decode_student
from deployment_runtime import DeploymentRenderer
from distill_compensation import student_residual, ratio_controls


def image_psnr(a, b):
    return -10 * math.log10(max(float((a-b).square().mean()), 1e-20))


@torch.no_grad()
def main():
    scene, source = load_scene(sys.argv[1])
    folder = ROOT/'reports'/scene['slug']
    preflight = json.loads((folder/'PREFLIGHT.json').read_text())
    path = ROOT/'models'/scene['slug']/'continuous_calibration/best.pt'
    student, _ = load_student(path, 'cuda', source.fingerprint)
    run = DeploymentRenderer(source.scene.model, source.scene.gaussians, source.scene.config, student=student)
    b = run._base
    checks = Comparisons(scene['slug']+'/GRAPH_PARITY_DIAGNOSTIC.json')
    rows = []
    for frame in preflight['validation_views']:
        for ratio in [.01,.05,.1,.3,.5,.7,.03,.08,.2,.4,.6]:
            for local in source.local_modes:
                case = source.sample(frame, ratio, local)
                weights, code = ratio_controls(ratio, run.knots, b.spatial.ratio_encoding_values)
                cam = case['camera']
                view = b.empty
                if local:
                    view = run._ext.view(b.anchors.centers, b.view_centers4, b.view_corners4,
                        cam.camera_center.contiguous(), cam.world_view_transform, cam.full_proj_transform,
                        b.model.selector_view_scene_scale, cam.image_width/cam.image_height, cam.znear)
                flat, delta, status = run._ext.compensate(b.offset, case['ids'], b.owners,
                    b.reference, b.tables, run._compensator, b.compensated_master, view,
                    run._view_weight, weights, code, local, b.joint_schema, case['softplus'])
                assert int(status)==0
                target = student_residual(student, case)
                original = dict(sh_degree=b.pc.active_sh_degree,
                    inputs=decode_student(target, case['master'], case['softplus']))
                raw_only = dict(sh_degree=b.pc.active_sh_degree,
                    inputs=decode_student(delta, case['master'], case['softplus']))
                fused = run._packet(flat, len(case['ids']))
                images = [render_prepared(cam, p, source.scene.background)['render']
                    for p in [original, raw_only, fused]]
                row = dict(frame=frame, ratio=ratio, local=local, K=len(case['ids']),
                    residual_allclose=bool(torch.allclose(delta,target,rtol=1e-4,atol=1e-5)),
                    residual_max_abs=float((delta-target).abs().max()),
                    pytorch_cuda_psnr=image_psnr(images[0],images[2]),
                    raw_only_psnr=image_psnr(images[0],images[1]),
                    decode_only_psnr=image_psnr(images[1],images[2]),
                    image_max_abs=float((images[0]-images[2]).abs().max()))
                row['attribute_max_abs'] = {k:float((original['inputs'][k]-v).abs().max())
                    if v.numel() else 0. for k,v in fused['inputs'].items() if isinstance(v,torch.Tensor)}
                if row['pytorch_cuda_psnr']<70:
                    row['field_replacement_psnr'] = {}
                    for key in ['means3D','opacities','scales','rotations','dc']:
                        packet = dict(sh_degree=fused['sh_degree'], inputs=dict(fused['inputs']))
                        packet['inputs'][key] = original['inputs'][key]
                        image = render_prepared(cam,packet,source.scene.background)['render']
                        row['field_replacement_psnr'][key] = image_psnr(images[0],image)
                if local:
                    first, ids, _ = run.prepare(cam, ratio_budget=ratio)
                    checks.same(f'{frame}/{ratio}/teacher_ids',ids,case['ids'])
                    run._pipeline._graph_enabled(False)
                    second, other_ids, _ = run.prepare(cam,ratio_budget=ratio)
                    run._pipeline._graph_enabled(True)
                    checks.same(f'{frame}/{ratio}/ids',ids,other_ids)
                    checks.packet(f'{frame}/{ratio}',first,second)
                rows.append(row)
    checks.finish()
    failed = [r for r in rows if r['pytorch_cuda_psnr']<70 or not r['residual_allclose']]
    result = dict(cases=len(rows), original_threshold_psnr=70, failed_cases=failed, rows=rows,
        student_sha256=sha(path), graph_bitwise_passed=True, models_modified=False,
        min_pytorch_cuda_psnr=min(r['pytorch_cuda_psnr'] for r in rows))
    save(folder/'DEPLOYMENT_DIAGNOSTIC.json',result)
    print('DEPLOYMENT_DIAGNOSTIC',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    main()

"""训练恢复、配置拒绝和新权重专用 CUDA 一致性。"""
from common import *
import csv
import math
import distill_compensation
from compensation_student import load_student, decode_student
from deployment_runtime import DeploymentRenderer
from distill_compensation import TeacherSamples, student_residual, ratio_controls
from diagnose import BASELINE


def invoke(arguments):
    sys.argv = ['distill_compensation.py', *arguments]
    distill_compensation.main()


def resume_check():
    refinement = (ROOT/'reports/FINAL_CANDIDATE_SELECTION.json').exists()
    output = ROOT/'models'/('continuous_resume_test' if refinement else 'objective_resume_test')
    arguments = ['train', str(CHECKPOINTS['both_li']), '--dataset', DATASET, '--output', str(output),
        '--initialize', str(BASELINE), '--steps', '2', '--batch-points', '256', '--image-every', '1',
        '--image-weight', '200', '--residual-weight', '.01', '--validation-metric', 'image_mse',
        '--validate-every', '2', '--lr', '.0002']
    if refinement:
        arguments += ['--ratio-jitter', '.5', '--paired-view-sampling']
    invoke(arguments)
    arguments = [v for v in arguments]
    begin = arguments.index('--initialize')
    arguments[begin:begin+2] = ['--resume', str(output/'last.pt')]
    arguments[arguments.index('--steps')+1] = '4'
    invoke(arguments)
    rows = list(csv.DictReader((output/'training.csv').open()))
    assert [int(r['step']) for r in rows] == [1, 2, 3, 4]
    student, state = load_student(output/'last.pt')
    assert state['step'] == 4 and state['metadata']['configuration']['residual_weight'] == .01
    assert len(state['optimizer']['state']) == len(list(student.parameters()))
    arguments[arguments.index('--steps')+1] = '6'
    arguments[arguments.index('--residual-weight')+1] = '1'
    try:
        invoke(arguments)
    except ValueError as error:
        assert str(error) == 'Resume configuration changed at residual_weight'
    else:
        raise AssertionError('Changed objective was accepted on resume')
    if refinement:
        arguments[arguments.index('--residual-weight')+1] = '.01'
        arguments[arguments.index('--ratio-jitter')+1] = '0'
        try:
            invoke(arguments)
        except ValueError as error:
            assert str(error) == 'Resume configuration changed at ratio_jitter'
        else:
            raise AssertionError('Changed Ratio distribution was accepted on resume')
        arguments[arguments.index('--ratio-jitter')+1] = '.5'
        arguments.remove('--paired-view-sampling')
        try:
            invoke(arguments)
        except ValueError as error:
            assert str(error) == 'Resume configuration changed at paired_view_sampling'
        else:
            raise AssertionError('Changed view RNG was accepted on resume')
    assert len(list(csv.DictReader((output/'training.csv').open()))) == 4
    save(ROOT/'reports/RESUME_TEST.json', dict(passed=True, actual_steps=4,
        adam_state_restored=True, changed_objective_rejected=True, teacher_binding=True))


@torch.no_grad()
def cuda_check():
    selection_path = ROOT/'reports/FINAL_CANDIDATE_SELECTION.json'
    if not selection_path.exists():
        selection_path = ROOT/'reports/CANDIDATE_SELECTION.json'
    selected = json.loads(selection_path.read_text())['selected']
    source = TeacherSamples(CHECKPOINTS['both_li'], DATASET)
    student, _ = load_student(ROOT/'models'/selected/'best.pt', 'cuda', source.fingerprint)
    run = DeploymentRenderer(source.scene.model, source.scene.gaussians, source.scene.config, student=student)
    b = run._base
    rows = []
    comparison = Comparisons('DEPLOYMENT_GRAPH_CHECK.json')
    for frame in [10, 200, 405, 605, 780]:
        for ratio in [.01, .05, .1, .3, .5, .7, .03, .08, .2, .4, .6]:
            for local in source.local_modes:
                case = source.sample(frame, ratio, local)
                weights, code = ratio_controls(ratio, run.knots, b.spatial.ratio_encoding_values)
                view = b.empty
                if local:
                    cam = case['camera']
                    view = run._ext.view(b.anchors.centers, b.view_centers4, b.view_corners4,
                        cam.camera_center.contiguous(), cam.world_view_transform, cam.full_proj_transform,
                        b.model.selector_view_scene_scale, cam.image_width/cam.image_height, cam.znear)
                flat, delta, status = run._ext.compensate(b.offset, case['ids'], b.owners, b.reference,
                    b.tables, run._compensator, b.compensated_master, view, run._view_weight, weights, code,
                    local, b.joint_schema, case['softplus'])
                truth = student_residual(student, case)
                assert int(status) == 0
                assert torch.allclose(delta, truth, rtol=1e-4, atol=1e-5), (frame, ratio, local)
                packet = dict(sh_degree=b.pc.active_sh_degree,
                    inputs=decode_student(truth, case['master'], case['softplus']))
                fused = run._packet(flat, len(case['ids']))
                a = render_prepared(case['camera'], packet, source.scene.background)['render']
                v = render_prepared(case['camera'], fused, source.scene.background)['render']
                mse = float((a-v).square().mean())
                psnr = -10*math.log10(max(mse, 1e-20))
                assert psnr >= 70, (frame, ratio, local, psnr)
                deployed_psnr = psnr
                if local:
                    first, ids, _ = run.prepare(case['camera'], ratio_budget=ratio)
                    comparison.same(f'{frame}/{ratio}/teacher_ids', ids, case['ids'])
                    deployed = render_prepared(case['camera'], first, source.scene.background)['render']
                    deployed_mse = float((a-deployed).square().mean())
                    deployed_psnr = -10*math.log10(max(deployed_mse, 1e-20))
                    assert deployed_psnr >= 70, (frame, ratio, deployed_psnr)
                    run._pipeline._graph_enabled(False)
                    second, second_ids, _ = run.prepare(case['camera'], ratio_budget=ratio)
                    run._pipeline._graph_enabled(True)
                    comparison.same(f'{frame}/{ratio}/ids', ids, second_ids)
                    comparison.packet(f'{frame}/{ratio}', first, second)
                rows.append(dict(frame=frame, ratio=ratio, local=local, K=len(case['ids']),
                    max_residual_abs=float((delta-truth).abs().max()), image_psnr=psnr,
                    deployed_image_psnr=deployed_psnr,
                    max_image_abs=float((a-v).abs().max())))
    comparison.finish()
    save(ROOT/'reports/TRAINED_CUDA_PARITY.json', dict(passed=True, selected=selected, cases=len(rows),
        residual_rtol=1e-4, residual_atol=1e-5, graph_native_bitwise=True,
        min_image_psnr=min(r['image_psnr'] for r in rows),
        min_deployed_image_psnr=min(r['deployed_image_psnr'] for r in rows),
        max_residual_abs=max(r['max_residual_abs'] for r in rows), rows=rows))
    print('CUDA_PARITY_PASS', len(rows), flush=True)


if __name__ == '__main__':
    resume_check()
    cuda_check()

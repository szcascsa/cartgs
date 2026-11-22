"""同一选点、同一新留出集评价学生，保存数值且不保存图片。"""
from common import *
import csv
import math
from compensation_student import load_student, distillation_loss, decode_student
from deployment_runtime import DeploymentRenderer
from distill_compensation import TeacherSamples, student_residual, render_student
from renderer import render_prepared
from diagnose import BASELINE


def psnr(image, target):
    mse = float((image.clamp(0, 1)-target.clamp(0, 1)).square().mean())
    return -10*math.log10(max(mse, 1e-12))


def main():
    protocol = json.loads((ROOT/'reports/EXPERIMENT_PROTOCOL.json').read_text())
    paths = dict(previous=BASELINE, control=ROOT/'models/control/best.pt',
        residual001=ROOT/'models/residual001/best.pt')
    eligible_arms = ['control', 'residual001']
    splits = [('fresh', protocol['fresh_test_frames']), ('development', protocol['old_test_frames'])]
    output_prefix = 'QUALITY'
    selection_file = 'CANDIDATE_SELECTION.json'
    if len(sys.argv) > 1:
        plan = json.loads(Path(sys.argv[1]).read_text())
        paths = {name:Path(path) for name, path in plan['checkpoints'].items()}
        eligible_arms = plan['eligible_arms']
        splits = list(plan['splits'].items())
        output_prefix = plan['output_prefix']
        selection_file = plan['selection_file']
    source = TeacherSamples(CHECKPOINTS['both_li'], DATASET)
    students = {}
    runtimes = {}
    identities = {}
    for name, path in paths.items():
        student, payload = load_student(path, 'cuda', source.fingerprint)
        students[name] = student
        identities[name] = dict(path=str(path), sha256=sha(path), step=payload['step'],
            validation=payload['metadata'].get('best_validation'),
            initial_validation=payload['metadata'].get('initial_validation'))
        runtimes[name] = DeploymentRenderer(source.scene.model, source.scene.gaussians,
            source.scene.config, student=student)
    selected = min(eligible_arms, key=lambda name: identities[name]['validation'])
    save(ROOT/'reports'/selection_file, dict(metric='validation_teacher_image_mse',
        selected=selected, identities=identities, selection_before_new_test=True))
    rows = []
    for label, frames in splits:
        for frame in frames:
            for ratio in [.01, .05, .1, .3, .5, .7, .03, .08, .2, .4, .6]:
                for local in source.local_modes:
                    with torch.no_grad():
                        case = source.sample(frame, ratio, local)
                        target = render_prepared(case['camera'], case['packet'], source.scene.background)['render']
                        gt = source.scene.ground_truth(frame)
                        if gt is None:
                            raise RuntimeError('Missing GT at frame ' + str(frame))
                        teacher_psnr = psnr(target, gt)
                        teacher_attributes = decode_student(case['target'], case['master'], case['softplus'])
                        teacher_shift = case['target'][:, :3].norm(dim=1)/case['reference'].flatten()
                        for name, student in students.items():
                            residual = student_residual(student, case)
                            if local:
                                packet, ids, _ = runtimes[name].prepare(case['camera'], ratio_budget=ratio)
                                assert torch.equal(ids, case['ids']), (name, frame, ratio)
                                image = render_prepared(case['camera'], packet, source.scene.background)['render']
                            else:
                                image = render_student(student, case, source.scene.background,
                                    source.runtime._base.pc.active_sh_degree, residual)
                            metrics = distillation_loss(student, case, residual)
                            if any(not torch.isfinite(v) for v in metrics.values()):
                                raise RuntimeError('Nonfinite evaluation attribute loss')
                            student_psnr = psnr(image, gt)
                            attributes = decode_student(residual, case['master'], case['softplus'])
                            if any(not torch.isfinite(v).all() for v in attributes.values() if isinstance(v, torch.Tensor)):
                                raise RuntimeError('Nonfinite student attributes during evaluation')
                            shift = residual[:, :3].norm(dim=1)/case['reference'].flatten()
                            shift_error = (residual[:, :3]-case['target'][:, :3]).norm(dim=1)/case['reference'].flatten()
                            scale_ratio = (attributes['scales']/teacher_attributes['scales']).flatten()
                            rows.append(dict(split=label, student=name, frame=frame, ratio=ratio,
                                local=local, K=len(case['ids']), teacher_gt_psnr=teacher_psnr,
                                student_gt_psnr=student_psnr, psnr_loss=teacher_psnr-student_psnr,
                                teacher_student_psnr=psnr(image, target),
                                teacher_student_mse=float((image-target).square().mean()),
                                teacher_xyz_p99=float(torch.quantile(teacher_shift, .99)),
                                student_xyz_p99=float(torch.quantile(shift, .99)),
                                student_xyz_max=float(shift.max()), xyz_error_p99=float(torch.quantile(shift_error, .99)),
                                scale_ratio_p01=float(torch.quantile(scale_ratio, .01)),
                                scale_ratio_p99=float(torch.quantile(scale_ratio, .99)), scale_ratio_max=float(scale_ratio.max()),
                                **{key:float(value) for key, value in metrics.items()}))
            print('EVAL', label, frame, 'rows', len(rows), flush=True)
        with (ROOT/'reports'/f'{output_prefix}_{label.upper()}.csv').open('w', newline='') as f:
            subset = [row for row in rows if row['split'] == label]
            writer = csv.DictWriter(f, fieldnames=list(subset[0]))
            writer.writeheader()
            writer.writerows(subset)
    summaries = []
    for split, _ in splits:
        for name in paths:
            for ratio in [.01, .05, .1, .3, .5, .7, .03, .08, .2, .4, .6]:
                for local in source.local_modes:
                    group = [r for r in rows if r['split'] == split and r['student'] == name
                        and r['ratio'] == ratio and r['local'] == local]
                    losses = [r['psnr_loss'] for r in group]
                    summaries.append(dict(split=split, student=name, ratio=ratio, local=local,
                        views=len(group), teacher_psnr=float(np.mean([r['teacher_gt_psnr'] for r in group])),
                        student_psnr=float(np.mean([r['student_gt_psnr'] for r in group])),
                        mean_loss=float(np.mean(losses)), p99_loss=float(np.percentile(losses, 99)),
                        worst_loss=max(losses), teacher_student_mse=float(np.mean([r['teacher_student_mse'] for r in group])),
                        teacher_student_psnr=float(np.mean([r['teacher_student_psnr'] for r in group])),
                        quality_gate=bool(np.mean(losses) <= .01 and np.percentile(losses, 99) <= .05 and max(losses) <= .1)))
    with (ROOT/'reports'/f'{output_prefix}_SUMMARY.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    save(ROOT/'reports'/f'{output_prefix}_SUMMARY.json', dict(selected=selected, identities=identities,
        case_comparisons=len(rows), splits=dict(splits), fresh_frames=dict(splits).get('fresh', []),
        development_frames=dict(splits).get('development', []), summaries=summaries, images_saved=0,
        selector_ids_identical=True, speed_measured=False))
    print('QUALITY_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()

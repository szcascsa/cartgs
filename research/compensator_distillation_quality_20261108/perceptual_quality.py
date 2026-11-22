"""新留出集的 SSIM、LPIPS 检查，不用于模型选择。"""
from common import *
import csv
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from compensation_student import load_student
from deployment_runtime import DeploymentRenderer
from distill_compensation import TeacherSamples, render_student
from diagnose import BASELINE


@torch.no_grad()
def main():
    protocol = json.loads((ROOT/'reports/EXPERIMENT_PROTOCOL.json').read_text())
    refinement = ROOT/'reports/REFINEMENT_PROTOCOL.json'
    selection_path = ROOT/'reports/CANDIDATE_SELECTION.json'
    if refinement.exists():
        protocol = json.loads(refinement.read_text())
        selection_path = ROOT/'reports/FINAL_CANDIDATE_SELECTION.json'
    selection = json.loads(selection_path.read_text())
    selected = selection['selected']
    source = TeacherSamples(CHECKPOINTS['both_li'], DATASET)
    students = {}
    runtimes = {}
    for name, path in [('previous', BASELINE), (selected, ROOT/'models'/selected/'best.pt')]:
        students[name], _ = load_student(path, 'cuda', source.fingerprint)
        runtimes[name] = DeploymentRenderer(source.scene.model, source.scene.gaussians,
            source.scene.config, student=students[name])
    perceptual = LearnedPerceptualImagePatchSimilarity(normalize=True, reduction='none').cuda()
    rows = []
    for frame in protocol['fresh_test_frames']:
        gt = source.scene.ground_truth(frame).clamp(0, 1)
        for ratio in [.01, .05, .1, .3, .5, .7, .03, .08, .2, .4, .6]:
            for local in source.local_modes:
                case = source.sample(frame, ratio, local)
                teacher_image = render_prepared(case['camera'], case['packet'], source.scene.background)['render'].clamp(0, 1)
                images = [teacher_image]
                names = ['teacher'] + list(students)
                for name, student in students.items():
                    if local:
                        packet, ids, _ = runtimes[name].prepare(case['camera'], ratio_budget=ratio)
                        assert torch.equal(ids, case['ids'])
                        image = render_prepared(case['camera'], packet, source.scene.background)['render']
                    else:
                        image = render_student(student, case, source.scene.background,
                            source.runtime._base.pc.active_sh_degree)
                    images.append(image.clamp(0, 1))
                predictions = torch.stack(images)
                targets = gt.unsqueeze(0).expand_as(predictions)
                structure = structural_similarity_index_measure(predictions, targets, data_range=1., reduction='none')
                distance = perceptual(predictions, targets).flatten()
                perceptual.reset()
                values = torch.stack([structure, distance], dim=1)
                assert torch.isfinite(values).all()
                for name, value in zip(names, values.cpu().tolist()):
                    rows.append(dict(student=name, frame=frame, ratio=ratio, local=local,
                        K=len(case['ids']), ssim=value[0], lpips=value[1]))
        print('PERCEPTUAL', frame, 'rows', len(rows), flush=True)
    with (ROOT/'reports/PERCEPTUAL_QUALITY.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for name in ['teacher', *students]:
        for ratio in [.01, .05, .1, .3, .5, .7, .03, .08, .2, .4, .6]:
            for local in source.local_modes:
                group = [r for r in rows if r['student'] == name and r['ratio'] == ratio and r['local'] == local]
                summary.append(dict(student=name, ratio=ratio, local=local, views=len(group),
                    ssim=float(np.mean([r['ssim'] for r in group])), lpips=float(np.mean([r['lpips'] for r in group]))))
    save(ROOT/'reports/PERCEPTUAL_QUALITY.json', dict(selected=selected, summaries=summary,
        cases=len(rows), used_for_checkpoint_selection=False, images_saved=0))
    print('PERCEPTUAL_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()

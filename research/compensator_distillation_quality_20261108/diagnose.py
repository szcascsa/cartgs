"""逐项测量蒸馏梯度和 Head 渲染影响，不更新参数。"""
from common import *
from compensation_student import load_student, distillation_loss
from distill_compensation import TeacherSamples, student_residual, render_student
from renderer import render_prepared


BASELINE = Path('/home/prmi/zwj/experiments/compensator_distillation_20261108/models/fr1_student32_image_calibration/best.pt')


def flatten(gradients, parameters):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten()
        for g, p in zip(gradients, parameters)])


def main():
    source = TeacherSamples(CHECKPOINTS['both_li'], DATASET)
    student, payload = load_student(BASELINE, 'cuda', source.fingerprint)
    parameters = list(student.parameters())
    rows = []
    for frame in [0, 395, 790]:
        for ratio in [.01, .05, .1, .3, .5, .7]:
            for local in source.local_modes:
                case = source.sample(frame, ratio, local)
                residual = student_residual(student, case)
                losses = distillation_loss(student, case, residual)
                gradients = {name: flatten(torch.autograd.grad(value, parameters, retain_graph=True,
                    allow_unused=True), parameters) for name, value in losses.items()}
                image = render_student(student, case, source.scene.background,
                    source.runtime._base.pc.active_sh_degree, residual)
                with torch.no_grad():
                    target = render_prepared(case['camera'], case['packet'], source.scene.background)['render']
                photo_loss = (image - target).square().mean()
                photo = flatten(torch.autograd.grad(photo_loss, parameters, retain_graph=True,
                    allow_unused=True), parameters)
                total = sum(gradients.values())
                row = dict(frame=frame, ratio=ratio, local=local, K=len(case['ids']),
                    photo_loss=float(photo_loss), attribute_loss=float(sum(losses.values())),
                    attribute_gradient=float(total.norm()), photo_gradient=float(photo.norm()),
                    weighted_photo_to_attribute=float(200 * photo.norm() / total.norm().clamp_min(1e-20)),
                    attribute_photo_cosine=float(torch.dot(photo, total) /
                        (photo.norm() * total.norm()).clamp_min(1e-20)), losses={})
                for name, gradient in gradients.items():
                    row['losses'][name] = dict(value=float(losses[name]), gradient=float(gradient.norm()),
                        photo_cosine=float(torch.dot(photo, gradient) /
                            (photo.norm() * gradient.norm()).clamp_min(1e-20)))
                with torch.no_grad():
                    mse = float((image - target).square().mean())
                    row['replace_head_mse'] = dict(student=mse)
                    for head, begin, end in [('xyz', 0, 3), ('scale', 3, 6),
                            ('rotation', 6, 10), ('other', 10, 14)]:
                        replaced = residual.detach().clone()
                        replaced[:, begin:end] = case['target'][:, begin:end]
                        rendered = render_student(student, case, source.scene.background,
                            source.runtime._base.pc.active_sh_degree, replaced)
                        row['replace_head_mse'][head] = float((rendered - target).square().mean())
                rows.append(row)
                print('DIAG', frame, ratio, local, row['weighted_photo_to_attribute'],
                    row['attribute_photo_cosine'], flush=True)
                save(ROOT/'reports/DIAGNOSIS.json', dict(student=str(BASELINE), step=payload['step'],
                    parameter_updates=0, rows=rows))
    print('DIAGNOSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

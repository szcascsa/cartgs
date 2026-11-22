"""新增目标的权重、旧默认和真实图像验证口径回归。"""
from common import *
from compensation_student import load_student, distillation_loss
from distill_compensation import (TeacherSamples, student_residual, render_student,
    attribute_objective, validation_loss)
from renderer import render_prepared
from diagnose import BASELINE


def main():
    source = TeacherSamples(CHECKPOINTS['both_li'], DATASET)
    student, _ = load_student(BASELINE, 'cuda', source.fingerprint)
    case = source.sample(395, .1, True)
    values = distillation_loss(student, case, student_residual(student, case))
    assert torch.equal(attribute_objective(values), sum(values.values()))
    parameters = list(student.parameters())
    expected = .01*values['residual'] + sum(v for k, v in values.items() if k != 'residual')
    actual = attribute_objective(values, .01)
    ga = torch.autograd.grad(actual, parameters, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(expected, parameters, allow_unused=True)
    assert torch.equal(actual, expected)
    assert all(a is None and b is None or a is not None and b is not None and torch.equal(a, b)
        for a, b in zip(ga, gb))
    with torch.no_grad():
        manual = []
        for local in source.local_modes:
            sample = source.sample(395, .1, local)
            teacher_image = render_prepared(sample['camera'], sample['packet'], source.scene.background)['render']
            student_image = render_student(student, sample, source.scene.background,
                source.runtime._base.pc.active_sh_degree)
            manual.append(float((teacher_image-student_image).square().mean()))
        for weight in [1., .01, 0.]:
            value = validation_loss(student, source, [395], [.1], 761, 200., weight, 'image_mse')
            assert value == float(np.mean(manual)), (value, manual)
    assert all(p.grad is None for p in source.runtime._base.model.parameters())
    assert all(p.grad is None for p in source.runtime._base.base_master)
    save(ROOT/'reports/OBJECTIVE_TEST.json', dict(passed=True, old_default_bitwise=True,
        reweight_gradient_bitwise=True, image_validation_independent_of_attribute_weight=True,
        teacher_gradients_absent=True))
    print('OBJECTIVE_TEST_PASS', flush=True)


if __name__ == '__main__':
    main()

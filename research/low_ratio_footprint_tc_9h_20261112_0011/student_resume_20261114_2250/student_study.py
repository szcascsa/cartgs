"""沿用冻结场景与原图协议，所有网络实验强制绑定匹配Student。"""
from pathlib import Path
import sys
import cProfile

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from joint_study import *
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(HERE))

CONDITIONS = [f'local{x:g}' for x in [.01, .03, .05, .1, .2, .3, .4, .5, .7]]
CONDITIONS += ['fixed0', 'fixed1', 'fixed1000', 'fixed5000', 'full']
PRIMARY = [f'local{x:g}' for x in [.1, .2, .3, .4, .5]]


def student_specs(tune=False):
    known = json.loads((HERE / 'reports/EXISTING_STUDENTS.json').read_text())
    return [spec for spec in SCENES if spec['id'] in known and (not tune or spec['slug'] in TUNE)]


def load_case(spec):
    _, source = load_source(spec['slug'])
    student, path, payload = student_for_spec(source, spec)
    assert student is not None, ('Student缺失', spec['id'])
    frozen = json.loads((HERE / 'reports/EXISTING_STUDENTS.json').read_text())[spec['id']]
    assert sha(path) == frozen['student_sha256']
    assert student.teacher_fingerprint == source.fingerprint
    student.eval().requires_grad_(False)
    return source, student, path


def identity(spec, source, student, path):
    return dict(scene=spec['id'], slug=spec['slug'], checkpoint=spec['checkpoint'],
        student=str(path), student_sha256=sha(path), fingerprint=source.fingerprint,
        parameters=sum(p.numel() for p in student.parameters()),
        structure={name: list(p.shape) for name, p in student.named_parameters()},
        selector_parameters={name: list(p.shape) for name, p in source.scene.model.named_parameters()
                             if 'selector' in name.lower()}, N=source.runtime.total)

"""最终候选必须在开发集冻结后读取；所有方法使用同一Student与Selector权重。"""
from student_study import *
from pending_runtime import PendingRenderer
from pose_runtime import PoseRenderer
from members_runtime import MembersStudentRenderer


def make_final(source, student, mode=38, **overrides):
    frozen = json.loads((HERE/'reports/FROZEN_STUDENT.json').read_text())
    config = dict(frozen['config'])
    config.update(overrides)
    return MembersStudentRenderer(source, student, mode, **config)


def final_engines(source, student):
    networks = network_variants(source, student)
    renders = {'F2': BaselineRenderer(), 'new': Renderer(38), 'alternative': Renderer(37)}
    variants = benchmark_variants(networks, renders)
    dispatch = {
        'F_native': PendingRenderer(source, student, 38, native_dispatch=True),
        'G_cpp_pose': PoseRenderer(source, student, 38),
        'H_final': make_final(source, student),
        'H_no_camera': make_final(source, student, reuse=False),
        'H_no_finish': make_final(source, student, finish=False),
        'H_no_members': make_final(source, student, members=12),
        'H_alternative37': make_final(source, student, mode=37),
    }
    return networks, renders, variants, dispatch


def dispatch_frame(name, engine, source, frame, background, condition, factory):
    if name == 'F_native':
        return engine.render(factory.from_source(source, frame), background, condition)
    return engine.render_from_source(source, frame, background, condition, factory)

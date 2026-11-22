"""Student组合验收：历史38和本轮182明确分开，所有执行路径使用同一权重。"""
from student_study import *
from pending_runtime import PendingRenderer
from pose_runtime import PoseRenderer
from members_runtime import MembersStudentRenderer
from joint_final_runtime import FinalStudentRenderer
from encoded_sign_runtime import EncodedSignRenderer


def final_engines(source,student):
    networks=network_variants(source,student)
    renders={'F2':BaselineRenderer(),'new':EncodedSignRenderer(182),'old38':Renderer(38)}
    variants=benchmark_variants(networks,renders)
    variants['E_tensor_best_old38']=(networks['tensor_best'],renders['old38'])
    config=json.loads((HERE/'reports/FROZEN_STUDENT.json').read_text())['config']
    def native(**changes):return FinalStudentRenderer(source,student,182,**{**config,**changes})
    dispatch={
        'F_native':PendingRenderer(source,student,38,native_dispatch=True),
        'G_cpp_pose':PoseRenderer(source,student,38),
        'H_old38':MembersStudentRenderer(source,student,38,**config),
        'I_final':native(),
        'I_no_camera':native(reuse=False),
        'I_no_finish':native(finish=False),
        'I_no_members':native(members=12),
        'I_no_fused':native(fused=False),
    }
    return networks,renders,variants,dispatch


def dispatch_frame(name,engine,source,frame,background,condition,factory):
    if name=='F_native':return engine.render(factory.from_source(source,frame),background,condition)
    return engine.render_from_source(source,frame,background,condition,factory)

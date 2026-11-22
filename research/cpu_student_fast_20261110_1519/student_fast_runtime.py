"""可选蒸馏学生CPU入口；复用已冻结AVX2/直方图/6点微核及原生交接。"""
from pathlib import Path
import hashlib
import copy
import sys
import torch

HERE=Path(__file__).resolve().parent
BASE=HERE
sys.path.insert(0,str(BASE))
from kernel_runtime import CpuRenderer as TeacherCpuRenderer,extension


class CpuRenderer(TeacherCpuRenderer):
    @torch.no_grad()
    def __init__(self,model,pc,noise,threads=6,student=None,*,student_checkpoint=None,student_kernel=2,student_decode=1,student_context=1,**kwargs):
        from compensation_student import load_student,teacher_fingerprint
        if student is not None and student_checkpoint is not None:
            raise ValueError('Use a student instance or a student checkpoint, not both')
        fingerprint=None
        if student_checkpoint is not None or student is not None:
            fingerprint=teacher_fingerprint(model,pc)
            if student_checkpoint is not None:
                student_checkpoint=Path(student_checkpoint)
                student,_=load_student(student_checkpoint,'cpu',fingerprint)
            else:
                if student.teacher_fingerprint!=fingerprint:
                    raise ValueError('Student belongs to a different teacher, map or Gaussian row order')
                student=copy.deepcopy(student)
            spatial=model.spatial_compensation
            if hasattr(student,'view_projection')!=hasattr(spatial,'view_projection'):
                raise ValueError('Student and Teacher view schemas differ')
            if next(student.parameters()).device.type!='cpu':
                raise ValueError('Student CPU inference requires CPU weights')
            student.eval().requires_grad_(False)
        super().__init__(model,pc,noise,threads,student=student,student_kernel=student_kernel,student_decode=student_decode,student_context=student_context,**kwargs)
        self.compensator_info=dict(kind='teacher' if student is None else 'student',
            width=64 if student is None else 32,teacher_fingerprint=fingerprint,
            student_checkpoint=None if student_checkpoint is None else str(student_checkpoint),
            student_sha256=None if student_checkpoint is None else hashlib.sha256(student_checkpoint.read_bytes()).hexdigest(),
            parameters=sum(v.numel() for v in (model.spatial_compensation if student is None else student).parameters()))

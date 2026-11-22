"""Replica16组冻结Teacher/对应学生；只比较Compensator替换。"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'cpu_student_inference_20261110_1448'
KERNEL=HERE.parent/'cpu_native_midratio_20261110_1407'
ORIGINAL=HERE.parent/'cpu_native_aggressive_20261110_1115'
FROZEN=HERE.parent/'compensator_distillation_replica_all_20261108'
sys.path[:0]=[str(ORIGINAL),str(BASE),str(KERNEL)]
from cpu_study import *
from student_runtime import CpuRenderer as Baseline
from mid_runtime import extension as baseline_extension
from student_fast_runtime import CpuRenderer
from kernel_runtime import extension
from compensation_student import teacher_fingerprint,load_student,decode_student
HERE=Path(__file__).resolve().parent
SCENES=[s for s in SCENES if s['dataset_kind']=='Replica']
CONDITIONS=['local0.1','local0.2','local0.3','local0.4','local0.5','local0.15','local0.25','local0.45','local0.01','local0.7','fixed0','fixed1000','fixed5000','global0.1','full']
PRIMARY=['local0.1','local0.2','local0.3','local0.4','local0.5']


def student_path(source):return FROZEN/'models'/source['spec']['slug']/'continuous_calibration/best.pt'


def native(source,variant,diagnose=False,**kwargs):
    if variant=='baseline':return Baseline(source['model'],source['pc'],source['reference'].noise,6,student_checkpoint=student_path(source),diagnose=diagnose,**kwargs)
    modes=dict(control=(0,0,0),heads=(1,0,0),compact=(2,0,0),vector=(2,1,0),context=(2,1,1))
    kernel,decode,context=modes[variant]
    return CpuRenderer(source['model'],source['pc'],source['reference'].noise,6,student_checkpoint=student_path(source),diagnose=diagnose,student_kernel=kernel,student_decode=decode,student_context=context,**kwargs)


def request(engine,cam,condition,**kwargs):
    if condition=='full':
        flat,ids,_,_=engine.native.prepare(cam.world_view_transform.contiguous(),cam.full_proj_transform.contiguous(),cam.camera_center.contiguous(),cam.image_width,cam.image_height,cam.znear,3,1.,0,False,False,False,None,None)
        return engine.packet(flat,len(ids)),ids,dict(candidate_gaussians=len(ids))
    kw=dict(view_conditioned=not condition.startswith('global'))
    if condition.startswith('local'):kw['ratio_budget']=float(condition[5:])
    elif condition.startswith('global'):kw['global_ratio']=float(condition[6:])
    else:kw['budget']=int(condition[5:])
    return engine.prepare(cam,**kw,**kwargs)


@torch.no_grad()
def ordinary_student(source,student,cam,ids,ratio,local=True):
    """相同CPU身份上的普通PyTorch学生，用于隔离部署数值误差。"""
    model,pc=source['model'],source['pc'];a=model.anchors;s=model.spatial_compensation;owners=a.rows[ids];reference=a.offset_scales[owners]
    offset=((torch.asinh((pc.get_xyz[ids]-a.centers[owners])/reference)-s.mean[3:6])/s.stddev[3:6]).clamp(-15,15)
    features=a.compensation_lookup(ids,ratio);lo,hi=s.ratio_encoding_values
    t=2*(math.log(ratio)-math.log(lo))/(math.log(hi)-math.log(lo))-1
    code=offset.new_tensor([[t,math.sin(math.pi*t/2),math.cos(math.pi*t/2)]])
    condition=None
    if local and int(getattr(model,'joint_view_conditioning_schema',0)):
        view=model.joint_branch_view_code(model.joint_view_code(cam),compensation=True)
        condition=torch.cat((view[owners],view.new_ones((len(ids),1))),1)
    residual=student(torch.cat((offset,features),1),code,reference,condition)
    master=[pc.get_xyz[ids],pc._opacity[ids],pc.get_scaling[ids],pc.get_rotation[ids],pc.get_features_dc[ids],pc.get_features_rest[ids]]
    return dict(sh_degree=pc.active_sh_degree,inputs=decode_student(residual,master,hasattr(s,'softplus_scale_schema'))),residual

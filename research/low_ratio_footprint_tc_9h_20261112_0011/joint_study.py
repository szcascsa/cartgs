"""冻结Teacher/Student独立四组比较：不把蒸馏质量变化计入执行后端。"""
from nine_study import *
from native_camera import NativeCameraPreparer
from compensation_student import load_student


def student_for_spec(source,spec):
    if spec['dataset_kind']=='Replica':return p.student_for(source,spec['slug'])
    if not spec.get('student'):return None,None,None
    path=Path(spec['student']);student,payload=load_student(path,'cuda',source.fingerprint)
    return student,path,payload


def network_variants(source,student):
    # 既有Tensor Compensator仅支持3534参数的32/32学生；Teacher保留其四个64D Head。
    # Teacher的TC组只加速Selector，Compensator明确保留原FP32，不冒充全TC。
    compensator=2 if student is not None else 0
    projection=1 if student is not None else 0
    original=p.make_engine(source,student,'A')
    network=p.make_engine(source,student,'C',compensator=compensator)
    best=p.make_engine(source,student,'D',frustum=1,projection=projection,device=1,compensator=compensator)
    # 本轮所有执行后端消费同一C++相机，排除旧R→H相机收益和相机舍入差。
    h=NativeCameraPreparer(p.MiniCam)
    return dict(fp32=(original,h),tensor_network=(network,h),tensor_best=(best,h))


def benchmark_variants(networks,renderers):
    return {
        'A_fp32_F2':(networks['fp32'],renderers['F2']),
        'B_fp32_new':(networks['fp32'],renderers['new']),
        'C_tensor_best_F2':(networks['tensor_best'],renderers['F2']),
        'D_tensor_best_new':(networks['tensor_best'],renderers['new']),
        'network_only_F2':(networks['tensor_network'],renderers['F2']),
    }

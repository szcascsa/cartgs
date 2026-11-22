"""开发结果后冻结联合候选；保留集不调参，最终按双轮证据决定是否保留。"""
from pathlib import Path
import hashlib
import json
import time

HERE=Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    state=json.loads((HERE/'reports/NATIVE_DEVELOPMENT_CONTROLLER.json').read_text())
    assert state['state']=='complete'
    for name in ['pending_smoke01','pending_native_smoke01']:
        assert json.loads((HERE/'reports/jobs'/(name+'.json')).read_text())['returncode']==0
    path=HERE/'reports/FROZEN_JOINT.json';assert not path.exists()
    sources={}
    for folder in ['joint_source','joint_source_v2']:
        for file in (HERE/folder).rglob('*'):
            if file.is_file():sources[str(file.relative_to(HERE))]=sha(file)
    for name in ['pending_runtime.py','joint_benchmark.py','joint_quality.py','joint_study.py']:
        sources[name]=sha(HERE/name)
    binaries={str(file):sha(file) for folder in ['joint_build','joint_build_v2'] for file in (HERE/folder).glob('*.so')}
    path.write_text(json.dumps(dict(state='frozen_pending_heldout_validation',time=time.time(),sources=sources,binaries=binaries,
        versions=['A_fp32_F2','B_fp32_new','C_tensor_best_F2','D_tensor_best_new','network_only_F2','E_pending','F_native'],
        selection_rule='E/F仅在主要Ratio双轮场景等权几何平均均获益、且配对场景bootstrap的95%下界大于1时才保留；否则保留D。所有不利场景保留。',
        same_weights=True,teacher_compensator='FP32原四Head',student_compensator='匹配冻结地图的32/32 TC Student',
        tune=['Replica__office2__rgbd','Replica__room0__rgbd','TUM_RGBD__freiburg1_xyz__rgbd','VECtor__corner-slow__stereo'],
        no_scene_specific_parameters=True),indent=2,ensure_ascii=False))
    jobs=[]
    for group in range(16):jobs.append(dict(name=f'joint_quality_g{group}',command=['{python}','-u','-B','joint_quality.py',
        '--output','{job}','--group',str(group),'--pending','--native']))
    for iteration in [1,2]:
        for group in (range(16) if iteration==1 else reversed(range(16))):
            jobs.append(dict(name=f'joint_formal_r{iteration}_g{group}',measure=True,command=['{python}','-u','-B','joint_benchmark.py',
                '--output','{job}','--round',str(iteration),'--group',str(group),'--pending','--native']))
    (HERE/'reports/JOINT_FORMAL_PLAN.json').write_text(json.dumps(dict(name='JOINT_FORMAL',jobs=jobs),indent=2))
    print('JOINT_FROZEN',len(sources),len(binaries),len(jobs))


if __name__=='__main__':main()

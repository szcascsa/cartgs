"""组合两个独立逐位候选：16 lane View投影＋本帧相机/请求直接提交。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_joint_incremental'
    shutil.copytree(HERE / 'source_direct_pose', root)
    (HERE / 'build_joint_incremental').mkdir()
    for name in ['student_grouped_view.cuh', 'deployment_runtime_kernels.cu']:
        shutil.copyfile(HERE / 'source_grouped_view' / name, root / name)
    path = root / 'deployment_prepare.cpp'
    text = path.read_text()
    assert text.count('compensator_mode<=9') == 1
    path.write_text(text.replace('compensator_mode<=9', 'compensator_mode<=12'))
    for name in ['deployment_prepare.cpp', 'raster/graph_main.cu']:
        path = root / name
        path.write_text(path.read_text().replace('mycartgs.student-direct-pose.shared.v1',
            'mycartgs.student-joint-incremental.shared.v1').replace('StudentDirectPoseRasterizer', 'StudentJointIncrementalRasterizer'))
    print('JOINT_INCREMENTAL_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()

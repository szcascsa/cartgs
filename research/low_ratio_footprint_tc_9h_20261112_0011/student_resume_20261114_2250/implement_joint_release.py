"""冻结联合候选：当前相机提交、16lane View、原Tensor Student和独立Alpha弱尾。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_joint_release'
    shutil.copytree(HERE / 'source_full_pose', root)
    (HERE / 'build_joint_release').mkdir()
    for name in ['cutoff_pixels.cuh', 'cutoff_diagnosed.cuh']:
        text = (HERE / 'source_tail_strong' / name).read_text()
        text = text.replace('Cut==1?3.f:Cut==2?4.f:Cut==3?6.f:8.f', 'Cut==0?2.f:Cut==1?3.f:4.f')
        (root / 'raster' / name).write_text(text)
    path = root / 'raster/preprocess_tasks.cuh'
    text = path.read_text().replace('template<int ThreadsPerTask>', 'template<int ThreadsPerTask,int Cut=-1>')
    anchor = 'logf(co.w/(1.0f/255.0f))'
    assert text.count(anchor) == 1
    text = text.replace(anchor, 'logf(co.w/((Cut<0?1.f:Cut==0?2.f:Cut==1?3.f:4.f)/255.f))')
    path.write_text(text)
    path = root / 'raster/graph_main.cu'
    text = path.read_text().replace('#include "encoded_sign_diagnosed.cuh"',
        '#include "encoded_sign_diagnosed.cuh"\n#include "cutoff_pixels.cuh"\n#include "cutoff_diagnosed.cuh"')
    text = text.replace('build_mode<=15', 'build_mode<=22').replace('pixel_mode<=200', 'pixel_mode<=320').replace('pixel_mode_<=200', 'pixel_mode_<=320')
    text = text.replace('build_mode_==14||build_mode_==15', 'build_mode_==14||build_mode_==15||build_mode_>=20')
    text = text.replace('build_mode_==13||build_mode_==15', 'build_mode_==13||build_mode_==15||build_mode_>=20')
    anchor = 'else if(build_mode_==10||build_mode_>=12)executePreprocessedTasks<32>'
    assert text.count(anchor) == 1
    prefix = []
    for build, cut in [(20, 0), (21, 1), (22, 2)]:
        prefix.append(f'      else if(build_mode_=={build})executePreprocessedTasks<32,{cut}><<<workers,128,0,capture>>>(tasks,task_count,inverse,xy,conic,radii,bits,words,grid);')
    text = text.replace('      ' + anchor, '\n'.join(prefix) + '\n      ' + anchor)
    anchor = next(line for line in text.splitlines() if 'if(pixel_mode_==200)renderEncodedSignPixels' in line)
    calls, diagnostics = [], []
    for mode, cut in [(203, 0), (301, 1), (302, 2)]:
        calls.append(f'      if(pixel_mode_=={mode+18})renderCutoffPixels<38,3,1,{cut}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);')
        diagnostics.append(f'    if(pixel_mode_=={mode+18})renderCutoffDiagnosedPixels<38,3,1,{cut}><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));')
    text = text.replace(anchor, anchor + '\n' + '\n'.join(calls))
    anchor = '    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '\n'.join(diagnostics) + '\n' + anchor)
    path.write_text(text)
    for name in ['deployment_prepare.cpp', 'raster/graph_main.cu']:
        path = root / name
        path.write_text(path.read_text().replace('mycartgs.student-full-pose.shared.v1',
            'mycartgs.student-joint-release.shared.v1').replace('StudentFullPoseRasterizer', 'StudentJointReleaseRasterizer'))
    print('JOINT_RELEASE_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()

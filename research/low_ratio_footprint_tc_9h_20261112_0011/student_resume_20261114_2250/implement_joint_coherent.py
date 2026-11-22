"""将已冻结155渲染器接入Student原生调度；网络、准备和共享句柄独立保存。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_joint_coherent';shutil.copytree(HERE/'source_members',root);(HERE/'build_joint_coherent').mkdir()
    raster=root/'raster'
    for name in ['branchless_pixels.cuh','branchless_diagnosed.cuh','coherent_pixels.cuh','coherent_diagnosed.cuh']:
        shutil.copyfile(HERE/'source_coherent'/name,raster/name)
    for name in ['sign_pixels.cuh','sign_diagnosed.cuh']:
        shutil.copyfile(HERE/'source_sign'/name,raster/name)
    path=raster/'graph_main.cu';text=path.read_text().replace('#include "diagnosed_pixels.cuh"',
        '#include "diagnosed_pixels.cuh"\n#include "branchless_pixels.cuh"\n#include "branchless_diagnosed.cuh"\n#include "coherent_pixels.cuh"\n#include "coherent_diagnosed.cuh"\n#include "sign_pixels.cuh"\n#include "sign_diagnosed.cuh"')
    text=text.replace('pixel_mode<=58','pixel_mode<=184').replace('pixel_mode_<=58','pixel_mode_<=184')
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==58)renderFoldedPixels<40>' in line)
    launch='      if(pixel_mode_==173)renderCoherentPixels<38,5,1><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);'
    launch+='\n      if(pixel_mode_==184)renderSignPixels<38,3,1><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);'
    text=text.replace(anchor,anchor+'\n'+launch)
    anchor='    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    launch='    if(pixel_mode_==173)renderCoherentDiagnosedPixels<38,5,1><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));'
    launch+='\n    if(pixel_mode_==184)renderSignDiagnosedPixels<38,3,1><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));'
    text=text.replace(anchor,launch+'\n'+anchor)
    text=text.replace('StudentMembersRasterizer','StudentJointCoherentRasterizer').replace('mycartgs.student-members.shared.v1','mycartgs.student-joint-coherent.shared.v1')
    path.write_text(text)
    path=root/'deployment_prepare.cpp';text=path.read_text().replace('mycartgs.student-members.shared.v1','mycartgs.student-joint-coherent.shared.v1');path.write_text(text)
    print('JOINT_COHERENT_SOURCE_READY')


if __name__=='__main__':main()

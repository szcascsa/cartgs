"""冻结182像素核接入已验证Student调度；其余源文件保持逐字相同。"""
from pathlib import Path
import shutil
import json

HERE=Path(__file__).resolve().parent


def main():
    assert json.loads((HERE/'reports/FROZEN_RENDER_FINAL.json').read_text())['mode']==182
    root=HERE/'source_joint_final';shutil.copytree(HERE/'source_members',root);(HERE/'build_joint_final').mkdir()
    raster=root/'raster'
    headers=['branchless_pixels.cuh','branchless_diagnosed.cuh','coherent_pixels.cuh','coherent_diagnosed.cuh',
        'encoded_pixels.cuh','encoded_diagnosed.cuh','encoded_sign_pixels.cuh','encoded_sign_diagnosed.cuh']
    for name in headers:shutil.copyfile(HERE/'source_encoded_sign'/name,raster/name)
    path=raster/'graph_main.cu';text=path.read_text().replace('#include "diagnosed_pixels.cuh"',
        '#include "diagnosed_pixels.cuh"\n'+'\n'.join('#include "'+name+'"' for name in headers))
    text=text.replace('pixel_mode<=58','pixel_mode<=200').replace('pixel_mode_<=58','pixel_mode_<=200')
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==58)renderFoldedPixels<40>' in line)
    launch='      if(pixel_mode_==200)renderEncodedSignPixels<38,3,1><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);'
    text=text.replace(anchor,anchor+'\n'+launch)
    anchor='    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    launch='    if(pixel_mode_==200)renderEncodedSignDiagnosedPixels<38,3,1><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));'
    text=text.replace(anchor,launch+'\n'+anchor).replace('StudentMembersRasterizer','StudentJointFinalRasterizer')
    text=text.replace('mycartgs.student-members.shared.v1','mycartgs.student-joint-final.shared.v1');path.write_text(text)
    path=root/'deployment_prepare.cpp';text=path.read_text().replace('mycartgs.student-members.shared.v1','mycartgs.student-joint-final.shared.v1');path.write_text(text)
    print('JOINT_FINAL_SOURCE_READY')


if __name__=='__main__':main()

"""保持逐像素原合成表达式，只改变热循环展开程度，复查指令体积与完整延迟。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_unroll';shutil.copytree(HERE.parent/'source',root);(HERE/'build_unroll').mkdir()
    kernel=(root/'folded_pixels.cuh').read_text()
    kernel=kernel[kernel.index('template<int Mode> __global__ void renderFoldedPixels'):]
    kernel=kernel.replace('template<int Mode> __global__', 'template<int Mode,int Unroll> __global__')
    kernel=kernel.replace('renderFoldedPixels','renderUnrolledPixels').replace('#pragma unroll 32','#pragma unroll Unroll')
    (root/'unrolled_pixels.cuh').write_text(kernel)
    path=root/'graph_main.cu';text=path.read_text()
    text=text.replace('#include "diagnosed_pixels.cuh"','#include "diagnosed_pixels.cuh"\n#include "unrolled_pixels.cuh"')
    text=text.replace('pixel_mode<=58','pixel_mode<=128')
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==58)renderFoldedPixels<40>' in line)
    lines=[];diagnosed=[]
    for mode,(unroll,kind) in enumerate([(u,m) for u in [1,2,4,8,16] for m in [37,38]],101):
        lines.append(f'      if(pixel_mode_=={mode+18})renderUnrolledPixels<{kind},{unroll}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);')
        diagnosed.append(f'    if(pixel_mode_=={mode+18})renderDiagnosedPixels<{kind}><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));')
    text=text.replace(anchor,anchor+'\n'+'\n'.join(lines)).replace('pixel_mode_>=55&&pixel_mode_<=58','pixel_mode_>=55&&pixel_mode_<=128')
    anchor='    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    text=text.replace(anchor,'\n'.join(diagnosed)+'\n'+anchor).replace('NineHourRasterizer','StudentUnrolledRasterizer')
    path.write_text(text)
    print('STUDENT_UNROLLED_SOURCE_READY')


if __name__=='__main__':main()

"""按连续深度范围的结合律并行合成；独立近似路径，保持输入、排序和严格K。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_prefix';shutil.copytree(HERE.parent/'source',root);(HERE/'build_prefix').mkdir()
    shutil.copy2(HERE/'prefix_pixels.cuh',root/'prefix_pixels.cuh')
    path=root/'graph_main.cu';text=path.read_text().replace('#include "diagnosed_pixels.cuh"','#include "diagnosed_pixels.cuh"\n#include "prefix_pixels.cuh"')
    text=text.replace('pixel_mode<=58','pixel_mode<=141')
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==58)renderFoldedPixels<40>' in line)
    lines=[];diagnosed=[]
    for mode,parts in [(121,1),(122,2),(123,4)]:
        lines.append(f'      if(pixel_mode_=={mode+18})renderPrefixPixels<{parts}><<<grid,{512//parts},0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,inputs);')
        diagnosed.append(f'    if(pixel_mode_=={mode+18})renderPrefixPixels<{parts},true><<<grid,{512//parts},0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,ko,inputs,reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()),image.data_ptr<float>());')
    text=text.replace(anchor,anchor+'\n'+'\n'.join(lines)).replace('pixel_mode_>=55&&pixel_mode_<=58','pixel_mode_>=55&&pixel_mode_<=141')
    anchor='    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    text=text.replace(anchor,'\n'.join(diagnosed)+'\n'+anchor).replace('NineHourRasterizer','StudentPrefixRasterizer')
    path.write_text(text)
    print('STUDENT_PREFIX_SOURCE_READY')


if __name__=='__main__':main()

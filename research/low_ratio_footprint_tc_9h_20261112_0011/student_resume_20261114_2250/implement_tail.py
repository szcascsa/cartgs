"""仅新建渲染器源码和二进制，原F2与先前近似渲染器保持冻结。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE/'source_tail'
    shutil.copytree(HERE.parent/'source', root)
    (HERE/'build_tail').mkdir()
    shutil.copy2(HERE/'tail_pixels.cuh', root/'tail_pixels.cuh')
    path = root/'graph_main.cu'
    text = path.read_text()
    text = text.replace('#include "diagnosed_pixels.cuh"', '#include "diagnosed_pixels.cuh"\n#include "tail_pixels.cuh"')
    text = text.replace('pixel_mode<=58', 'pixel_mode<=63')
    anchor = next(line for line in text.splitlines() if 'if(pixel_mode_==58)renderFoldedPixels<40>' in line)
    lines = []
    for index, stop in enumerate([2,5,10,20,35], 59):
        lines.append(f'      if(pixel_mode_=={index})renderTailPixels<{stop}><<<grid,128,0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,inputs);')
    text = text.replace(anchor, anchor+'\n'+'\n'.join(lines))
    text = text.replace('pixel_mode_>=55&&pixel_mode_<=58', 'pixel_mode_>=55&&pixel_mode_<=63')
    anchor = '    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    assert text.count(anchor) == 1
    lines = ['    if(pixel_mode_>=59){', '      auto tail=torch::zeros({2},parameters.options().dtype(at::kLong));']
    for index, stop in enumerate([2,5,10,20,35], 59):
        lines.append(f'      if(pixel_mode_=={index})renderTailPixels<{stop},true><<<grid,128,0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,ko,inputs,image.defined()?reinterpret_cast<unsigned*>(flags.data_ptr<int>()):nullptr,reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()),reinterpret_cast<unsigned long long*>(tail.data_ptr<int64_t>()),image.data_ptr<float>());')
    lines += ['      C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions,tail};', '    }']
    text = text.replace(anchor, '\n'.join(lines)+'\n'+anchor)
    text = text.replace('NineHourRasterizer', 'StudentTailRasterizer')
    path.write_text(text)
    print('STUDENT_TAIL_SOURCE_READY', root)


if __name__ == '__main__':
    main()

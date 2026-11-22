"""有界全局Tail Workspace；单线程/Tile替代原128线程suffix scan，并融合终止颜色发布。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
MODES={131:2,132:5,133:10,134:20,135:35}


def main():
    root=HERE/'source_serial_tail';shutil.copytree(HERE.parent/'source',root);(HERE/'build_serial_tail').mkdir()
    header=(HERE/'serial_tail_pixels.cuh').read_text()
    text=(HERE/'tail_pixels.cuh').read_text();text=text[text.index('template<int StopPercent'):]
    text=text.replace('renderTailPixels','renderSerialTailPixels')
    text=text.replace('const uint32_t* depth_keys,const GraphInputs* inputs,','const uint32_t* depth_keys,const GraphInputs* inputs,const float* cached_tail,')
    text=text.replace('rgb[256],tail[257]','rgb[256]')
    text=text.replace('    using Scan=cub::BlockScan<float4,128>;__shared__ typename Scan::TempStorage temp;\n','')
    start=text.index('      float4 suffix[2];');end=text.index('      #pragma unroll 8',start)
    text=text[:start]+text[end:]
    start=text.index('        const int id=lists[tile*capacity+j];',text.index('      #pragma unroll 8'))
    end=text.index('\n      }\n    }\n  }else{',start)
    text=text[:start]+'''        const int id=lists[tile*capacity+j];const float4 a=abc[j];const float2 b=ef[j];
        const float power=fmaf(a.x,lx*lx,fmaf(a.z,ly*ly,fmaf(a.y,lx*ly,fmaf(a.w,lx,fmaf(b.x,ly,b.y)))));
        const float next=power+8.f*(a.y*lx+a.z*(2.f*ly+8.f)+b.x);
        const float* tail=cached_tail+(int64_t(tile)*(capacity+1)+j+1)*3;
        blendSerialTail<StopPercent,Diagnose>(power,rgb[j],done[0],T[0],C[0],tail,id,flags,&visits,&tail_pixels,&tail_members,count-j-1);
        blendSerialTail<StopPercent,Diagnose>(next,rgb[j],done[1],T[1],C[1],tail,id,flags,&visits,&tail_pixels,&tail_members,count-j-1);
''' +text[end:]
    (root/'serial_tail_pixels.cuh').write_text(header+'\n'+text)
    path=root/'graph_main.cu';text=path.read_text()
    text=text.replace('#include "diagnosed_pixels.cuh"','#include "diagnosed_pixels.cuh"\n#include "serial_tail_pixels.cuh"')
    text=text.replace('Tensor geometry,sort,bits,directory,lists,counts,tasks;', 'Tensor geometry,sort,bits,directory,lists,counts,tasks,tail;')
    text=text.replace('{geometry,sort,bits,directory,lists,counts,tasks}', '{geometry,sort,bits,directory,lists,counts,tasks,tail}')
    text=text.replace('Tensor pool_geometry,pool_sort,pool_bits,pool_directory,pool_lists,pool_counts,pool_tasks;',
        'Tensor pool_geometry,pool_sort,pool_bits,pool_directory,pool_lists,pool_counts,pool_tasks,pool_tail;')
    text=text.replace('{pool_geometry,pool_sort,pool_bits,pool_directory,pool_lists,pool_counts,pool_tasks}',
        '{pool_geometry,pool_sort,pool_bits,pool_directory,pool_lists,pool_counts,pool_tasks,pool_tail}')
    # 未定义的tail Tensor不能查询storage。
    text=text.replace('count+=t.storage().nbytes();return count;', 'if(t.defined())count+=t.storage().nbytes();return count;')
    text=text.replace('pixel_mode<=58','pixel_mode<=153')
    anchor='    plan->lists=pool_lists;plan->counts=pool_counts;plan->tasks=pool_tasks;'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'''
    if(pixel_mode_>=149){
      TORCH_CHECK(int64_t(tiles)*(list_capacity+1)*3*sizeof(float)<=128ll<<20,"Tail workspace exceeds 128 MiB");
      if(!pool_tail.defined()||pool_tail.numel()!=int64_t(tiles)*(list_capacity+1)*3)
        pool_tail=torch::empty({int64_t(tiles)*(list_capacity+1)*3},options.dtype(at::kFloat));
      plan->tail=pool_tail;
    }
''')
    anchor='    if(pixel_mode_>=6) {'
    assert text.count(anchor)==1
    text=text.replace(anchor,'''    if(pixel_mode_>=149)prepareSerialTail<<<(tiles+127)/128,128,0,capture>>>(W,H,lists,counts,list_capacity,xy,conic,rgb,inputs,plan->tail.data_ptr<float>());
''' +anchor)
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==58)renderFoldedPixels<40>' in line)
    lines=[];diagnosed=[]
    for mode,stop in MODES.items():
        lines.append(f'      if(pixel_mode_=={mode+18})renderSerialTailPixels<{stop}><<<grid,128,0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,inputs,plan->tail.data_ptr<float>());')
        diagnosed.append(f'      if(pixel_mode_=={mode+18})renderSerialTailPixels<{stop},true><<<grid,128,0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,ko,inputs,p.tail.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()),reinterpret_cast<unsigned long long*>(tail.data_ptr<int64_t>()),image.data_ptr<float>());')
    text=text.replace(anchor,anchor+'\n'+'\n'.join(lines)).replace('pixel_mode_>=55&&pixel_mode_<=58','pixel_mode_>=55&&pixel_mode_<=153')
    anchor='    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    text=text.replace(anchor,'    if(pixel_mode_>=149){\n      auto tail=torch::zeros({2},parameters.options().dtype(at::kLong));\n'+'\n'.join(diagnosed)+'\n      C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions,tail};\n    }\n'+anchor)
    text=text.replace('NineHourRasterizer','StudentSerialTailRasterizer');path.write_text(text)
    print('STUDENT_SERIAL_TAIL_SOURCE_READY')


if __name__=='__main__':main()

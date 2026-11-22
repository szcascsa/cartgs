"""低成本尾部近似：前缀归一化或相邻颜色补全，避免整Tile suffix scan固定成本。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
MODES={46:(1,2),47:(1,5),48:(1,10),49:(1,20),50:(2,5),51:(2,10),52:(2,20),53:(3,10),54:(4,10),55:(3,5),56:(1,35),57:(2,35),58:(4,20)}


def main():
    root=HERE/'source_normal'
    shutil.copytree(HERE/'source_tail',root)
    (HERE/'build_normal').mkdir()
    text=(HERE/'tail_pixels.cuh').read_text()
    text=text[text.index('template<int StopPercent'):]
    text=text.replace('template<int StopPercent,bool Diagnose=false>','template<int Style,int StopPercent,bool Diagnose=false>')
    text=text.replace('renderTailPixels','renderNormalTailPixels')
    text=text.replace('rgb[256],tail[257]','rgb[256]')
    text=text.replace('    using Scan=cub::BlockScan<float4,128>;__shared__ typename Scan::TempStorage temp;\n','')
    start=text.index('      float4 suffix[2];')
    end=text.index('      #pragma unroll 8',start)
    text=text[:start]+text[end:]
    start=text.index('        for(int p=0;p<2;++p)if(!done[p]')
    end=text.index('          T[p]=0.f;done[p]=true;',start)
    text=text[:start]+'''        for(int p=0;p<2;++p)if(!done[p]&&j+1<count&&T[p]<StopPercent*.01f){
          if constexpr(Style==1){
            const float inverse=1.f/(1.f-T[p]);
            C[p][0]*=inverse;C[p][1]*=inverse;C[p][2]*=inverse;
          }else{
            float3 estimate=Style==3?rgb[j+1]:rgb[j];
            if constexpr(Style==4){
              const float inverse=.5f/(1.f-T[p]);
              estimate={fmaf(C[p][0],inverse,.5f*estimate.x),fmaf(C[p][1],inverse,.5f*estimate.y),fmaf(C[p][2],inverse,.5f*estimate.z)};
            }
            C[p][0]=fmaf(T[p],estimate.x,C[p][0]);C[p][1]=fmaf(T[p],estimate.y,C[p][1]);C[p][2]=fmaf(T[p],estimate.z,C[p][2]);
          }
''' +text[end:]
    (root/'normal_pixels.cuh').write_text(text)
    path=root/'graph_main.cu';text=path.read_text()
    text=text.replace('#include "tail_pixels.cuh"','#include "tail_pixels.cuh"\n#include "normal_pixels.cuh"')
    text=text.replace('pixel_mode<=63','pixel_mode<=76')
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==63)renderTailPixels<35>' in line)
    lines=[]
    for mode,(style,stop) in MODES.items():
        lines.append(f'      if(pixel_mode_=={mode+18})renderNormalTailPixels<{style},{stop}><<<grid,128,0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,inputs);')
    text=text.replace(anchor,anchor+'\n'+'\n'.join(lines))
    text=text.replace('pixel_mode_>=55&&pixel_mode_<=63','pixel_mode_>=55&&pixel_mode_<=76')
    anchor='    if(pixel_mode_>=59){'
    assert text.count(anchor)==1
    lines=['    if(pixel_mode_>=64){','      auto tail=torch::zeros({2},parameters.options().dtype(at::kLong));']
    for mode,(style,stop) in MODES.items():
        lines.append(f'      if(pixel_mode_=={mode+18})renderNormalTailPixels<{style},{stop},true><<<grid,128,0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,ko,inputs,reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()),reinterpret_cast<unsigned long long*>(tail.data_ptr<int64_t>()),image.data_ptr<float>());')
    lines+=['      C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions,tail};','    }']
    text=text.replace(anchor,'\n'.join(lines)+'\n'+anchor)
    text=text.replace('StudentTailRasterizer','StudentNormalRasterizer')
    path.write_text(text)
    print('STUDENT_NORMAL_TAIL_SOURCE_READY',len(MODES))


if __name__=='__main__':main()

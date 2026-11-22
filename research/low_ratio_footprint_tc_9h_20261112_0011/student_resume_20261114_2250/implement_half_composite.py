"""FP32投影/power/Alpha/T，颜色累加使用half2将两像素打包。"""
from pathlib import Path
import shutil
import re

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_half';shutil.copytree(HERE.parent/'source',root);(HERE/'build_half').mkdir()
    text=(root/'folded_pixels.cuh').read_text()
    blend=text[text.index('template<int Mode> __device__ __forceinline__ void blendFolded'):text.index('template<int Mode> __device__ __forceinline__ void foldedPair')]
    blend=blend.replace('void blendFolded(', 'float foldedHalfWeight(')
    blend=blend.replace('float power,float3 color,bool& done,float& T,float* C)', 'float power,bool& done,float& T)')
    blend=blend.replace('return;', 'return 0.f;')
    start=blend.index('  if(Mode==30')
    end=blend.index('  constexpr float Stop',start)
    blend=blend[:start]+'  const float weight=alpha*T;T=next;\n'+blend[end:]
    blend=blend.replace('  if(Stop>0.f&&T<Stop)done=true;','  if(Stop>0.f&&T<Stop)done=true;\n  return weight;')
    kernel=text[text.index('template<int Mode> __global__ void renderFoldedPixels'):]
    kernel=kernel.replace('template<int Mode> __global__','template<int Mode,bool Diagnose=false> __global__')
    kernel=kernel.replace('renderFoldedPixels','renderHalfPixels')
    kernel=kernel.replace('const GraphInputs* inputs) {','const GraphInputs* inputs,unsigned* flags=nullptr,unsigned long long* interactions=nullptr,float* diagnostic_image=nullptr) {')
    kernel=kernel.replace('float* out=inputs->image;', 'float* out=Diagnose?diagnostic_image:inputs->image;unsigned long long visits=0;')
    kernel=kernel.replace('  const int count=counts[tile];', '  __half2 sums[3]={__float2half2_rn(0.f),__float2half2_rn(0.f),__float2half2_rn(0.f)};\n  bool half_used=false;\n  const int count=counts[tile];')
    kernel=kernel.replace('__shared__ float3 rgb[256];', '__shared__ __half2 rgb[256][3];')
    kernel=kernel.replace('abc[j]=f.a;ef[j]=f.b;rgb[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};',
        'abc[j]=f.a;ef[j]=f.b;for(int ch=0;ch<3;++ch)rgb[j][ch]=__float2half2_rn(colors[id*3+ch]);')
    kernel=kernel.replace('if(Mode>=28&&Mode<=32&&!valid[j])', 'if(false)')
    kernel=kernel.replace('  }else{\n    #pragma unroll 32', '  }else{\n    half_used=true;\n    #pragma unroll 32')
    start=kernel.index('      if(false)')
    end=kernel.index('\n    }\n  }\n  } else',start)
    kernel=kernel[:start]+'''      const float4 a=abc[j];const float2 b=ef[j];
      const float power=fmaf(a.x,lx*lx,fmaf(a.z,ly*ly,fmaf(a.y,lx*ly,fmaf(a.w,lx,fmaf(b.x,ly,b.y)))));
      const float next=power+8.f*(a.y*lx+a.z*(2.f*ly+8.f)+b.x);
      const float w0=foldedHalfWeight<Mode>(power,done[0],T[0]);
      const float w1=foldedHalfWeight<Mode>(next,done[1],T[1]);
      const __half2 weight=__floats2half2_rn(w0,w1);
      #pragma unroll
      for(int ch=0;ch<3;++ch)sums[ch]=__hfma2(rgb[j][ch],weight,sums[ch]);
      if constexpr(Diagnose){if(w0||w1)atomicOr(flags+lists[tile*capacity+j],1u);visits+=(w0>0.f)+(w1>0.f);}
''' +kernel[end:]
    # 全Tile异常/列表溢出仍走原FP32合成，不消费half缓存。
    kernel=re.sub(r'compositeScalingPair<9>\((.*?)\);',lambda m:
        'if constexpr(Diagnose)debugOriginalPair(xy[id],conic[id],{colors[id*3],colors[id*3+1],colors[id*3+2]},pix,done,T,C,-CUDART_INF_F,id,flags,&visits);else '+m[0],kernel,flags=re.S)
    anchor='  #pragma unroll\n  for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out'
    kernel=kernel.replace(anchor,'''  if(half_used)for(int ch=0;ch<3;++ch){C[0][ch]=__low2float(sums[ch]);C[1][ch]=__high2float(sums[ch]);}
  if constexpr(Diagnose)if(visits)atomicAdd(interactions,visits);
'''+anchor)
    (root/'half_pixels.cuh').write_text('#include <cuda_fp16.h>\n'+blend+'\n'+kernel)
    path=root/'graph_main.cu';text=path.read_text().replace('#include "diagnosed_pixels.cuh"','#include "diagnosed_pixels.cuh"\n#include "half_pixels.cuh"')
    text=text.replace('pixel_mode<=58','pixel_mode<=100')
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==58)renderFoldedPixels<40>' in line)
    lines=[];diagnosed=[]
    for mode,kind in [(81,37),(82,38)]:
        lines.append(f'      if(pixel_mode_=={mode+18})renderHalfPixels<{kind}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);')
        diagnosed.append(f'    if(pixel_mode_=={mode+18})renderHalfPixels<{kind},true><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()),image.data_ptr<float>());')
    text=text.replace(anchor,anchor+'\n'+'\n'.join(lines)).replace('pixel_mode_>=55&&pixel_mode_<=58','pixel_mode_>=55&&pixel_mode_<=100')
    anchor='    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    text=text.replace(anchor,'\n'.join(diagnosed)+'\n'+anchor).replace('NineHourRasterizer','StudentHalfRasterizer')
    path.write_text(text)
    print('STUDENT_HALF_SOURCE_READY')


if __name__=='__main__':main()

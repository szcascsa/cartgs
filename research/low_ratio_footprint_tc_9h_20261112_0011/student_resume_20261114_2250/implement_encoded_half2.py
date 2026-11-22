"""近似候选：由bit-exp直接打包FP16 Alpha，保留182的提升多项式和相邻像素布局。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_encoded_half2'
    shutil.copytree(HERE / 'source_encoded_sign', root)
    (HERE / 'build_encoded_half2').mkdir()
    helper = '''#include <cuda_fp16.h>
__device__ __forceinline__ unsigned half2Bits(__half2 x){const __half2_raw r=x;return unsigned(r.x)|(unsigned(r.y)<<16);}
__device__ __forceinline__ __half2 half2Value(unsigned x){__half2_raw r;r.x=x&65535;r.y=x>>16;return __half2(r);}
__device__ __forceinline__ unsigned encodedHalf(float x,bool done){
  const int bits=min(__float2int_rz(x),__float_as_int(.99f));
  return done||bits<__float_as_int(1.f/255.f)?0u:unsigned(((bits+4096)>>13)-0x1c000);
}
template<bool Diagnose> __device__ __forceinline__ void encodedHalfPair(float4 a,float2 b,const __half2* color,
    float x,float y,__half2& T,__half2* C,unsigned id,unsigned* flags,unsigned long long* visits){
  const float power=fmaf(a.x,x*x,fmaf(a.z,y*y,fmaf(a.y,x*y,fmaf(a.w,x,fmaf(b.x,y,b.y)))));
  const float next=power+(a.y*x+a.z*(2.f*y+1.f)+b.x);
  const unsigned old=half2Bits(T),alpha0=encodedHalf(power,old&0x8000),alpha1=encodedHalf(next,old&0x80000000u);
  const __half2 alpha=half2Value(alpha0|(alpha1<<16)),weight=__hmul2(T,alpha);
  __half2 updated=__hsub2(T,weight);
  #pragma unroll
  for(int ch=0;ch<3;++ch)C[ch]=__hfma2(color[ch],weight,C[ch]);
  const unsigned stop=(half2Bits(__hlt2(updated,__float2half2_rn(.01f)))<<2)&0x80008000u;
  T=half2Value(half2Bits(updated)|stop);
  if constexpr(Diagnose){if(alpha0||alpha1)atomicOr(flags+id,1u);*visits+=bool(alpha0)+bool(alpha1);}
}
'''
    for name in ['encoded_sign_pixels.cuh', 'encoded_sign_diagnosed.cuh']:
        text = (root / name).read_text()
        text = text[text.index('template<int Mode,int Kind,int Layout> __global__'):]
        text = text.replace('renderEncodedSign', 'renderEncodedHalf2')
        text = text.replace('__shared__ float3 rgb[256];', '__shared__ __half2 half_rgb[256][3];')
        anchor = '        if(!safe)atomicOr(&unsafe_tile,1u);'
        assert text.count(anchor) == 1
        text = text.replace(anchor, '''        for(int ch=0;ch<3;++ch)safe=safe&&isfinite(colors[id*3+ch])&&fabsf(colors[id*3+ch])<=65504.f;
''' + anchor)
        anchor = 'abc[j]=f.a;ef[j]=f.b;rgb[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};'
        assert text.count(anchor) == 1
        text = text.replace(anchor, 'abc[j]=f.a;ef[j]=f.b;for(int ch=0;ch<3;++ch)half_rgb[j][ch]=__float2half2_rn(colors[id*3+ch]);')
        anchor = '  const int count=counts[tile];'
        assert text.count(anchor) == 1
        text = text.replace(anchor, '''  bool half_used=false;
  __half2 packed_T=half2Value((inside[0]?0x3c00u:0xbc00u)|((inside[1]?0x3c00u:0xbc00u)<<16));
  __half2 packed_C[3]={__float2half2_rn(0.f),__float2half2_rn(0.f),__float2half2_rn(0.f)};
''' + anchor)
        start = text.index('    #pragma unroll 32\n    for(int j=0;j<count;++j){')
        end = text.index('\n  }\n  } else', start)
        diagnostic = name.endswith('_diagnosed.cuh')
        call = f'encodedHalfPair<{"true" if diagnostic else "false"}>(abc[j],ef[j],half_rgb[j],lx,ly,packed_T,packed_C,'
        call += 'lists[tile*capacity+j],flags,&visits);' if diagnostic else '0,nullptr,nullptr);'
        text = text[:start] + '''    half_used=true;
    #pragma unroll 32
    for(int j=0;j<count;++j){
      if((j&7)==0&&__all_sync(0xffffffff,(half2Bits(packed_T)&0x80008000u)==0x80008000u))break;
      ''' + call + '\n    }' + text[end:]
        # 已移除的非启用Mode28..32分支之外，不允许留下对旧颜色数组的引用。
        assert 'rgb[j]' not in text.replace('half_rgb[j]', '')
        anchor = '  #pragma unroll\n  for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out'
        assert text.count(anchor) == 1
        text = text.replace(anchor, '''  if(half_used){
    for(int ch=0;ch<3;++ch){C[0][ch]=__low2float(packed_C[ch]);C[1][ch]=__high2float(packed_C[ch]);}
    T[0]=__low2float(packed_T);T[1]=__high2float(packed_T);
  }
''' + anchor)
        (root / name.replace('encoded_sign', 'encoded_half2')).write_text((helper if not diagnostic else '') + text)
    path = root / 'graph_main.cu'
    text = path.read_text().replace('#include "encoded_sign_diagnosed.cuh"',
        '#include "encoded_sign_diagnosed.cuh"\n#include "encoded_half2_pixels.cuh"\n#include "encoded_half2_diagnosed.cuh"')
    text = text.replace('pixel_mode<=202', 'pixel_mode<=299').replace('pixel_mode_<=202', 'pixel_mode_<=299')
    anchor = next(line for line in text.splitlines() if 'if(pixel_mode_==202)renderEncodedSignPixels' in line)
    text = text.replace(anchor, anchor + '\n      if(pixel_mode_==299)renderEncodedHalf2Pixels<38,3,1><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);')
    anchor = '    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '    if(pixel_mode_==299)renderEncodedHalf2DiagnosedPixels<38,3,1><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));\n' + anchor)
    path.write_text(text.replace('StudentEncodedSignRasterizer', 'StudentEncodedHalf2Rasterizer'))
    print('ENCODED_HALF2_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()

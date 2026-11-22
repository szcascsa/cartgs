"""FP32 power，打包FP16 Alpha/T/RGB与位掩码早停，减少两个像素的标量递推。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_packed_state';shutil.copytree(HERE/'source_half',root);(HERE/'build_packed_state').mkdir()
    path=root/'half_pixels.cuh';text=path.read_text()
    start=text.index('template<int Mode> __device__');end=text.index('template<int Mode,bool Diagnose',start)
    helper='''__device__ __forceinline__ unsigned packedHalfBits(__half2 x){const __half2_raw raw=x;return unsigned(raw.x)|(unsigned(raw.y)<<16);}
__device__ __forceinline__ __half2 packedHalfValue(unsigned bits){__half2_raw raw;raw.x=bits&65535u;raw.y=bits>>16;return __half2(raw);}
__device__ __forceinline__ unsigned packedHalfLess(__half2 a,__half2 b){
  unsigned value=packedHalfBits(__hlt2(a,b))<<2,mask;
  asm("prmt.b32 %0,%1,0,0xbb99;":"=r"(mask):"r"(value));return mask;
}
__device__ __forceinline__ float foldedPackedAlpha(float power,bool done){
  if(done||power<-5.541263545158426f)return 0.f;
  const float alpha=fminf(.99f,boundedExp<1>(power));return alpha<1.f/255.f?0.f:alpha;
}

'''
    text=text[:start]+helper+text[end:]
    text=text.replace('renderHalfPixels','renderPackedStatePixels')
    anchor='  bool half_used=false;'
    text=text.replace(anchor,'''  __half2 packed_T=__float2half2_rn(1.f);
  unsigned done_mask=(done[0]?65535u:0u)|(done[1]?0xffff0000u:0u);
'''+anchor)
    start=text.index('    half_used=true;');end=text.index('\n  }\n  } else',start)
    part=text[start:end]
    part=part.replace('done[0]&&done[1]','done_mask==0xffffffffu')
    start2=part.index('      const float w0=');end2=part.index('\n    }',start2)
    part=part[:start2]+'''      const __half2 alpha=__floats2half2_rn(foldedPackedAlpha(power,done_mask&65535u),foldedPackedAlpha(next,done_mask&0xffff0000u));
      __half2 updated=__hmul2(packed_T,__hsub2(__float2half2_rn(1.f),alpha));
      const unsigned crossing=packedHalfLess(updated,__float2half2_rn(.0001f));
      const __half2 active_alpha=packedHalfValue(packedHalfBits(alpha)&~crossing);
      updated=packedHalfValue((packedHalfBits(updated)&~crossing)|(packedHalfBits(packed_T)&crossing));
      const __half2 weight=__hmul2(packed_T,active_alpha);
      #pragma unroll
      for(int ch=0;ch<3;++ch)sums[ch]=__hfma2(rgb[j][ch],weight,sums[ch]);
      packed_T=updated;done_mask|=crossing;
      if(Mode==38)done_mask|=packedHalfLess(packed_T,__float2half2_rn(.01f));
      if constexpr(Diagnose){const unsigned raw=packedHalfBits(active_alpha);if(raw)atomicOr(flags+lists[tile*capacity+j],1u);visits+=bool(raw&65535u)+bool(raw&0xffff0000u);}
''' +part[end2:]
    text=text[:start]+part+text[end:]
    text=text.replace('  if(half_used)for(int ch=0;ch<3;++ch){C[0][ch]=__low2float(sums[ch]);C[1][ch]=__high2float(sums[ch]);}',
        '  if(half_used){for(int ch=0;ch<3;++ch){C[0][ch]=__low2float(sums[ch]);C[1][ch]=__high2float(sums[ch]);}T[0]=__low2float(packed_T);T[1]=__high2float(packed_T);}')
    path.write_text(text)
    path=root/'graph_main.cu';text=path.read_text().replace('renderHalfPixels','renderPackedStatePixels')
    text=text.replace('StudentHalfRasterizer','StudentPackedStateRasterizer');path.write_text(text)
    print('STUDENT_PACKED_STATE_SOURCE_READY')


if __name__=='__main__':main()

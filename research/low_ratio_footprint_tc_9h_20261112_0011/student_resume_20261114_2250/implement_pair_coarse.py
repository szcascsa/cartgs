"""独立近似候选：仅当Tile内纵向log-alpha变化有界时，邻接两像素共享中点Alpha。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_pair_coarse'
    shutil.copytree(HERE / 'source_encoded_sign', root)
    (HERE / 'build_pair_coarse').mkdir()
    helper = '''// 此路径仍逐Gaussian、逐像素更新T/C；只近似两相邻像素的Alpha。
__device__ __forceinline__ void coarseBlend(float alpha,float3 color,float& T,float* C){
  alpha=T<0.f?0.f:alpha;const float weight=alpha*T,next=T-weight;
  C[0]=fmaf(color.x,weight,C[0]);C[1]=fmaf(color.y,weight,C[1]);C[2]=fmaf(color.z,weight,C[2]);
  T=__int_as_float(__float_as_int(next)|(next<.01f?int(0x80000000u):0));
}
__device__ __forceinline__ void coarsePair(FoldedGaussian f,float3 color,float x,float y,float* T,float C[2][3]){
  y+=.5f;const float power=fmaf(f.a.x,x*x,fmaf(f.a.z,y*y,fmaf(f.a.y,x*y,fmaf(f.a.w,x,fmaf(f.b.x,y,f.b.y)))));
  const float alpha=encodedAlpha(power,false);coarseBlend(alpha,color,T[0],C[0]);coarseBlend(alpha,color,T[1],C[1]);
}
__device__ __forceinline__ void diagnosedCoarsePair(DebugFoldedGaussian f,float3 color,float x,float y,float* T,float C[2][3],unsigned id,unsigned* flags,unsigned long long* visits){
  y+=.5f;const float power=fmaf(f.a.x,x*x,fmaf(f.a.z,y*y,fmaf(f.a.y,x*y,fmaf(f.a.w,x,fmaf(f.b.x,y,f.b.y)))));
  const float alpha=encodedAlpha(power,false);
  for(int p=0;p<2;++p){if(T[p]>=0.f&&alpha*T[p]>0.f){atomicOr(flags+id,1u);++*visits;}coarseBlend(alpha,color,T[p],C[p]);}
}
'''
    for name in ['encoded_sign_pixels.cuh', 'encoded_sign_diagnosed.cuh']:
        text = (root / name).read_text()
        text = text[text.index('template<int Mode,int Kind,int Layout> __global__'):]
        text = text.replace('template<int Mode,int Kind,int Layout>', 'template<int Mode,int Kind,int Layout,int Quality>')
        text = text.replace('renderEncodedSign', 'renderCoarsePair')
        text = text.replace('__shared__ float4 abc[256];', '__shared__ unsigned near_pair[256];__shared__ float4 abc[256];')
        anchor = '      if constexpr(true){'
        assert text.count(anchor) == 1
        text = text.replace(anchor, '''      // log(alpha(y+1))-log(alpha(y))对整个16x16 Tile的保守上界。
      const float variation=fabsf(f.a.y)*7.5f+fabsf(f.a.z)*16.f+fabsf(f.b.x);
      near_pair[j]=variation<=(Quality==1?.01f:Quality==2?.03f:.1f);
''' + anchor)
        if name.endswith('_pixels.cuh'):
            anchor = '}else encodedSignPair<Kind,Step>({abc[j],ef[j]},rgb[j],lx,ly,T,C);'
            replacement = '}else if(near_pair[j])coarsePair({abc[j],ef[j]},rgb[j],lx,ly,T,C);\n      else encodedSignPair<Kind,Step>({abc[j],ef[j]},rgb[j],lx,ly,T,C);'
        else:
            anchor = '}else diagnosedEncodedSignPair<Kind,Step>({abc[j],ef[j]},rgb[j],lx,ly,T,C,lists[tile*capacity+j],flags,&visits);'
            replacement = '}else if(near_pair[j])diagnosedCoarsePair({abc[j],ef[j]},rgb[j],lx,ly,T,C,lists[tile*capacity+j],flags,&visits);\n      else diagnosedEncodedSignPair<Kind,Step>({abc[j],ef[j]},rgb[j],lx,ly,T,C,lists[tile*capacity+j],flags,&visits);'
        assert text.count(anchor) == 1, name
        text = text.replace(anchor, replacement)
        (root / name.replace('encoded_sign', 'pair_coarse')).write_text((helper if name.endswith('_pixels.cuh') else '') + text)
    path = root / 'graph_main.cu'
    text = path.read_text().replace('#include "encoded_sign_diagnosed.cuh"',
        '#include "encoded_sign_diagnosed.cuh"\n#include "pair_coarse_pixels.cuh"\n#include "pair_coarse_diagnosed.cuh"')
    text = text.replace('pixel_mode<=202', 'pixel_mode<=261').replace('pixel_mode_<=202', 'pixel_mode_<=261')
    anchor = next(line for line in text.splitlines() if 'if(pixel_mode_==202)renderEncodedSignPixels' in line)
    calls, diagnostics = [], []
    for mode, quality in [(241, 1), (242, 2), (243, 3)]:
        calls.append(f'      if(pixel_mode_=={mode+18})renderCoarsePairPixels<38,3,1,{quality}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);')
        diagnostics.append(f'    if(pixel_mode_=={mode+18})renderCoarsePairDiagnosedPixels<38,3,1,{quality}><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));')
    text = text.replace(anchor, anchor + '\n' + '\n'.join(calls))
    anchor = '    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '\n'.join(diagnostics) + '\n' + anchor)
    path.write_text(text.replace('StudentEncodedSignRasterizer', 'StudentPairCoarseRasterizer'))
    print('PAIR_COARSE_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()

"""投影属性按两个float4和一个float打包，减少像素循环的shared指令。"""
from pathlib import Path
import re
import shutil

HERE=Path(__file__).resolve().parent
source=HERE/'source'
assert not (HERE/'source_v3').exists()
shutil.copytree(source,HERE/'source_v3')
shutil.copy2(HERE/'build/mycartgs_render_count_scaling_v1.so',HERE/'reports/candidates_v3.so')
text=(source/'scaling_pixels.cuh').read_text()
pixel=text[text.index('template<int Mode> __global__ void renderScalingPixels'):]
pixel=pixel.replace('template<int Mode> __global__ void renderScalingPixels',
                    'template<int Unroll> __global__ void renderPackedPixels')
pixel=pixel.replace('__shared__ float cached_bound[Mode==2?256:1];__shared__ float2 cached_xy[256];__shared__ float4 cached_conic[256];__shared__ float3 cached_color[256];',
                    '__shared__ float4 cached_a[256],cached_b[256];__shared__ float cached_c[256];')
old="""int id=lists[tile*capacity+j];cached_xy[j]=xy[id];cached_conic[j]=conic[id];
      cached_color[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};if(Mode==2)cached_bound[j]=alphaPowerBound(conic[id].w);"""
new="""int id=lists[tile*capacity+j];float2 p=xy[id];float4 c=conic[id];
      cached_a[j]=make_float4(p.x,p.y,c.x,c.y);
      cached_b[j]=make_float4(c.z,c.w,colors[id*3],colors[id*3+1]);cached_c[j]=colors[id*3+2];"""
assert old in pixel
pixel=pixel.replace(old,new)
pixel=re.sub(r'    constexpr int Unroll=Mode==5\?2:.*?;\n','',pixel)
old='float2 point=cached_xy[j];float4 co=cached_conic[j];float3 color=cached_color[j];float bound=Mode==2?cached_bound[j]:-CUDART_INF_F;'
new="""float4 a=loadPackedFour(cached_a+j),b=loadPackedFour(cached_b+j);
      float2 point={a.x,a.y};float4 co={a.z,a.w,b.x,b.y};float3 color={b.z,b.w,cached_c[j]};"""
assert old in pixel
pixel=pixel.replace(old,new)
pixel=pixel.replace('float bound=-CUDART_INF_F;','')
pixel=pixel.replace('compositeScalingPair<Mode>(point,co,color,pix,done,T,C,bound);',
                    'compositeScalingPair<0>(point,co,color,pix,done,T,C,-CUDART_INF_F);')
assert 'Mode' not in pixel
load="""
__device__ __forceinline__ float4 loadPackedFour(const float4* p) {
  float4 x;const unsigned address=unsigned(__cvta_generic_to_shared(p));
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
      :"=f"(x.x),"=f"(x.y),"=f"(x.z),"=f"(x.w):"r"(address));
  return x;
}
"""
(source/'packed_pixels.cuh').write_text(load+pixel)
main=source/'graph_main.cu'
text=main.read_text().replace('#include "scaling_pixels.cuh"',
                            '#include "scaling_pixels.cuh"\n#include "packed_pixels.cuh"')
text=text.replace('pixel_mode<=14','pixel_mode<=17')
line=next(x for x in text.splitlines() if 'if(pixel_mode_==14)renderScalingPixels<9>' in x)
extra='\n'.join(f'      if(pixel_mode_=={m})renderPackedPixels<{u}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);'
                for m,u in [(15,8),(16,16),(17,32)])
text=text.replace(line,line+'\n'+extra)
main.write_text(text)
print('CANDIDATES_V4_READY',flush=True)

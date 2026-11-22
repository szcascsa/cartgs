"""按输入容量界缩小常驻共享属性，不分段加载，不增加循环同步。"""
from pathlib import Path
import re
import shutil

HERE=Path(__file__).resolve().parent
source=HERE/'source'
assert not (HERE/'source_direct_fixed').exists()
shutil.copytree(source,HERE/'source_direct_fixed')
shutil.copy2(HERE/'build/mycartgs_large_footprint_subtiles_v1.so',HERE/'reports/direct_tiles_fixed.so')
text=(source/'scaling_pixels.cuh').read_text()
pixel=text[text.index('template<int Mode> __global__ void renderScalingPixels'):]
pixel=pixel.replace('template<int Mode> __global__ void renderScalingPixels',
                    'template<int Cache,int Layout,int Unroll=32> __global__ void renderShortCachePixels')
pixel=pixel.replace('constexpr int Pixels=2,Rows=8,Threads=128;',
                    'constexpr int Pixels=Layout==3?1:2,Rows=16/Pixels,Threads=256/Pixels;')
pixel=pixel.replace('int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;',
"""int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
  if(Layout==1){x=blockIdx.x*16+(thread*2)%16;y=blockIdx.y*16+(thread*2)/16;}
  if(Layout==2){x=blockIdx.x*16+thread%16;y=blockIdx.y*16+(thread/16)*2;}""")
pixel=pixel.replace('pix[p]={(float)x,(float)(y+p*Rows)};inside[p]=x<W&&y+p*Rows<H;done[p]=!inside[p];',
"""pix[p]={(float)(x+(Layout==1?p:0)),(float)(y+(Layout==1?0:Layout==2?p:p*Rows))};
    inside[p]=pix[p].x<W&&pix[p].y<H;done[p]=!inside[p];""")
pixel=pixel.replace('out[ch*H*W+(y+p*Rows)*W+x]','out[ch*H*W+int(pix[p].y)*W+int(pix[p].x)]')
pixel=pixel.replace('int count=counts[tile];','int count=counts[tile];')
pixel=pixel.replace('__shared__ float cached_bound[Mode==2?256:1];','')
pixel=pixel.replace('cached_xy[256]','cached_xy[Cache]').replace('cached_conic[256]','cached_conic[Cache]').replace('cached_color[256]','cached_color[Cache]')
pixel=pixel.replace("if(Mode==2)cached_bound[j]=alphaPowerBound(conic[id].w);","")
pixel=re.sub(r'    constexpr int Unroll=Mode==5\?2:.*?;\n','',pixel)
pixel=pixel.replace('float bound=Mode==2?cached_bound[j]:-CUDART_INF_F;','')
pixel=pixel.replace('float bound=-CUDART_INF_F;','')
pixel=pixel.replace('compositeScalingPair<Mode>(point,co,color,pix,done,T,C,bound);',
"""#pragma unroll
      for(int p=0;p<Pixels;++p)compositeValues(point,co,color,pix[p],done[p],T[p],C[p]);""")
assert 'Mode' not in pixel
(source/'short_cache_pixels.cuh').write_text(pixel)
main=source/'graph_main.cu'
text=main.read_text().replace('#include "direct_tile_pixels.cuh"','#include "direct_tile_pixels.cuh"\n#include "short_cache_pixels.cuh"')
text=text.replace('pixel_mode<=39','pixel_mode<=45')
text=text.replace('const bool direct_tiles=pixel_mode_>=38;','const bool direct_tiles=pixel_mode_==38||pixel_mode_==39;')
line=next(x for x in text.splitlines() if 'if(pixel_mode_==37)renderEarlyExitPixels' in x)
extra=[]
for mode,layout,unroll in [(40,0,32),(41,1,32),(42,2,32),(43,3,32),(44,0,8),(45,0,16)]:
    for bound in (32,64,128,256):
        conditional=f'capacity<={bound}' if bound<256 else 'true'
        prefix='if' if bound==32 else 'else if'
        block='dim3(16,16)' if layout==3 else 'dim3(16,8)'
        extra.append(f'      {"if(pixel_mode_=="+str(mode)+"){ " if bound==32 else ""}{prefix}({conditional})renderShortCachePixels<{bound},{layout},{unroll}><<<grid,{block},0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);'+(' }' if bound==256 else ''))
main.write_text(text.replace(line,line+'\n'+'\n'.join(extra)))
print('SHORT_CACHE_READY',flush=True)

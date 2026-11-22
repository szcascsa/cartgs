"""按F2 SASS固定power的FMA方向，再共享两个像素相同的横向乘积。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
source=HERE/'source'
assert not (HERE/'source_compact_pixel').exists()
shutil.copytree(source,HERE/'source_compact_pixel')
shutil.copy2(HERE/'build/mycartgs_large_footprint_subtiles_v1.so',HERE/'reports/compact_pixel_first.so')
head=(HERE/'exact_common_head.cuh').read_text()
text=(source/'scaling_pixels.cuh').read_text()
pixel=text[text.index('template<int Mode> __global__ void renderScalingPixels'):]
pixel=pixel.replace('template<int Mode> __global__ void renderScalingPixels',
                    'template<int Unroll,int Early=32> __global__ void renderExactCommonPixels')
pixel=pixel.replace('Mode==2','false')
pixel=pixel.replace('    constexpr int Unroll=Mode==5?2:Mode==6?4:Mode==7?8:Mode==8?16:Mode==9?32:1;\n','')
pixel=pixel.replace('compositeScalingPair<Mode>(point,co,color,pix,done,T,C,bound);',
                    'compositeExactPair(point,co,color,pix,done,T,C);')
pixel=pixel.replace('if((j&31)==0)','if((j&(Early-1))==0)')
pixel=pixel.replace('float bound=false?cached_bound[j]:-CUDART_INF_F;','')
pixel=pixel.replace('float bound=-CUDART_INF_F;','')
assert 'Mode' not in pixel
(source/'exact_common_pixels.cuh').write_text(head+'\n'+pixel)
main=source/'graph_main.cu'
text=main.read_text().replace('#include "compact_pixel.cuh"','#include "compact_pixel.cuh"\n#include "exact_common_pixels.cuh"')
text=text.replace('pixel_mode<=64','pixel_mode<=68')
text=text.replace('if(pixel_mode_>=62) {','if(pixel_mode_>=62&&pixel_mode_<=64) {')
text=text.replace('if(pixel_mode_<62)compactTileBits','if(pixel_mode_<62||pixel_mode_>64)compactTileBits')
line=next(x for x in text.splitlines() if 'if(pixel_mode_==37)renderEarlyExitPixels' in x)
extra='\n'.join(f'      if(pixel_mode_=={m})renderExactCommonPixels<{unroll},{early}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);'
               for m,unroll,early in [(65,8,32),(66,16,32),(67,32,32),(68,32,8)])
main.write_text(text.replace(line,line+'\n'+extra))
print('EXACT_COMMON_READY',flush=True)

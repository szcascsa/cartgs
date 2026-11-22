"""限制像素核资源占用，检验多CTA并发能否摊薄大footprint合成成本。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
source=HERE/'source'
assert not (HERE/'source_shared_alpha').exists()
shutil.copytree(source,HERE/'source_shared_alpha')
shutil.copy2(HERE/'build/mycartgs_large_footprint_subtiles_v1.so',HERE/'reports/shared_alpha.so')
text=(source/'scaling_pixels.cuh').read_text()
pixel=text[text.index('template<int Mode> __global__ void renderScalingPixels'):]
pixel=pixel.replace('template<int Mode> __global__ void renderScalingPixels',
    'template<int Blocks,int Early=32> __global__ __launch_bounds__(128,Blocks) void renderOccupancyPixels')
pixel=pixel.replace('Mode==2','false')
pixel=pixel.replace('Mode==5?2:Mode==6?4:Mode==7?8:Mode==8?16:Mode==9?32:1','32')
pixel=pixel.replace('compositeScalingPair<Mode>','compositeScalingPair<9>')
pixel=pixel.replace('if((j&31)==0)','if((j&(Early-1))==0)')
assert 'Mode' not in pixel
(source/'occupancy_pixels.cuh').write_text(pixel)
main=source/'graph_main.cu'
text=main.read_text().replace('#include "shared_alpha_pixels.cuh"','#include "shared_alpha_pixels.cuh"\n#include "occupancy_pixels.cuh"')
text=text.replace('pixel_mode<=57','pixel_mode<=61')
line=next(x for x in text.splitlines() if 'if(pixel_mode_==37)renderEarlyExitPixels' in x)
extra='\n'.join(f'      if(pixel_mode_=={m})renderOccupancyPixels<{blocks},{early}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);'
               for m,blocks,early in [(58,12,32),(59,16,32),(60,12,8),(61,16,8)])
main.write_text(text.replace(line,line+'\n'+extra))
print('OCCUPANCY_READY',flush=True)

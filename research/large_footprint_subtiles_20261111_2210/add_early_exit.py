"""无需跨线程混合：让已结束的独立像素链及时退出长成员循环。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
source=HERE/'source'
assert not (HERE/'source_macrotiles').exists()
shutil.copytree(source,HERE/'source_macrotiles')
shutil.copy2(HERE/'build/mycartgs_large_footprint_subtiles_v1.so',HERE/'reports/subtiles_macros.so')
text=(source/'scaling_pixels.cuh').read_text()
pixel=text[text.index('template<int Mode> __global__ void renderScalingPixels'):]
pixel=pixel.replace('template<int Mode> __global__ void renderScalingPixels',
                    'template<int Early> __global__ void renderEarlyExitPixels')
pixel=pixel.replace('Mode==2','false')
pixel=pixel.replace('Mode==5?2:Mode==6?4:Mode==7?8:Mode==8?16:Mode==9?32:1','32')
pixel=pixel.replace('compositeScalingPair<Mode>','compositeScalingPair<9>')
pixel=pixel.replace('if((j&31)==0) {','if(Early==0 || (j&(Early-1))==0) {')
pixel=pixel.replace('if(__all_sync(0xffffffff,all_done))break;',
                    'if(Early==0?all_done:__all_sync(0xffffffff,all_done))break;')
assert 'Mode' not in pixel
(source/'early_exit_pixels.cuh').write_text(pixel)
main=source/'graph_main.cu'
text=main.read_text().replace('#include "macrotile_pixels.cuh"','#include "macrotile_pixels.cuh"\n#include "early_exit_pixels.cuh"')
text=text.replace('pixel_mode<=32','pixel_mode<=37')
line=next(x for x in text.splitlines() if 'if(pixel_mode_==32)renderMacrotilePixels' in x)
extra='\n'.join(f'      if(pixel_mode_=={mode})renderEarlyExitPixels<{early}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);'
                for mode,early in [(33,0),(34,2),(35,4),(36,8),(37,16)])
main.write_text(text.replace(line,line+'\n'+extra))
print('EARLY_EXIT_READY',flush=True)

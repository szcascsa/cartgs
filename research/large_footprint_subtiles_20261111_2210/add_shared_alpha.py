"""以shared保存独立Alpha，随后按原Gaussian顺序合成，避免寄存器批次膨胀。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
source=HERE/'source'
assert not (HERE/'source_cache_policies').exists()
shutil.copytree(source,HERE/'source_cache_policies')
shutil.copy2(HERE/'build/mycartgs_large_footprint_subtiles_v1.so',HERE/'reports/cache_policies.so')
shutil.copy2(HERE/'shared_alpha_pixels.cuh',source/'shared_alpha_pixels.cuh')
main=source/'graph_main.cu'
text=main.read_text().replace('#include "short_cache_pixels.cuh"','#include "short_cache_pixels.cuh"\n#include "shared_alpha_pixels.cuh"')
text=text.replace('pixel_mode<=51','pixel_mode<=57')
line=next(x for x in text.splitlines() if 'if(pixel_mode_==37)renderEarlyExitPixels' in x)
extra='\n'.join(f'      if(pixel_mode_=={mode})renderSharedAlphaPixels<{batch},{transpose}><<<grid,256,0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,inputs);'
                for mode,batch,transpose in [(52,4,False),(53,8,False),(54,16,False),(55,4,True),(56,8,True),(57,16,True)])
extra=extra.replace('<4,False>','<4,false>').replace('<8,False>','<8,false>').replace('<16,False>','<16,false>')
extra=extra.replace('<4,True>','<4,true>').replace('<8,True>','<8,true>').replace('<16,True>','<16,true>')
main.write_text(text.replace(line,line+'\n'+extra))
print('SHARED_ALPHA_READY',flush=True)

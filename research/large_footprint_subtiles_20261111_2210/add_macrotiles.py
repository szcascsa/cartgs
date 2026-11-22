"""邻近Tile共享原深度顺序的成员属性；像素链仍独立、顺序不变。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
source=HERE/'source'
assert not (HERE/'source_alpha').exists()
shutil.copytree(source,HERE/'source_alpha')
shutil.copy2(HERE/'build/mycartgs_large_footprint_subtiles_v1.so',HERE/'reports/subtiles_alpha.so')
shutil.copy2(HERE/'macrotile_pixels.cuh',source/'macrotile_pixels.cuh')
main=source/'graph_main.cu'
text=main.read_text().replace('#include "alpha_pipeline.cuh"','#include "alpha_pipeline.cuh"\n#include "macrotile_pixels.cuh"')
text=text.replace('pixel_mode<=28','pixel_mode<=32')
line=next(x for x in text.splitlines() if 'if(pixel_mode_==28)renderAlphaPipeline' in x)
extra='\n'.join(f'      if(pixel_mode_=={mode})renderMacrotilePixels<{across},{down},{unroll}><<<dim3((grid.x+{across-1})/{across},(grid.y+{down-1})/{down}),128*{across*down},0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,inputs);'
                for mode,across,down,unroll in [(29,2,1,8),(30,2,1,32),(31,2,2,8),(32,1,2,8)])
main.write_text(text.replace(line,line+'\n'+extra))
print('MACROTILE_READY',flush=True)

"""真实Tile成员数控制缓存使用，溢出用完整原目录，不分段加载。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
source=HERE/'source'
assert not (HERE/'source_short_cache').exists()
shutil.copytree(source,HERE/'source_short_cache')
shutil.copy2(HERE/'build/mycartgs_large_footprint_subtiles_v1.so',HERE/'reports/short_cache_first.so')
header=source/'short_cache_pixels.cuh'
text=header.read_text().replace('if(count<=capacity) {','if(count<=min(capacity,Cache)) {')
header.write_text(text)
main=source/'graph_main.cu'
text=main.read_text().replace('pixel_mode<=45','pixel_mode<=51')
line=next(x for x in text.splitlines() if 'if(pixel_mode_==37)renderEarlyExitPixels' in x)
extra='\n'.join(f'      if(pixel_mode_=={mode})renderShortCachePixels<{cache},{layout},32><<<grid,dim3(16,{block_y}),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);'
                for mode,cache,layout,block_y in [(46,32,0,8),(47,64,0,8),(48,128,0,8),(49,64,1,8),(50,128,1,8),(51,64,3,16)])
main.write_text(text.replace(line,line+'\n'+extra))
print('CACHE_POLICIES_READY',flush=True)

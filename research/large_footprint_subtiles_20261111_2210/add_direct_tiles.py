"""少点大footprint的Tile直接成员生成，避免写出再读取全图位表。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
source=HERE/'source'
assert not (HERE/'source_early').exists()
shutil.copytree(source,HERE/'source_early')
shutil.copy2(HERE/'build/mycartgs_large_footprint_subtiles_v1.so',HERE/'reports/subtiles_early.so')
shutil.copy2(HERE/'direct_tile_pixels.cuh',source/'direct_tile_pixels.cuh')
main=source/'graph_main.cu'
text=main.read_text().replace('#include "early_exit_pixels.cuh"','#include "early_exit_pixels.cuh"\n#include "direct_tile_pixels.cuh"')
text=text.replace('pixel_mode<=37','pixel_mode<=39')
text=text.replace('const bool fused_tasks=build_mode_==10||build_mode_==11;',
                  'const bool direct_tiles=pixel_mode_>=38;\n    const bool fused_tasks=!direct_tiles&&(build_mode_==10||build_mode_==11);')
start=text.index('    if(build_mode_==7||build_mode_==8) {')
end=text.index('    if(pixel_mode_>=6) {',start)
text=text[:start]+'    if(!direct_tiles){\n'+text[start:end]+'    }\n'+text[end:]
old='    if(pixel_mode_>=6) {'
new="""    if(direct_tiles) {
      if(pixel_mode_==38)renderDirectTiles<128><<<grid,128,0,capture>>>(capacity,W,H,sorted,key_out,touched,radii,xy,conic,rgb,inputs);
      else renderDirectTiles<64><<<grid,128,0,capture>>>(capacity,W,H,sorted,key_out,touched,radii,xy,conic,rgb,inputs);
    } else if(pixel_mode_>=6) {"""
text=text.replace(old,new)
main.write_text(text)
print('DIRECT_TILES_READY',flush=True)

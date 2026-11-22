"""清零、预处理、排序均保持F2；只减少成员列表写回与读回。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
source=HERE/'source'
assert not (HERE/'source_occupancy').exists()
shutil.copytree(source,HERE/'source_occupancy')
shutil.copy2(HERE/'build/mycartgs_large_footprint_subtiles_v1.so',HERE/'reports/occupancy_first.so')
shutil.copy2(HERE/'compact_pixel.cuh',source/'compact_pixel.cuh')
main=source/'graph_main.cu'
text=main.read_text().replace('#include "occupancy_pixels.cuh"','#include "occupancy_pixels.cuh"\n#include "compact_pixel.cuh"')
text=text.replace('pixel_mode<=61','pixel_mode<=64')
text=text.replace('      compactTileBits<<<','      if(pixel_mode_<62)compactTileBits<<<')
old='    if(direct_tiles) {\n      if(pixel_mode_==38)'
new="""    if(pixel_mode_>=62) {
      if(pixel_mode_==62)renderCompactPixel<8><<<grid,128,0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,touched,radii,capacity,inputs);
      if(pixel_mode_==63)renderCompactPixel<32><<<grid,128,0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,touched,radii,capacity,inputs);
      if(pixel_mode_==64)renderCompactPixel<16><<<grid,128,0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,touched,radii,capacity,inputs);
    } else if(direct_tiles) {
      if(pixel_mode_==38)"""
assert old in text
main.write_text(text.replace(old,new))
print('COMPACT_PIXEL_READY',flush=True)

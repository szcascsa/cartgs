"""保存首轮源码与二进制身份，扩展GPU实际K循环和独立像素warp候选。"""
from pathlib import Path
import shutil,tarfile

HERE=Path(__file__).resolve().parent
snapshot=HERE/'source_v1.tar.gz';assert not snapshot.exists()
with tarfile.open(snapshot,'w:gz') as archive:archive.add(HERE/'source',arcname='source')
path=HERE/'source/footprint_build.cuh';source=path.read_text(encoding='utf-8')
source=source.replace('__global__ void buildTileWords(', 'template<bool Dynamic=false> __global__ void buildTileWords(',1)
source=source.replace('const int* radii,uint32_t* bits,int words,dim3 grid) {\n  const int tile=',
    'const int* radii,uint32_t* bits,int words,dim3 grid,const GraphInputs* dynamic=nullptr) {\n  const int tile=',1)
source=source.replace('  for(int word=warp;word<words;word+=4) {',
    '  const int active_words=Dynamic?(min(n,dynamic->P)+31)/32:words;\n  for(int word=warp;word<active_words;word+=4) {',1)
needle='  if(tile==0&&!threadIdx.x){bits[int64_t(grid.x)*grid.y*words]=0;'
assert source.count(needle)==1
source=source.replace(needle,'  if(Dynamic)for(int word=active_words+threadIdx.x;word<words;word+=blockDim.x)bits[int64_t(tile)*words+word]=0;\n'+needle,1)
source=source.replace('__global__ void buildCompactedTiles(', 'template<bool Dynamic=false> __global__ void buildCompactedTiles(',1)
source=source.replace('uint32_t* lists,uint32_t* counts,int list_capacity,dim3 grid) {',
    'uint32_t* lists,uint32_t* counts,int list_capacity,dim3 grid,const GraphInputs* dynamic=nullptr) {',1)
source=source.replace('  for(int word=0;word<words;++word) {',
    '  const int active_words=Dynamic?(min(n,dynamic->P)+31)/32:words;\n  for(int word=0;word<active_words;++word) {',1)
source=source.replace('word==words-1','word==active_words-1',1)
source=source.replace('  if(!lane)counts[tile]=carry;',
    '  if(Dynamic){\n    for(int word=active_words+lane;word<words;word+=32)bits[int64_t(tile)*words+word]=0;\n    for(int group=(active_words+31)/32+lane;group<groups;group+=32)directory[tile*groups+group]=0;\n  }\n  if(!lane)counts[tile]=carry;',1)
path.write_text(source,encoding='utf-8')
path=HERE/'source/graph_main.cu';source=path.read_text(encoding='utf-8')
source=source.replace('#include "footprint_build.cuh"','#include "footprint_build.cuh"\n#include "warp_pixels.cuh"',1)
source=source.replace('FootprintGraphRasterizer','FootprintGraphRasterizerV2')
source=source.replace('int max_plans=64;int build_mode_=0;', 'int max_plans=64;int build_mode_=0;int pixel_mode_=0;',1)
source=source.replace('int build_mode=0):max_plans(count),build_mode_(build_mode),',
    'int build_mode=0,int pixel_mode=0):max_plans(count),build_mode_(build_mode),pixel_mode_(pixel_mode),',1)
source=source.replace('build_mode<=7&&ordered','build_mode<=9&&pixel_mode>=0&&pixel_mode<=3&&ordered',1)
source=source.replace('if(build_mode_==7) {','if(build_mode_==7||build_mode_==8) {',1)
needle='      buildCompactedTiles<<<(tiles+3)/4,128,0,capture>>>(capacity,sorted,xy,conic,rectangles,limits,\n          bits,words,directory,groups,lists,counts,list_capacity,grid);'
assert source.count(needle)==1
source=source.replace(needle,'      if(build_mode_==7)buildCompactedTiles<false><<<(tiles+3)/4,128,0,capture>>>(capacity,sorted,xy,conic,rectangles,limits,bits,words,directory,groups,lists,counts,list_capacity,grid);\n      else buildCompactedTiles<true><<<(tiles+3)/4,128,0,capture>>>(capacity,sorted,xy,conic,rectangles,limits,bits,words,directory,groups,lists,counts,list_capacity,grid,inputs);',1)
source=source.replace('if(build_mode_!=6) {','if(build_mode_!=6&&build_mode_!=9) {',1)
source=source.replace('else if(build_mode_==6)buildTileWords<<<tiles,128,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid);',
    'else if(build_mode_==6)buildTileWords<false><<<tiles,128,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid);\n      else if(build_mode_==9)buildTileWords<true><<<tiles,128,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid,inputs);',1)
needle='    if(variant==5)renderMultiPixel<2>'
assert source.count(needle)==1
source=source.replace(needle,'''    if(pixel_mode_==1)renderWarpPixels<8><<<grid,dim3(16,2),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,inputs);
    else if(pixel_mode_==2)renderWarpPixels<4><<<grid,dim3(16,4),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,inputs);
    else if(pixel_mode_==3)renderWarpPixels<2><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,inputs);
    else if(variant==5)renderMultiPixel<2>''',1)
source=source.replace('pybind11::init<int,bool,bool,int>()','pybind11::init<int,bool,bool,int,int>()',1)
source=source.replace('pybind11::arg("build_mode")=0)', 'pybind11::arg("build_mode")=0,pybind11::arg("pixel_mode")=0)',1)
path.write_text(source,encoding='utf-8');print('FOOTPRINT_V2_GENERATED')

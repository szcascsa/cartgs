"""独立V3：直接只读属性加载和较小预处理CTA；保留V2源/二进制。"""
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
snapshot=HERE/'source_v2.tar.gz';assert not snapshot.exists()
with tarfile.open(snapshot,'w:gz') as archive:archive.add(HERE/'source',arcname='source')
path=HERE/'source/graph_kernels.cuh';source=path.read_text(encoding='utf-8')
start=source.index('template<int Pixels> __global__ void renderMultiPixel(')
end=source.index('// Tile成员只在本Block枚举一次',start)
direct=source[start:end].replace('renderMultiPixel(', 'renderDirectPixels(',1)
begin=direct.index('    __shared__ float2 cached_xy[256];')
finish=direct.index('    for(int j=0;j<count;++j) {',begin)
direct=direct[:begin]+direct[finish:]
needle='      float2 point=cached_xy[j];float4 co=cached_conic[j];float3 color=cached_color[j];'
assert direct.count(needle)==1
direct=direct.replace(needle,'      const int id=lists[tile*capacity+j];\n      const float2 point=xy[id];const float4 co=conic[id];const float3 color={colors[id*3],colors[id*3+1],colors[id*3+2]};',1)
(HERE/'source/direct_pixels.cuh').write_text('// 不预分配每Tile 9KiB共享属性；低成员数直接复用GPU只读缓存。\n'+direct,encoding='utf-8')
path=HERE/'source/graph_forward.cu';source=path.read_text(encoding='utf-8')
source=source.replace('uint32_t* depth_keys,uint32_t* depth_ids,dim3 grid,cudaStream_t stream) {',
    'uint32_t* depth_keys,uint32_t* depth_ids,dim3 grid,cudaStream_t stream,int threads) {',1)
source=source.replace('preprocessCUDA<3,true,true,false,true><<<(capacity+255)/256,256,0,stream>>>',
    'preprocessCUDA<3,true,true,false,true><<<(capacity+threads-1)/threads,threads,0,stream>>>',1)
path.write_text(source,encoding='utf-8')
path=HERE/'source/graph_inputs.h';source=path.read_text(encoding='utf-8').replace('dim3 grid,cudaStream_t stream);','dim3 grid,cudaStream_t stream,int threads=256);',1)
path.write_text(source,encoding='utf-8')
path=HERE/'source/graph_main.cu';source=path.read_text(encoding='utf-8').replace('FootprintGraphRasterizerV2','FootprintGraphRasterizerV3')
source=source.replace('#include "warp_pixels.cuh"','#include "warp_pixels.cuh"\n#include "direct_pixels.cuh"',1)
source=source.replace('int pixel_mode_=0;', 'int pixel_mode_=0;int preprocess_threads_=256;',1)
source=source.replace('int pixel_mode=0):max_plans', 'int pixel_mode=0,int preprocess_threads=256):max_plans',1)
source=source.replace('pixel_mode_(pixel_mode),ordered_', 'pixel_mode_(pixel_mode),preprocess_threads_(preprocess_threads),ordered_',1)
source=source.replace('pixel_mode<=3&&ordered', 'pixel_mode<=5&&(preprocess_threads==64||preprocess_threads==128||preprocess_threads==256)&&ordered',1)
source=source.replace('id_in,grid,capture);','id_in,grid,capture,preprocess_threads_);',1)
needle='    if(pixel_mode_==1)renderWarpPixels<8>'
assert source.count(needle)==1
source=source.replace(needle,'''    if(pixel_mode_==4)renderDirectPixels<2><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
    else if(pixel_mode_==5)renderDirectPixels<1><<<grid,dim3(16,16),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
    else if(pixel_mode_==1)renderWarpPixels<8>''',1)
source=source.replace('pybind11::init<int,bool,bool,int,int>()','pybind11::init<int,bool,bool,int,int,int>()',1)
source=source.replace('pybind11::arg("pixel_mode")=0)', 'pybind11::arg("pixel_mode")=0,pybind11::arg("preprocess_threads")=256)',1)
path.write_text(source,encoding='utf-8')
path=HERE/'footprint_runtime.py';source=path.read_text(encoding='utf-8').replace('build_v2','build_v3').replace('large_footprint_graph_v2','large_footprint_graph_v3')
source=source.replace('\n\n@lru_cache',"\nMODES.update({15:'pixel_direct2',16:'pixel_direct1',17:'preprocess64',18:'preprocess128',19:'gaussian4_pre64',20:'gaussian4_pre64_direct2'})\n\n@lru_cache",1)
source=source.replace('build_mode={10:0,11:0,12:0,13:8,14:4}', 'build_mode={10:0,11:0,12:0,13:8,14:4,15:0,16:0,17:0,18:0,19:5,20:5}',1)
source=source.replace('pixel_mode={10:1,11:2,12:3,13:1,14:1}', 'pixel_mode={10:1,11:2,12:3,13:1,14:1,15:4,16:5,20:4}',1)
source=source.replace('max_plans,True,True,build_mode,pixel_mode)', 'max_plans,True,True,build_mode,pixel_mode,{17:64,18:128,19:64,20:64}.get(mode,256))',1)
path.write_text(source,encoding='utf-8');print('FOOTPRINT_V3_GENERATED')

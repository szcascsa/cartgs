"""V4：把大footprint任务生成融入已有预处理，保留稳定排序和有序成员。"""
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
snapshot=HERE/'source_v3.tar.gz';assert not snapshot.exists()
with tarfile.open(snapshot,'w:gz') as archive:archive.add(HERE/'source',arcname='source')
path=HERE/'source/graph_forward.cu';source=path.read_text(encoding='utf-8')
source=source.replace('bool Dynamic = false>', 'bool Dynamic = false, bool Schedule = false>',1)
source=source.replace('const GraphInputs* dynamic=nullptr) {','const GraphInputs* dynamic=nullptr,uint2* tasks=nullptr,uint32_t* task_count=nullptr) {',1)
needle='  if(Metadata) {\n    footprint_rects[idx]='
assert source.count(needle)==1
source=source.replace(needle,'''  if(Schedule) {
    const unsigned chunks=(tiles_touched[idx]+255)/256;
    const unsigned first=atomicAdd(task_count,chunks);
    for(unsigned chunk=0;chunk<chunks;++chunk)tasks[first+chunk]=make_uint2(idx,chunk*256);
  }
'''+needle,1)
source=source.replace('dim3 grid,cudaStream_t stream,int threads) {','dim3 grid,cudaStream_t stream,int threads,uint2* tasks,uint32_t* task_count) {',1)
start=source.index('  preprocessCUDA<3,true,true,false,true><<<',source.index('void graph_preprocess('))
end=source.index('\n}',start)
old=source[start:end]
schedule=old.replace('preprocessCUDA<3,true,true,false,true>', 'preprocessCUDA<3,true,true,false,true,true>',1).replace('nullptr,nullptr,inputs);','nullptr,nullptr,inputs,tasks,task_count);',1)
source=source[:start]+'  if(tasks) {\n'+schedule+'\n  } else {\n'+old+'\n  }'+source[end:]
path.write_text(source,encoding='utf-8')
path=HERE/'source/graph_inputs.h';source=path.read_text(encoding='utf-8').replace('int threads=256);','int threads=256,uint2* tasks=nullptr,uint32_t* task_count=nullptr);',1);path.write_text(source,encoding='utf-8')
path=HERE/'source/graph_kernels.cuh';source=path.read_text(encoding='utf-8')
source=source.replace('uint32_t* output,uint32_t* ids,bool final) {\n  int index=', 'uint32_t* output,uint32_t* ids,bool final,uint32_t* inverse=nullptr) {\n  int index=',1)
source=source.replace('if(final){output[index]=value>>32;ids[index]=uint32_t(value);}', 'if(final){output[index]=value>>32;ids[index]=uint32_t(value);if(inverse)inverse[uint32_t(value)]=index;}',1)
source=source.replace('uint32_t* output,uint32_t* ids,bool final) {\n  using Sort=cub::BlockRadixSort<uint32_t,128,4,uint32_t>;',
    'uint32_t* output,uint32_t* ids,bool final,uint32_t* inverse=nullptr) {\n  using Sort=cub::BlockRadixSort<uint32_t,128,4,uint32_t>;',1)
source=source.replace('if(final){output[i]=keys[j];ids[i]=values[j];}', 'if(final){output[i]=keys[j];ids[i]=values[j];if(inverse)inverse[values[j]]=i;}',1)
path.write_text(source,encoding='utf-8')
path=HERE/'source/graph_main.cu';source=path.read_text(encoding='utf-8').replace('FootprintGraphRasterizerV3','FootprintGraphRasterizerV4')
source=source.replace('#include "direct_pixels.cuh"','#include "direct_pixels.cuh"\n#include "preprocess_tasks.cuh"',1)
source=source.replace('build_mode<=9', 'build_mode<=11',1)
needle='    graph_preprocess(capacity,inputs,radii,xy,depths,covariance,rgb,conic,clamped,touched,key_in,id_in,grid,capture,preprocess_threads_);'
assert source.count(needle)==1
source=source.replace(needle,'''    const bool fused_tasks=build_mode_==10||build_mode_==11;
    auto inverse=fused_tasks?reinterpret_cast<uint32_t*>(pool_metadata.data_ptr()):nullptr;
    if(fused_tasks)clearGraphBits<<<(bit_bytes/4+255)/256,256,0,capture>>>(bits,bit_bytes/4);
    graph_preprocess(capacity,inputs,radii,xy,depths,covariance,rgb,conic,clamped,touched,key_in,id_in,grid,capture,
        preprocess_threads_,fused_tasks?tasks:nullptr,fused_tasks?task_count:nullptr);''',1)
source=source.replace('sorted,capacity<=512);','sorted,capacity<=512,inverse);',1)
source=source.replace('sorted,width*2>=capacity);','sorted,width*2>=capacity,inverse);',1)
source=source.replace('if(build_mode_!=6&&build_mode_!=9) {', 'if(build_mode_!=6&&build_mode_!=9&&!fused_tasks) {',1)
needle='      else if(build_mode_==9)buildTileWords<true><<<tiles,128,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid,inputs);'
assert source.count(needle)==1
source=source.replace(needle,needle+'''
      else if(build_mode_==10)executePreprocessedTasks<32><<<workers,128,0,capture>>>(tasks,task_count,inverse,xy,conic,radii,bits,words,grid);
      else if(build_mode_==11)executePreprocessedTasks<128><<<workers,128,0,capture>>>(tasks,task_count,inverse,xy,conic,radii,bits,words,grid);''',1)
path.write_text(source,encoding='utf-8')
path=HERE/'source/direct_pixels.cuh';source=path.read_text(encoding='utf-8').replace('constexpr int Rows=16/Pixels,Threads=256/Pixels;', 'constexpr int Rows=16/Pixels;',1).replace('int tile=blockIdx.y*gridDim.x+blockIdx.x,thread=threadIdx.y*blockDim.x+threadIdx.x;', 'int tile=blockIdx.y*gridDim.x+blockIdx.x;',1);path.write_text(source,encoding='utf-8')
path=HERE/'footprint_runtime.py';source=path.read_text(encoding='utf-8').replace('build_v3','build_v4').replace('large_footprint_graph_v3','large_footprint_graph_v4')
source=source.replace('\n@lru_cache',"\nMODES.update({21:'fused_tasks_warp64',22:'fused_tasks_block64',23:'fused_tasks_block128',24:'fused_tasks_warp128'})\n\n@lru_cache",1)
source=source.replace('19:5,20:5}', '19:5,20:5,21:10,22:11,23:11,24:10}',1)
source=source.replace('{17:64,18:128,19:64,20:64}', '{17:64,18:128,19:64,20:64,21:64,22:64,23:128,24:128}',1)
path.write_text(source,encoding='utf-8');print('FOOTPRINT_V4_GENERATED')

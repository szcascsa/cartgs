"""精确渲染候选：预处理发布有界footprint元数据，避免每256-Tile任务重复解码BBox/log。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_task_metadata'
    shutil.copytree(HERE / 'source_encoded_sign', root)
    (HERE / 'build_task_metadata').mkdir()
    path = root / 'graph_inputs.h'
    text = path.read_text()
    assert text.count('uint32_t* task_count=nullptr);') == 1
    path.write_text(text.replace('uint32_t* task_count=nullptr);', 'uint32_t* task_count=nullptr,uint4* footprint_rects=nullptr,float* footprint_limits=nullptr);'))
    path = root / 'graph_forward.cu'
    text = path.read_text()
    anchor = 'uint32_t* depth_keys,uint32_t* depth_ids,dim3 grid,cudaStream_t stream,int threads,uint2* tasks,uint32_t* task_count) {'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor.replace('task_count) {', 'task_count,uint4* footprint_rects,float* footprint_limits) {'))
    anchor = '  if(tasks) {\n  preprocessCUDA<3,true,true,false,true,true>'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''  if(footprint_rects){
  preprocessCUDA<3,true,true,true,true,true><<<(capacity+threads-1)/threads,threads,0,stream>>>(capacity,0,0,
      nullptr,nullptr,1.f,nullptr,nullptr,nullptr,nullptr,clamped,nullptr,nullptr,nullptr,nullptr,nullptr,
      0,0,0.f,0.f,0.f,0.f,radii,xy,depths,covariance,rgb,conic,grid,touched,false,
      depth_keys,depth_ids,0,0,0,0,0,footprint_rects,footprint_limits,inputs,tasks,task_count);
  }else if(tasks) {
  preprocessCUDA<3,true,true,false,true,true>''')
    path.write_text(text)
    # 这里仍使用独立SoA rect/limit，避免改变预处理输出布局与数学。
    helper = '''template<int ThreadsPerTask> __global__ void executeCachedTasks(const uint2* tasks,const uint32_t* task_count,
    const uint32_t* inverse,const float2* xy,const float4* conic,const uint4* rects,const float* limits,uint32_t* bits,int words,dim3 grid){
  constexpr int TasksPerBlock=128/ThreadsPerTask;
  const int worker=blockIdx.x*TasksPerBlock+threadIdx.x/ThreadsPerTask,lane=threadIdx.x%ThreadsPerTask;
  for(unsigned task=worker;task<*task_count;task+=gridDim.x*TasksPerBlock){
    const uint2 work=tasks[task];const int id=work.x,rank=inverse[id];const uint4 rect=rects[id];
    const int width=rect.z-rect.x,area=width*(rect.w-rect.y);const float limit=limits[id];
    const float2 point=xy[id];const float4 co=conic[id];
    for(int k=work.y+lane;k<min(area,int(work.y)+256);k+=ThreadsPerTask){
      const int x=rect.x+k%width,y=rect.y+k/width;
      if(orderedTilePower(point,co,x,y)<=limit)atomicOr(bits+(y*grid.x+x)*words+rank/32,1u<<(rank&31));
    }
  }
}
'''
    (root / 'task_metadata.cuh').write_text(helper)
    path = root / 'graph_main.cu'
    text = path.read_text().replace('#include "preprocess_tasks.cuh"', '#include "preprocess_tasks.cuh"\n#include "task_metadata.cuh"')
    text = text.replace('build_mode<=11', 'build_mode<=12')
    text = text.replace('build_mode_==10||build_mode_==11', 'build_mode_==10||build_mode_==11||build_mode_==12')
    text = text.replace('buffer(int64_t(capacity)*20)', 'buffer(int64_t(capacity)*24)')
    text = text.replace('int64_t(capacity)*148+8192', 'int64_t(capacity)*152+8192')
    anchor = '    auto inverse=fused_tasks?reinterpret_cast<uint32_t*>(pool_metadata.data_ptr()):nullptr;'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '''
    auto rects=build_mode_==12?reinterpret_cast<uint4*>(inverse+capacity):nullptr;
    auto limits=build_mode_==12?reinterpret_cast<float*>(rects+capacity):nullptr;
''')
    anchor = 'preprocess_threads_,fused_tasks?tasks:nullptr,fused_tasks?task_count:nullptr);'
    assert text.count(anchor) == 1
    text = text.replace(anchor, 'preprocess_threads_,fused_tasks?tasks:nullptr,fused_tasks?task_count:nullptr,rects,limits);')
    anchor = next(line for line in text.splitlines() if 'else if(build_mode_==11)executePreprocessedTasks' in line)
    text = text.replace(anchor, anchor + '\n      else if(build_mode_==12)executeCachedTasks<32><<<workers,128,0,capture>>>(tasks,task_count,inverse,xy,conic,rects,limits,bits,words,grid);')
    path.write_text(text.replace('StudentEncodedSignRasterizer', 'StudentTaskMetadataRasterizer'))
    print('TASK_METADATA_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()

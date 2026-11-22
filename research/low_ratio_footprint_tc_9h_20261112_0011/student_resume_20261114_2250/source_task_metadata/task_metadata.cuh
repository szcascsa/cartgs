template<int ThreadsPerTask> __global__ void executeCachedTasks(const uint2* tasks,const uint32_t* task_count,
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

// 任务由本帧预处理生成；排序最后一阶段写inverse rank。只并行整数Tile成员构建。
// ThreadsPerTask=32为每Warp一个任务，128为每CTA一个任务；无K/Ratio路径分流。
template<int ThreadsPerTask,int Cut=0> __global__ void executePreprocessedTasks(const uint2* tasks,const uint32_t* task_count,
    const uint32_t* inverse,const float2* xy,const float4* conic,const int* radii,uint32_t* bits,int words,dim3 grid) {
  constexpr int TasksPerBlock=128/ThreadsPerTask;
  const int worker=blockIdx.x*TasksPerBlock+threadIdx.x/ThreadsPerTask;
  const int lane=threadIdx.x%ThreadsPerTask;
  const unsigned count=*task_count;
  for(unsigned task=worker;task<count;task+=gridDim.x*TasksPerBlock) {
    const uint2 work=tasks[task];const int id=work.x,rank=inverse[id];
    uint2 lo,hi;getRect(xy[id],radii[id],lo,hi,grid);
    const int width=hi.x-lo.x,area=width*(hi.y-lo.y);
    const float2 point=xy[id];const float4 co=conic[id];const float limit=logf(co.w/((Cut==0?1.f:Cut==1?3.f:Cut==2?4.f:Cut==3?6.f:8.f)/255.f));
    for(int k=work.y+lane;k<min(area,int(work.y)+256);k+=ThreadsPerTask) {
      const int x=lo.x+k%width,y=lo.y+k/width;
      if(orderedTilePower(point,co,x,y)<=limit)
        atomicOr(bits+(y*grid.x+x)*words+rank/32,1u<<(rank&31));
    }
  }
}

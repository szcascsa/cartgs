// Copyright follows the copied GRAPHDECO/CaRtGS reference; see reference sources.
// 独立推理候选：稳定Warp展开、短列表混合、共享深度序列Tile Bitset。
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cfloat>
#include "forward.h"
#include "graph_inputs.h"
#include "auxiliary.h"
#include "fused_preprocess.h"
#include "preprocess_sort.h"
#include "rasterizer_impl.h"
namespace cg=cooperative_groups;

void sparse_render_short(dim3,const uint2*,const uint32_t*,int,int,const float2*,const float*,const float4*,const float*,float*);

// 参考控制核及Tile有效性公式由只读快照逐字生成。
#include "reference_emit.cuh"
#include "ordered_tile_coverage.cuh"
__device__ bool touches(float2 xy,float4 co,int x,int y,float limit,bool ordered=false) {
  if(ordered)return orderedTilePower(xy,co,x,y)<=limit;
  glm::vec2 p;
  const float v=max_contrib_power_rect_gaussian_float<15,15>(co,xy,
      glm::vec2(x*16,y*16),glm::vec2((x+1)*16-1,(y+1)*16-1),p);
  return v<=limit;
}
__device__ uint64_t sentinel() {return (uint64_t(0xffffffff)<<32)|__float_as_uint(FLT_MAX);}

__global__ void emitWarp(int n,const float2* xy,const float4* co,const float* z,
    const uint32_t* offsets,uint64_t* keys,uint32_t* ids,const int* radii,dim3 grid) {
  const int lane=threadIdx.x&31,i=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
  if(i>=n||radii[i]<=0)return;
  uint2 lo,hi;getRect(xy[i],radii[i],lo,hi,grid);
  const float2 point=xy[i];const float4 conic=co[i];const float limit=logf(conic.w/(1.0f/255.0f));
  const int w=hi.x-lo.x,area=w*(hi.y-lo.y);uint32_t out=i?offsets[i-1]:0,end=offsets[i];
  for(int begin=0;begin<area;begin+=32) {
    int k=begin+lane,x=lo.x+k%w,y=lo.y+k/w;
    const bool valid=k<area && touches(point,conic,x,y,limit);
    unsigned mask=__ballot_sync(0xffffffff,valid);
    if(valid) {
      const unsigned rank=__popc(mask&((1u<<lane)-1));
      keys[out+rank]=(uint64_t(y*grid.x+x)<<32)|__float_as_uint(z[i]);ids[out+rank]=i;
    }
    out+=__popc(mask);
  }
  for(uint32_t j=out+lane;j<end;j+=32){keys[j]=sentinel();ids[j]=0xffffffff;}
}

__global__ void rangesKernel(int n,const uint64_t* keys,uint2* ranges,int tiles) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  unsigned tile=keys[i]>>32;
  if(!i){if(tile<tiles)ranges[tile].x=0;}
  else {unsigned prev=keys[i-1]>>32;if(prev!=tile){if(prev<tiles)ranges[prev].y=i;if(tile<tiles)ranges[tile].x=i;}}
  if(i==n-1&&tile<tiles)ranges[tile].y=n;
}
__global__ void depthKeys(int n,const float* z,const int* radii,uint32_t* keys,uint32_t* ids) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){keys[i]=radii[i]>0?__float_as_uint(z[i]):0xffffffff;ids[i]=i;}
}

// 整数(depth bits, input id)全序保证同深度稳定性；所有规模使用块排序再归并。
__global__ void blockDepthSort(int n,const uint32_t* input,uint64_t* temporary,
    uint32_t* output,uint32_t* ids,bool final) {
  using Sort=cub::BlockRadixSort<uint64_t,256,4>;
  __shared__ typename Sort::TempStorage storage;
  uint64_t keys[4];
  for(int j=0;j<4;++j) {
    unsigned i=blockIdx.x*1024+threadIdx.x*4+j;
    keys[j]=i<unsigned(n)?((uint64_t(input[i])<<32)|i):uint64_t(-1);
  }
  Sort(storage).Sort(keys);
  for(int j=0;j<4;++j) {
    unsigned i=blockIdx.x*1024+threadIdx.x*4+j;
    if(i<unsigned(n)) {
      if(final){output[i]=keys[j]>>32;ids[i]=uint32_t(keys[j]);}
      else temporary[i]=keys[j];
    }
  }
}

__global__ void mergeDepthSort(int n,int width,const uint64_t* input,uint64_t* temporary,
    uint32_t* output,uint32_t* ids,bool final,uint32_t* inverse=nullptr) {
  int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=n)return;
  int base=(index/(width*2))*(width*2),diagonal=index-base;
  int left=min(width,n-base),right=min(width,max(0,n-base-width));
  const uint64_t* A=input+base;const uint64_t* B=A+left;
  int lo=max(0,diagonal-right),hi=min(diagonal,left);
  while(lo<hi) {
    int i=(lo+hi)/2,j=diagonal-i;
    if(i<left&&j>0&&B[j-1]>=A[i])lo=i+1;else hi=i;
  }
  int i=lo,j=diagonal-i;
  uint64_t value=(i<left&&(j>=right||A[i]<B[j]))?A[i]:B[j];
  if(final){output[index]=value>>32;ids[index]=uint32_t(value);if(inverse)inverse[uint32_t(value)]=index;}else temporary[index]=value;
}

__global__ void blockDepthSort512(int n,const uint32_t* input,uint64_t* temporary,
    uint32_t* output,uint32_t* ids,bool final,uint32_t* inverse=nullptr) {
  using Sort=cub::BlockRadixSort<uint32_t,128,4,uint32_t>;
  __shared__ typename Sort::TempStorage storage;
  uint32_t keys[4],values[4];
  for(int j=0;j<4;++j) {
    unsigned i=blockIdx.x*512+threadIdx.x*4+j;
    keys[j]=i<unsigned(n)?input[i]:0xffffffff;values[j]=i;
  }
  // CUB BlockRadixSort文档保证稳定；padding位于所有真实输入之后。
  Sort(storage).Sort(keys,values);
  for(int j=0;j<4;++j) {
    unsigned i=blockIdx.x*512+threadIdx.x*4+j;
    if(i<unsigned(n)) {
      if(final){output[i]=keys[j];ids[i]=values[j];if(inverse)inverse[values[j]]=i;}
      else temporary[i]=(uint64_t(keys[j])<<32)|values[j];
    }
  }
}

template<int RadixBits> __global__ void blockDepthSort512Radix(int n,const uint32_t* input,uint64_t* temporary,
    uint32_t* output,uint32_t* ids,bool final) {
  using Sort=cub::BlockRadixSort<uint32_t,128,4,uint32_t,RadixBits>;
  __shared__ typename Sort::TempStorage storage;
  uint32_t keys[4],values[4];
  for(int j=0;j<4;++j) {
    unsigned i=blockIdx.x*512+threadIdx.x*4+j;
    keys[j]=i<unsigned(n)?input[i]:0xffffffff;values[j]=i;
  }
  Sort(storage).Sort(keys,values);
  for(int j=0;j<4;++j) {
    unsigned i=blockIdx.x*512+threadIdx.x*4+j;
    if(i<unsigned(n)) {
      if(final){output[i]=keys[j];ids[i]=values[j];}
      else temporary[i]=(uint64_t(keys[j])<<32)|values[j];
    }
  }
}

struct DepthIntegerLess {__device__ bool operator()(uint64_t left,uint64_t right)const{return left<right;}};
__global__ void blockDepthMerge512(int n,const uint32_t* input,uint64_t* temporary,
    uint32_t* output,uint32_t* ids,bool final) {
  using Sort=cub::BlockMergeSort<uint64_t,128,4>;
  __shared__ typename Sort::TempStorage storage;
  uint64_t keys[4];
  for(int j=0;j<4;++j) {
    unsigned i=blockIdx.x*512+threadIdx.x*4+j;
    keys[j]=i<unsigned(n)?((uint64_t(input[i])<<32)|i):uint64_t(-1);
  }
  Sort(storage).Sort(keys,DepthIntegerLess());
  for(int j=0;j<4;++j) {
    unsigned i=blockIdx.x*512+threadIdx.x*4+j;
    if(i<unsigned(n)) {
      if(final){output[i]=keys[j]>>32;ids[i]=uint32_t(keys[j]);}
      else temporary[i]=keys[j];
    }
  }
}
template<bool Metadata=false> __global__ void buildBits(int n,const uint32_t* sorted,const float2* xy,const float4* co,
    const int* radii,uint32_t* bits,int words,dim3 grid,const uint4* rectangles=nullptr,const float* limits=nullptr) {
  int lane=threadIdx.x&31,rank=(blockIdx.x*blockDim.x+threadIdx.x)>>5;if(rank>=n)return;
  int i=sorted[rank];if(radii[i]<=0)return;uint2 lo,hi;
  if(Metadata){uint4 r=rectangles[i];lo=make_uint2(r.x,r.y);hi=make_uint2(r.z,r.w);}else getRect(xy[i],radii[i],lo,hi,grid);
  const float2 point=xy[i];const float4 conic=co[i];const float limit=Metadata?limits[i]:logf(conic.w/(1.0f/255.0f));
  int w=hi.x-lo.x,area=w*(hi.y-lo.y);
  for(int k=lane;k<area;k+=32){int x=lo.x+k%w,y=lo.y+k/w;
    if(touches(point,conic,x,y,limit))atomicOr(bits+(y*grid.x+x)*words+rank/32,1u<<(rank&31));}
}

// 大footprint拆成固定256个Tile的任务；小footprint保留每Gaussian一个Warp。
// 任务次序不参与颜色累加，只以atomicOr写回稳定深度rank的成员位。
template<bool Metadata=false,bool Ordered=false> __global__ void scheduleBits(int n,const uint32_t* sorted,const float2* xy,const float4* co,
    const int* radii,uint32_t* bits,int words,dim3 grid,uint2* tasks,uint32_t* task_count,
    const uint4* rectangles=nullptr,const float* limits=nullptr) {
  int lane=threadIdx.x&31,rank=(blockIdx.x*blockDim.x+threadIdx.x)>>5;if(rank>=n)return;
  int i=sorted[rank];if(radii[i]<=0)return;uint2 lo,hi;
  if(Metadata){uint4 r=rectangles[i];lo=make_uint2(r.x,r.y);hi=make_uint2(r.z,r.w);}else getRect(xy[i],radii[i],lo,hi,grid);
  int w=hi.x-lo.x,area=w*(hi.y-lo.y);
  if(area>256) {
    int chunks=(area+255)/256;unsigned start=0;
    if(!lane)start=atomicAdd(task_count,chunks);
    start=__shfl_sync(0xffffffff,start,0);
    for(int j=lane;j<chunks;j+=32)tasks[start+j]=make_uint2(rank,j*256);
  } else {
    const float2 point=xy[i];const float4 conic=co[i];const float limit=Metadata?limits[i]:logf(conic.w/(1.0f/255.0f));
    for(int k=lane;k<area;k+=32){int x=lo.x+k%w,y=lo.y+k/w;
      if(touches(point,conic,x,y,limit,Ordered))atomicOr(bits+(y*grid.x+x)*words+rank/32,1u<<(rank&31));}
  }
}

template<bool Metadata=false,bool Ordered=false> __global__ void executeBitTasks(const uint2* tasks,const uint32_t* task_count,const uint32_t* sorted,
    const float2* xy,const float4* co,const int* radii,uint32_t* bits,int words,dim3 grid,
    const uint4* rectangles=nullptr,const float* limits=nullptr) {
  const unsigned count=*task_count;
  for(unsigned task=blockIdx.x;task<count;task+=gridDim.x) {
    const uint2 work=tasks[task];int rank=work.x,i=sorted[rank];
    uint2 lo,hi;
    if(Metadata){uint4 r=rectangles[i];lo=make_uint2(r.x,r.y);hi=make_uint2(r.z,r.w);}else getRect(xy[i],radii[i],lo,hi,grid);
    int w=hi.x-lo.x,area=w*(hi.y-lo.y);
    const float2 point=xy[i];const float4 conic=co[i];const float limit=Metadata?limits[i]:logf(conic.w/(1.0f/255.0f));
    for(int k=work.y+threadIdx.x;k<min(area,int(work.y)+256);k+=blockDim.x) {
      int x=lo.x+k%w,y=lo.y+k/w;
      if(touches(point,conic,x,y,limit,Ordered))atomicOr(bits+(y*grid.x+x)*words+rank/32,1u<<(rank&31));
    }
  }
}

// 每Gaussian一个CTA，CTA内并行真实footprint；所有输入规模使用同一分派。
__global__ void buildBitsGaussianBlock(int n,const uint32_t* sorted,const float2* xy,const float4* co,
    const int* radii,uint32_t* bits,int words,dim3 grid) {
  const int rank=blockIdx.x;if(rank>=n)return;
  const int i=sorted[rank];if(radii[i]<=0)return;
  uint2 lo,hi;getRect(xy[i],radii[i],lo,hi,grid);
  const int width=hi.x-lo.x,area=width*(hi.y-lo.y);
  const float2 point=xy[i];const float4 conic=co[i];const float limit=logf(conic.w/(1.0f/255.0f));
  for(int index=threadIdx.x;index<area;index+=blockDim.x) {
    int x=lo.x+index%width,y=lo.y+index/width;
    if(touches(point,conic,x,y,limit))atomicOr(bits+(y*grid.x+x)*words+rank/32,1u<<(rank&31));
  }
}
__global__ void renderBits(int n,int W,int H,const uint32_t* bits,int words,const uint32_t* sorted,
    const float2* xy,const float4* co,const float* colors,const float* bg,float* out) {
  int tile=blockIdx.y*gridDim.x+blockIdx.x;
  int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
  const bool inside=x<W&&y<H;bool done=!inside;float T=1.0f,C[3]={0};
  float2 pix={(float)x,(float)y};
  for(int w=0;w<words;++w) {
    if(__all_sync(0xffffffff,done))break;
    unsigned mask=bits[tile*words+w];
    while(mask) {
      int rank=w*32+__ffs(mask)-1;mask&=mask-1;
      const int i=sorted[rank];float2 point=xy[i];float4 con_o=co[i];
      if(!done) {
        float2 d={point.x-pix.x,point.y-pix.y};
        float power=-0.5f*(con_o.x*d.x*d.x+con_o.z*d.y*d.y)-con_o.y*d.x*d.y;
        if(power>0.0f)continue;
        float alpha=min(0.99f,con_o.w*exp(power));if(alpha<1.0f/255.0f)continue;
        float test_T=T*(1-alpha);if(test_T<0.0001f){done=true;continue;}
        for(int ch=0;ch<3;++ch)C[ch]+=colors[i*3+ch]*alpha*T;
        T=test_T;
      }
    }
  }
  if(inside)for(int ch=0;ch<3;++ch)out[ch*H*W+y*W+x]=C[ch]+T*bg[ch];
}

// 每32个位表字生成一个目录字，避免每像素线性扫描大量空字。
__global__ void summarizeBits(const uint32_t* bits,int words,int tiles,uint32_t* directory,int groups) {
  int lane=threadIdx.x&31,index=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
  if(index>=tiles*groups)return;
  int tile=index/groups,group=index%groups,w=group*32+lane;
  unsigned value=w<words?bits[tile*words+w]:0;
  unsigned mask=__ballot_sync(0xffffffff,value!=0);
  if(!lane)directory[index]=mask;
}

// 复现原入口num_rendered<=256且全sentinel时直接复制背景的符号位语义。
// 只由Tile0的一个Warp计算；最多检查257个有效深度，常规大footprint快速结束。
__device__ bool copySparseBackground(const uint32_t* bits,int words,int tiles,int n,const uint32_t* sorted,
    const uint32_t* keys,const uint32_t* touched,int lane) {
  if(keys[0]==0xffffffff)return true;
  unsigned total=0;const int limit=min(n,257);
  for(int base=0;base<limit;base+=32) {
    int rank=base+lane;unsigned value=rank<limit&&keys[rank]!=0xffffffff?touched[sorted[rank]]:0;
    for(int offset=16;offset;offset/=2)value+=__shfl_down_sync(0xffffffff,value,offset);
    total+=__shfl_sync(0xffffffff,value,0);
    if(total>256)return false;
  }
  if(total==0)return true;
  const int visible_words=min(words,8);
  for(int base=0;base<tiles*visible_words;base+=32) {
    int i=base+lane;bool nonempty=i<tiles*visible_words&&bits[(i/visible_words)*words+i%visible_words]!=0;
    if(__any_sync(0xffffffff,nonempty))return false;
  }
  return true;
}

// 一次按深度顺序枚举Tile成员，短列表占固定容量；溢出仅记录真实总数。
__global__ void compactTileBits(uint32_t* bits,int words,int tiles,const uint32_t* sorted,
    uint32_t* directory,int groups,uint32_t* lists,uint32_t* counts,int capacity,
    int n,const uint32_t* depth_keys,const uint32_t* touched) {
  int lane=threadIdx.x&31,tile=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
  if(tile>=tiles)return;
  unsigned carry=0;
  for(int group=0;group<groups;++group) {
    int w=group*32+lane;unsigned mask=w<words?bits[tile*words+w]:0;
    unsigned nonempty=__ballot_sync(0xffffffff,mask!=0);
    if(!lane)directory[tile*groups+group]=nonempty;
    unsigned count=__popc(mask),sum=count;
    for(int offset=1;offset<32;offset*=2){unsigned v=__shfl_up_sync(0xffffffff,sum,offset);if(lane>=offset)sum+=v;}
    unsigned position=carry+sum-count;
    for(unsigned remaining=mask;remaining;remaining&=remaining-1) {
      int rank=w*32+__ffs(remaining)-1;
      if(position<unsigned(capacity))lists[tile*capacity+position]=sorted[rank];
      ++position;
    }
    carry+=__shfl_sync(0xffffffff,sum,31);
  }
  if(!lane)counts[tile]=carry;
  if(tile==0) {
    bool copy=copySparseBackground(bits,words,tiles,n,sorted,depth_keys,touched,lane);
    if(!lane)bits[int64_t(tiles)*words+1]=copy;
  }
}

__device__ __forceinline__ void compositeOne(int id,const float2 pix,const float2* xy,const float4* conic,
    const float* colors,bool& done,float& T,float* C) {
  if(done)return;
  const float2 point=xy[id];const float4 con_o=conic[id];
  float2 d={point.x-pix.x,point.y-pix.y};
  float power=-0.5f*(con_o.x*d.x*d.x+con_o.z*d.y*d.y)-con_o.y*d.x*d.y;
  if(power>0.0f)return;
  float alpha=min(0.99f,con_o.w*exp(power));if(alpha<1.0f/255.0f)return;
  float test_T=T*(1-alpha);if(test_T<0.0001f){done=true;return;}
  for(int ch=0;ch<3;++ch)C[ch]+=colors[id*3+ch]*alpha*T;
  T=test_T;
}

template<bool Compact> __global__ void renderHierarchical(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,
    const uint32_t* counts,int capacity,const float2* xy,const float4* conic,const float* colors,
    const float* bg,float* out,const uint32_t* sorted_depth_keys) {
  int tile=blockIdx.y*gridDim.x+blockIdx.x;
  int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
  bool inside=x<W&&y<H,done=!inside;float T=1.0f,C[3]={0};float2 pix={(float)x,(float)y};
  // 原入口全不可见(L=0)直接复制背景，不能以0+1*bg改变负零符号。
  if(sorted_depth_keys[0]==0xffffffff||bits[int64_t(gridDim.x)*gridDim.y*words+1]) {
    if(inside)for(int ch=0;ch<3;++ch)out[ch*H*W+y*W+x]=bg[ch];
    return;
  }
  if(Compact&&counts[tile]<=unsigned(capacity)) {
    int count=counts[tile];
    for(int j=0;j<count;++j) {
      if((j&31)==0&&__all_sync(0xffffffff,done))break;
      compositeOne(lists[tile*capacity+j],pix,xy,conic,colors,done,T,C);
    }
  } else {
    for(int group=0;group<groups;++group) {
      if(__all_sync(0xffffffff,done))break;
      unsigned nonempty=directory[tile*groups+group];
      while(nonempty) {
        int w=group*32+__ffs(nonempty)-1;nonempty&=nonempty-1;
        unsigned members=bits[tile*words+w];
        while(members) {
          int rank=w*32+__ffs(members)-1;members&=members-1;
          compositeOne(sorted[rank],pix,xy,conic,colors,done,T,C);
        }
      }
    }
  }
  if(inside)for(int ch=0;ch<3;++ch)out[ch*H*W+y*W+x]=C[ch]+T*bg[ch];
}
__global__ void fillBackground(int count,int pixels,const float* bg,float* out) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)out[i]=bg[i/pixels];
}

// 只移动属性读取位置；每个像素仍严格按原列表顺序执行同一FP32表达式。
__device__ __forceinline__ void compositeValues(float2 point,float4 con_o,float3 color,float2 pix,
    bool& done,float& T,float* C) {
  if(done)return;
  float2 d={point.x-pix.x,point.y-pix.y};
  float power=-0.5f*(con_o.x*d.x*d.x+con_o.z*d.y*d.y)-con_o.y*d.x*d.y;
  if(power>0.0f)return;
  float alpha=min(0.99f,con_o.w*exp(power));if(alpha<1.0f/255.0f)return;
  float test_T=T*(1-alpha);if(test_T<0.0001f){done=true;return;}
  C[0]+=color.x*alpha*T;C[1]+=color.y*alpha*T;C[2]+=color.z*alpha*T;
  T=test_T;
}

__device__ __forceinline__ float alphaValue(float2 point,float4 con_o,float2 pix) {
  float2 d={point.x-pix.x,point.y-pix.y};
  float power=-0.5f*(con_o.x*d.x*d.x+con_o.z*d.y*d.y)-con_o.y*d.x*d.y;
  if(power>0.0f)return -1.f;
  float alpha=min(0.99f,con_o.w*exp(power));if(alpha<1.0f/255.0f)return -1.f;
  return alpha;
}

__device__ __forceinline__ void compositeAlpha(float alpha,float3 color,bool& done,float& T,float* C) {
  if(done||alpha<0.f)return;
  float test_T=T*(1-alpha);if(test_T<0.0001f){done=true;return;}
  C[0]+=color.x*alpha*T;C[1]+=color.y*alpha*T;C[2]+=color.z*alpha*T;
  T=test_T;
}

template<int Cache> __global__ void renderCached(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,
    const uint32_t* counts,int capacity,const float2* xy,const float4* conic,const float* colors,
    const float* bg,float* out,const uint32_t* sorted_depth_keys,const GraphInputs* dynamic=nullptr) {
  if(dynamic){bg=dynamic->background;out=dynamic->image;}
  int tile=blockIdx.y*gridDim.x+blockIdx.x,thread=threadIdx.y*blockDim.x+threadIdx.x,lane=thread&31;
  int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
  bool inside=x<W&&y<H,done=!inside;float T=1.0f,C[3]={0};float2 pix={(float)x,(float)y};
  if(sorted_depth_keys[0]==0xffffffff||bits[int64_t(gridDim.x)*gridDim.y*words+1]) {
    if(inside)for(int ch=0;ch<3;++ch)out[ch*H*W+y*W+x]=bg[ch];
    return;
  }
  const int count=counts[tile];
  if(count<=capacity) {
    if(Cache!=2) {
      __shared__ float2 cached_xy[256];
      __shared__ float4 cached_conic[256];
      __shared__ float3 cached_color[256];
      for(int j=thread;j<count;j+=256) {
        int id=lists[tile*capacity+j];cached_xy[j]=xy[id];cached_conic[j]=conic[id];
        cached_color[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};
      }
      __syncthreads();
      if(Cache==1) {
        for(int j=0;j<count;++j) {
          if((j&31)==0&&__all_sync(0xffffffff,done))break;
          compositeValues(cached_xy[j],cached_conic[j],cached_color[j],pix,done,T,C);
        }
      } else {
        constexpr int Batch=Cache==3?2:4;
        for(int base=0;base<count;base+=Batch) {
          if((base&31)==0&&__all_sync(0xffffffff,done))break;
          float alpha[Batch];
          #pragma unroll
          for(int j=0;j<Batch;++j)alpha[j]=base+j<count&&!done?alphaValue(cached_xy[base+j],cached_conic[base+j],pix):-1.f;
          #pragma unroll
          for(int j=0;j<Batch;++j)if(base+j<count)compositeAlpha(alpha[j],cached_color[base+j],done,T,C);
        }
      }
    } else {
      for(int base=0;base<count;base+=32) {
        if(__all_sync(0xffffffff,done))break;
        float2 cached_xy{};float4 cached_conic{};float3 cached_color{};
        if(base+lane<count) {
          int id=lists[tile*capacity+base+lane];cached_xy=xy[id];cached_conic=conic[id];
          cached_color={colors[id*3],colors[id*3+1],colors[id*3+2]};
        }
        const int batch=min(32,count-base);
        for(int j=0;j<batch;++j) {
          float2 point={__shfl_sync(0xffffffff,cached_xy.x,j),__shfl_sync(0xffffffff,cached_xy.y,j)};
          float4 co={__shfl_sync(0xffffffff,cached_conic.x,j),__shfl_sync(0xffffffff,cached_conic.y,j),
              __shfl_sync(0xffffffff,cached_conic.z,j),__shfl_sync(0xffffffff,cached_conic.w,j)};
          float3 color={__shfl_sync(0xffffffff,cached_color.x,j),__shfl_sync(0xffffffff,cached_color.y,j),__shfl_sync(0xffffffff,cached_color.z,j)};
          compositeValues(point,co,color,pix,done,T,C);
        }
      }
    }
  } else {
    for(int group=0;group<groups;++group) {
      if(__all_sync(0xffffffff,done))break;
      unsigned nonempty=directory[tile*groups+group];
      while(nonempty) {
        int w=group*32+__ffs(nonempty)-1;nonempty&=nonempty-1;unsigned members=bits[tile*words+w];
        while(members) {
          int rank=w*32+__ffs(members)-1;members&=members-1;
          compositeOne(sorted[rank],pix,xy,conic,colors,done,T,C);
        }
      }
    }
  }
  if(inside)for(int ch=0;ch<3;++ch)out[ch*H*W+y*W+x]=C[ch]+T*bg[ch];
}

// 计时外诊断：只有真正通过Alpha阈值且写入C的Gaussian才记为参与。
// 同时输出诊断图，调用者须与实际渲染逐位核对，避免诊断口径漂移。
__global__ void diagnoseAlpha(int W,int H,const uint32_t* bits,int words,const uint32_t* sorted,
    const float2* xy,const float4* conic,const float* colors,const float* bg,const uint32_t* sorted_keys,
    float* out,uint32_t* participating,unsigned long long* interactions) {
  int tile=blockIdx.y*gridDim.x+blockIdx.x;
  int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
  bool inside=x<W&&y<H,done=!inside;float T=1.f,C[3]={0};float2 pix={(float)x,(float)y};unsigned accepted=0;
  if(sorted_keys[0]==0xffffffff) {
    if(inside)for(int ch=0;ch<3;++ch)out[ch*H*W+y*W+x]=bg[ch];return;
  }
  for(int w=0;w<words;++w) {
    if(__all_sync(0xffffffff,done))break;
    unsigned mask=bits[tile*words+w];
    while(mask) {
      int rank=w*32+__ffs(mask)-1;mask&=mask-1;int id=sorted[rank];
      if(!done) {
        float2 point=xy[id];float4 co=conic[id];float2 d={point.x-pix.x,point.y-pix.y};
        float power=-0.5f*(co.x*d.x*d.x+co.z*d.y*d.y)-co.y*d.x*d.y;
        if(power>0.0f)continue;
        float alpha=min(0.99f,co.w*exp(power));if(alpha<1.0f/255.0f)continue;
        float test_T=T*(1-alpha);if(test_T<0.0001f){done=true;continue;}
        for(int ch=0;ch<3;++ch)C[ch]+=colors[id*3+ch]*alpha*T;
        T=test_T;atomicOr(participating+id,1u);++accepted;
      }
    }
  }
  if(accepted)atomicAdd(interactions,(unsigned long long)accepted);
  if(inside)for(int ch=0;ch<3;++ch)out[ch*H*W+y*W+x]=C[ch]+T*bg[ch];
}

// 一个线程交错处理2或4个独立像素，复用Gaussian属性并增加独立T链的指令并行度。
template<int Pixels> __global__ void renderMultiPixel(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,
    const uint32_t* counts,int capacity,const float2* xy,const float4* conic,const float* colors,
    const float* bg,float* out,const uint32_t* sorted_depth_keys,const GraphInputs* dynamic=nullptr) {
  if(dynamic){bg=dynamic->background;out=dynamic->image;}
  constexpr int Rows=16/Pixels,Threads=256/Pixels;
  int tile=blockIdx.y*gridDim.x+blockIdx.x,thread=threadIdx.y*blockDim.x+threadIdx.x;
  int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
  float T[Pixels],C[Pixels][3];float2 pix[Pixels];bool done[Pixels],inside[Pixels];
  #pragma unroll
  for(int p=0;p<Pixels;++p) {
    pix[p]={(float)x,(float)(y+p*Rows)};inside[p]=x<W&&y+p*Rows<H;done[p]=!inside[p];
    T[p]=1.f;C[p][0]=C[p][1]=C[p][2]=0.f;
  }
  if(sorted_depth_keys[0]==0xffffffff||bits[int64_t(gridDim.x)*gridDim.y*words+1]) {
    #pragma unroll
    for(int p=0;p<Pixels;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*Rows)*W+x]=bg[ch];
    return;
  }
  int count=counts[tile];
  if(count<=capacity) {
    __shared__ float2 cached_xy[256];__shared__ float4 cached_conic[256];__shared__ float3 cached_color[256];
    for(int j=thread;j<count;j+=Threads) {
      int id=lists[tile*capacity+j];cached_xy[j]=xy[id];cached_conic[j]=conic[id];
      cached_color[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};
    }
    __syncthreads();
    for(int j=0;j<count;++j) {
      if((j&31)==0) {
        bool all_done=true;
        #pragma unroll
        for(int p=0;p<Pixels;++p)all_done=all_done&&done[p];
        if(__all_sync(0xffffffff,all_done))break;
      }
      float2 point=cached_xy[j];float4 co=cached_conic[j];float3 color=cached_color[j];
      #pragma unroll
      for(int p=0;p<Pixels;++p)compositeValues(point,co,color,pix[p],done[p],T[p],C[p]);
    }
  } else {
    for(int group=0;group<groups;++group) {
      bool all_done=true;
      #pragma unroll
      for(int p=0;p<Pixels;++p)all_done=all_done&&done[p];
      if(__all_sync(0xffffffff,all_done))break;
      unsigned nonempty=directory[tile*groups+group];
      while(nonempty) {
        int w=group*32+__ffs(nonempty)-1;nonempty&=nonempty-1;unsigned members=bits[tile*words+w];
        while(members) {
          int rank=w*32+__ffs(members)-1;members&=members-1;int id=sorted[rank];
          float2 point=xy[id];float4 co=conic[id];float3 color={colors[id*3],colors[id*3+1],colors[id*3+2]};
          #pragma unroll
          for(int p=0;p<Pixels;++p)compositeValues(point,co,color,pix[p],done[p],T[p],C[p]);
        }
      }
    }
  }
  #pragma unroll
  for(int p=0;p<Pixels;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*Rows)*W+x]=C[p][ch]+T[p]*bg[ch];
}

// Tile成员只在本Block枚举一次，直接留在shared中进入像素混合，省去独立compact核。
__global__ void renderTileFused(int W,int H,const uint32_t* bits,int words,uint32_t* directory,int groups,
    const uint32_t* sorted,uint32_t* global_lists,uint32_t* counts,int capacity,
    const float2* xy,const float4* conic,const float* colors,const float* bg,float* out,
    const uint32_t* sorted_keys,bool inspect) {
  int tile=blockIdx.y*gridDim.x+blockIdx.x,thread=threadIdx.y*blockDim.x+threadIdx.x,lane=thread&31;
  int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
  bool inside0=x<W&&y<H,inside1=x<W&&y+8<H,done0=!inside0,done1=!inside1;
  float T0=1.f,T1=1.f,C0[3]={0},C1[3]={0};float2 pix0={(float)x,(float)y},pix1={(float)x,(float)(y+8)};
  if(sorted_keys[0]==0xffffffff) {
    if(!thread)counts[tile]=0;
    for(int group=thread;group<groups;group+=128)directory[tile*groups+group]=0;
    for(int ch=0;ch<3;++ch){if(inside0)out[ch*H*W+y*W+x]=bg[ch];if(inside1)out[ch*H*W+(y+8)*W+x]=bg[ch];}
    return;
  }
  __shared__ uint32_t members[256],count;
  __shared__ float2 cached_xy[256];__shared__ float4 cached_conic[256];__shared__ float3 cached_color[256];
  if(thread<32) {
    unsigned carry=0;
    for(int group=0;group<groups;++group) {
      int word=group*32+lane;unsigned mask=word<words?bits[tile*words+word]:0;
      unsigned nonempty=__ballot_sync(0xffffffff,mask!=0);
      if(!lane)directory[tile*groups+group]=nonempty;
      unsigned number=__popc(mask),sum=number;
      for(int offset=1;offset<32;offset*=2){unsigned value=__shfl_up_sync(0xffffffff,sum,offset);if(lane>=offset)sum+=value;}
      unsigned position=carry+sum-number;
      for(unsigned remaining=mask;remaining;remaining&=remaining-1) {
        int rank=word*32+__ffs(remaining)-1;
        if(position<unsigned(capacity)) {
          members[position]=sorted[rank];
          if(inspect)global_lists[tile*capacity+position]=sorted[rank];
        }
        ++position;
      }
      carry+=__shfl_sync(0xffffffff,sum,31);
    }
    if(!lane){count=carry;counts[tile]=carry;}
  }
  __syncthreads();
  if(count<=unsigned(capacity)) {
    for(int j=thread;j<int(count);j+=128) {
      int id=members[j];cached_xy[j]=xy[id];cached_conic[j]=conic[id];
      cached_color[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};
    }
    __syncthreads();
    for(int j=0;j<int(count);++j) {
      if((j&31)==0&&__all_sync(0xffffffff,done0&&done1))break;
      float2 point=cached_xy[j];float4 co=cached_conic[j];float3 color=cached_color[j];
      compositeValues(point,co,color,pix0,done0,T0,C0);compositeValues(point,co,color,pix1,done1,T1,C1);
    }
  } else {
    for(int group=0;group<groups;++group) {
      if(__all_sync(0xffffffff,done0&&done1))break;
      unsigned nonempty=directory[tile*groups+group];
      while(nonempty) {
        int word=group*32+__ffs(nonempty)-1;nonempty&=nonempty-1;unsigned mask=bits[tile*words+word];
        while(mask) {
          int rank=word*32+__ffs(mask)-1;mask&=mask-1;int id=sorted[rank];
          float2 point=xy[id];float4 co=conic[id];float3 color={colors[id*3],colors[id*3+1],colors[id*3+2]};
          compositeValues(point,co,color,pix0,done0,T0,C0);compositeValues(point,co,color,pix1,done1,T1,C1);
        }
      }
    }
  }
  for(int ch=0;ch<3;++ch){if(inside0)out[ch*H*W+y*W+x]=C0[ch]+T0*bg[ch];if(inside1)out[ch*H*W+(y+8)*W+x]=C1[ch]+T1*bg[ch];}
}

#include "chunked_pixels.cuh"


__global__ void updateGraphInputs(GraphInputs value,GraphInputs* destination){if(!threadIdx.x)*destination=value;}

__global__ void clearGraphBits(uint32_t* bits,int64_t count){int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)bits[i]=0;}

// stable key=(原始FP32深度位, 原始输入行ID)，不改变同深度次序。
__global__ void warpDepthSort32(int n,const uint32_t* input,uint32_t* output,
    uint32_t* ids,uint32_t* inverse) {
  const unsigned lane=threadIdx.x;
  uint64_t key=lane<unsigned(n)?(uint64_t(input[lane])<<32)|lane:uint64_t(-1);
  for(int size=2;size<=32;size*=2)for(int distance=size/2;distance;distance/=2) {
    const uint64_t other=(uint64_t(__shfl_xor_sync(0xffffffff,uint32_t(key>>32),distance))<<32)|
        __shfl_xor_sync(0xffffffff,uint32_t(key),distance);
    const bool low=(lane&distance)==0,ascending=(lane&size)==0;
    if((low==ascending)?other<key:other>key)key=other;
  }
  if(lane<unsigned(n)){output[lane]=key>>32;ids[lane]=uint32_t(key);if(inverse)inverse[uint32_t(key)]=lane;}
}

template<int Threads,int Items> __global__ void smallDepthSort(int n,const uint32_t* input,
    uint32_t* output,uint32_t* ids,uint32_t* inverse) {
  using Sort=cub::BlockRadixSort<uint32_t,Threads,Items,uint32_t>;
  __shared__ typename Sort::TempStorage storage;
  uint32_t keys[Items],values[Items];
  #pragma unroll
  for(int j=0;j<Items;++j){int i=threadIdx.x*Items+j;keys[j]=i<n?input[i]:0xffffffff;values[j]=i;}
  Sort(storage).Sort(keys,values);
  #pragma unroll
  for(int j=0;j<Items;++j){int i=threadIdx.x*Items+j;
    if(i<n){output[i]=keys[j];ids[i]=values[j];if(inverse)inverse[values[j]]=i;}}
}

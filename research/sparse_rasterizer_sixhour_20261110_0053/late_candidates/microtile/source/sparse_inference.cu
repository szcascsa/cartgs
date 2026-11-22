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
    uint32_t* output,uint32_t* ids,bool final) {
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
  if(final){output[index]=value>>32;ids[index]=uint32_t(value);}else temporary[index]=value;
}

__global__ void blockDepthSort512(int n,const uint32_t* input,uint64_t* temporary,
    uint32_t* output,uint32_t* ids,bool final) {
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
      if(final){output[i]=keys[j];ids[i]=values[j];}
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
    const float* bg,float* out,const uint32_t* sorted_depth_keys) {
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
    const float* bg,float* out,const uint32_t* sorted_depth_keys) {
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

template<int Slices> __global__ void renderMicrotile(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,
    const uint32_t* counts,int capacity,const float2* xy,const float4* conic,const float* colors,
    const float* bg,float* out,const uint32_t* sorted_depth_keys) {
  constexpr int Pixels=1,Rows=16/Slices,Threads=256/Slices;
  int tile=(blockIdx.y/Slices)*gridDim.x+blockIdx.x,thread=threadIdx.y*blockDim.x+threadIdx.x;
  int x=blockIdx.x*16+threadIdx.x,y=(blockIdx.y/Slices)*16+(blockIdx.y%Slices)*Rows+threadIdx.y;
  float T[Pixels],C[Pixels][3];float2 pix[Pixels];bool done[Pixels],inside[Pixels];
  #pragma unroll
  for(int p=0;p<Pixels;++p) {
    pix[p]={(float)x,(float)(y+p*Rows)};inside[p]=x<W&&y+p*Rows<H;done[p]=!inside[p];
    T[p]=1.f;C[p][0]=C[p][1]=C[p][2]=0.f;
  }
  if(sorted_depth_keys[0]==0xffffffff||bits[int64_t(gridDim.x)*(gridDim.y/Slices)*words+1]) {
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

using Tensor=torch::Tensor;
struct MicrotileStudyRasterizer {
  Tensor geom_,bin_,image_,bits_,sort_,directory_,lists_,counts_,tasks_,metadata_;int device_=-1;cudaStream_t stream_=nullptr;
  size_t max_bytes_=256ull<<20;
  static char* storage(Tensor& t,size_t bytes,torch::TensorOptions options) {
    // Scratch扩容丢弃旧内容，避免resize_复制未使用/未初始化的旧临时字节。
    if(!t.defined()||bytes>t.storage().nbytes())t=torch::empty({int64_t(bytes)},options.dtype(at::kByte));
    else t.resize_({int64_t(bytes)});
    return reinterpret_cast<char*>(t.data_ptr());
  }
  template<class T> Tensor snapshot(T* ptr,std::vector<int64_t> shape,at::ScalarType dtype,Tensor owner) {
    return torch::from_blob(ptr,shape,[owner](void*){},owner.options().dtype(dtype)).clone();
  }
  std::vector<Tensor> render(const Tensor& bg,const Tensor& xyz,const Tensor& colors,const Tensor& opacity,
      const Tensor& scales,const Tensor& rotation,double modifier,const Tensor& covariance,
      const Tensor& view,const Tensor& proj,double tanx,double tany,int64_t H,int64_t W,
      const Tensor& dc,const Tensor& sh,int64_t degree,const Tensor& camera,bool prefiltered,bool debug,
      int mode,bool short_lists,bool inspect,int render_variant,int list_capacity,bool fused_keys,bool strided_camera,int build_variant,int sort_variant,bool footprint_metadata,bool preprocess_sort,bool ordered_coverage) {
    TORCH_CHECK(!torch::GradMode::is_enabled(),"SparseRasterizer is frozen inference only");
    TORCH_CHECK(mode>=0&&mode<=4&&H>0&&W>0&&xyz.dim()==2&&xyz.size(1)==3,"Raster input/strategy");
    TORCH_CHECK(render_variant>=0&&render_variant<=11&&list_capacity>0&&list_capacity<=256,"Render variant/capacity");
    TORCH_CHECK(mode==4||render_variant==0,"Cached attributes require tile lists");
    TORCH_CHECK(!fused_keys||mode>=2,"Fused depth keys require depth sorting");
    TORCH_CHECK(build_variant>=0&&build_variant<=2&&(build_variant==0||mode>=2),"Tile build variant");
    TORCH_CHECK(sort_variant>=0&&sort_variant<=5&&(sort_variant==0||mode>=2),"Stable depth sort variant");
    TORCH_CHECK(!footprint_metadata||(fused_keys&&strided_camera&&mode==4),"Footprint metadata requires fused preprocess");
    TORCH_CHECK(!preprocess_sort||(fused_keys&&strided_camera&&sort_variant==2&&!footprint_metadata),"Preprocess/sort fusion config");
    TORCH_CHECK(!ordered_coverage||(build_variant==1&&!footprint_metadata),"Ordered coverage requires task build");
    c10::cuda::CUDAGuard guard(xyz.device());auto stream=c10::cuda::getCurrentCUDAStream();
    if(device_>=0)TORCH_CHECK(device_==xyz.get_device()&&stream_==stream.stream(),"Use one renderer per device/stream");
    device_=xyz.get_device();stream_=stream.stream();
    TORCH_CHECK(xyz.scalar_type()==at::kFloat&&bg.numel()==3,"FP32/RGB required");
    for(const auto& t:{bg,xyz,colors,opacity,scales,rotation,covariance,view,proj,dc,sh,camera})
      TORCH_CHECK(t.is_cuda()&&t.device()==xyz.device()&&t.scalar_type()==at::kFloat,"Input CUDA FP32/device");
    int P=xyz.size(0),M=sh.numel()?sh.size(1):0;
    auto out=mode>=3&&P?torch::empty({3,H,W},xyz.options()):torch::zeros({3,H,W},xyz.options());
    auto radii=mode>=3&&P?torch::empty({P},xyz.options().dtype(at::kInt)):torch::zeros({P},xyz.options().dtype(at::kInt));
    if(!P)return {out,radii};
    // 同原入口保留contiguous转换；所有计算状态逐帧重建。
    auto b=bg.contiguous(),x=xyz.contiguous(),c=colors.contiguous(),a=opacity.contiguous(),s=scales.contiguous();
    auto q=rotation.contiguous(),cov=covariance.contiguous(),v=strided_camera?view:view.contiguous(),p=strided_camera?proj:proj.contiguous();
    auto d=dc.contiguous(),h=sh.contiguous(),cam=strided_camera?camera:camera.contiguous();
    dim3 grid((W+15)/16,(H+15)/16);int tiles=grid.x*grid.y;
    size_t scan_bytes=0;cub::DeviceScan::InclusiveSum(nullptr,scan_bytes,(uint32_t*)nullptr,(uint32_t*)nullptr,P,stream);
    size_t gbytes=size_t(P)*(4+3+4+8+24+16+12+4+4)+scan_bytes+2048;
    char* ptr=storage(geom_,gbytes,xyz.options());
    if(inspect)C10_CUDA_CHECK(cudaMemsetAsync(ptr,0,gbytes,stream));
    using CudaRasterizer::obtain;
    float* depths;bool* clamped;int* internal;float2* xy;float* cov3;float4* conic;float* rgb;uint32_t* touched;uint32_t* offsets;char* scan;
    obtain(ptr,depths,P,128);obtain(ptr,clamped,3*P,128);obtain(ptr,internal,P,128);obtain(ptr,xy,P,128);
    obtain(ptr,cov3,6*P,128);obtain(ptr,conic,P,128);obtain(ptr,rgb,3*P,128);obtain(ptr,touched,P,128);
    obtain(ptr,offsets,P,128);obtain(ptr,scan,scan_bytes,128);
    uint64_t* sorted_keys=nullptr;uint32_t* sorted_ids=nullptr;uint2* ranges=nullptr;uint32_t* bits=nullptr;int L=0,words=(P+31)/32;
    size_t depth_sort_bytes=0;uint32_t *key_in=nullptr,*key_out=nullptr,*id_in=nullptr;char* depth_sort_temp=nullptr;
    uint64_t *merge_in=nullptr,*merge_out=nullptr;
    auto allocate_depth_sort=[&]() {
      if(sort_variant==0)C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,depth_sort_bytes,(uint32_t*)nullptr,(uint32_t*)nullptr,(uint32_t*)nullptr,(uint32_t*)nullptr,P,0,32,stream));
      char* allocation=storage(sort_,size_t(P)*(sort_variant?32:16)+depth_sort_bytes+2048,xyz.options());
      obtain(allocation,key_in,P,128);obtain(allocation,key_out,P,128);obtain(allocation,id_in,P,128);
      obtain(allocation,sorted_ids,P,128);obtain(allocation,depth_sort_temp,depth_sort_bytes,128);
      if(sort_variant){obtain(allocation,merge_in,P,128);obtain(allocation,merge_out,P,128);}
    };
    if(fused_keys)allocate_depth_sort();
    uint4* rectangles=nullptr;float* limits=nullptr;
    if(preprocess_sort) {
      sixhour_preprocess_sort(P,degree,M,x.data_ptr<float>(),reinterpret_cast<glm::vec3*>(s.data_ptr<float>()),modifier,
        reinterpret_cast<glm::vec4*>(q.data_ptr<float>()),a.data_ptr<float>(),d.data_ptr<float>(),h.data_ptr<float>(),clamped,
        cov.numel()?cov.data_ptr<float>():nullptr,c.numel()?c.data_ptr<float>():nullptr,v.data_ptr<float>(),p.data_ptr<float>(),
        reinterpret_cast<glm::vec3*>(cam.data_ptr<float>()),W,H,W/(2.0f*float(tanx)),H/(2.0f*float(tany)),
        tanx,tany,radii.data_ptr<int>(),xy,depths,cov3,rgb,conic,grid,touched,prefiltered,key_in,id_in,
        v.stride(0),v.stride(1),p.stride(0),p.stride(1),cam.stride(0),merge_in,key_out,sorted_ids,P<=512);
    }
    else if(footprint_metadata) {
      char* allocation=storage(metadata_,size_t(P)*20+512,xyz.options());
      obtain(allocation,rectangles,P,128);obtain(allocation,limits,P,128);
      sixhour_preprocess_metadata(P,degree,M,x.data_ptr<float>(),reinterpret_cast<glm::vec3*>(s.data_ptr<float>()),modifier,
        reinterpret_cast<glm::vec4*>(q.data_ptr<float>()),a.data_ptr<float>(),d.data_ptr<float>(),h.data_ptr<float>(),clamped,
        cov.numel()?cov.data_ptr<float>():nullptr,c.numel()?c.data_ptr<float>():nullptr,v.data_ptr<float>(),p.data_ptr<float>(),
        reinterpret_cast<glm::vec3*>(cam.data_ptr<float>()),W,H,W/(2.0f*float(tanx)),H/(2.0f*float(tany)),
        tanx,tany,radii.data_ptr<int>(),xy,depths,cov3,rgb,conic,grid,touched,prefiltered,key_in,id_in,
        v.stride(0),v.stride(1),p.stride(0),p.stride(1),cam.stride(0),rectangles,limits);
    }
    else if(strided_camera)sixhour_preprocess_strided(P,degree,M,x.data_ptr<float>(),reinterpret_cast<glm::vec3*>(s.data_ptr<float>()),modifier,
        reinterpret_cast<glm::vec4*>(q.data_ptr<float>()),a.data_ptr<float>(),d.data_ptr<float>(),h.data_ptr<float>(),clamped,
        cov.numel()?cov.data_ptr<float>():nullptr,c.numel()?c.data_ptr<float>():nullptr,v.data_ptr<float>(),p.data_ptr<float>(),
        reinterpret_cast<glm::vec3*>(cam.data_ptr<float>()),W,H,W/(2.0f*float(tanx)),H/(2.0f*float(tany)),
        tanx,tany,radii.data_ptr<int>(),xy,depths,cov3,rgb,conic,grid,touched,prefiltered,key_in,id_in,
        v.stride(0),v.stride(1),p.stride(0),p.stride(1),cam.stride(0),fused_keys);
    else if(fused_keys)sixhour_preprocess_keys(P,degree,M,x.data_ptr<float>(),reinterpret_cast<glm::vec3*>(s.data_ptr<float>()),modifier,
        reinterpret_cast<glm::vec4*>(q.data_ptr<float>()),a.data_ptr<float>(),d.data_ptr<float>(),h.data_ptr<float>(),clamped,
        cov.numel()?cov.data_ptr<float>():nullptr,c.numel()?c.data_ptr<float>():nullptr,v.data_ptr<float>(),p.data_ptr<float>(),
        reinterpret_cast<glm::vec3*>(cam.data_ptr<float>()),W,H,W/(2.0f*float(tanx)),H/(2.0f*float(tany)),
        tanx,tany,radii.data_ptr<int>(),xy,depths,cov3,rgb,conic,grid,touched,prefiltered,key_in,id_in);
    else FORWARD::preprocess(P,degree,M,x.data_ptr<float>(),reinterpret_cast<glm::vec3*>(s.data_ptr<float>()),modifier,
        reinterpret_cast<glm::vec4*>(q.data_ptr<float>()),a.data_ptr<float>(),d.data_ptr<float>(),h.data_ptr<float>(),clamped,
        cov.numel()?cov.data_ptr<float>():nullptr,c.numel()?c.data_ptr<float>():nullptr,v.data_ptr<float>(),p.data_ptr<float>(),
        reinterpret_cast<glm::vec3*>(cam.data_ptr<float>()),W,H,W/(2.0f*float(tanx)),H/(2.0f*float(tany)),
        tanx,tany,radii.data_ptr<int>(),xy,depths,cov3,rgb,conic,grid,touched,prefiltered);
    const float* feature=c.numel()?c.data_ptr<float>():rgb;
    if(mode<2) {
      C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(scan,scan_bytes,touched,offsets,P,stream));
      C10_CUDA_CHECK(cudaMemcpyAsync(&L,offsets+P-1,sizeof(int),cudaMemcpyDeviceToHost,stream));C10_CUDA_CHECK(cudaStreamSynchronize(stream));
      if(L>0) {
        size_t sort_bytes=0;cub::DeviceRadixSort::SortPairs(nullptr,sort_bytes,(uint64_t*)nullptr,(uint64_t*)nullptr,(uint32_t*)nullptr,(uint32_t*)nullptr,L,0,32+int(ceil(log2(double(tiles+1)))),stream);
        size_t nbytes=size_t(L)*24+sort_bytes+1024;TORCH_CHECK(nbytes<=max_bytes_,"Binning workspace exceeds 256 MiB");
        ptr=storage(bin_,nbytes,xyz.options());uint64_t* unsorted;uint32_t* ids;char* temp;
        obtain(ptr,unsorted,L,128);obtain(ptr,sorted_keys,L,128);obtain(ptr,ids,L,128);obtain(ptr,sorted_ids,L,128);obtain(ptr,temp,sort_bytes,128);
        if(mode==0)emitSerial<<<(P+255)/256,256,0,stream>>>(P,xy,conic,depths,offsets,unsorted,ids,radii.data_ptr<int>(),grid,nullptr);
        else emitWarp<<<(P+3)/4,128,0,stream>>>(P,xy,conic,depths,offsets,unsorted,ids,radii.data_ptr<int>(),grid);
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp,sort_bytes,unsorted,sorted_keys,ids,sorted_ids,L,0,32+int(ceil(log2(double(tiles+1)))),stream));
        ranges=reinterpret_cast<uint2*>(storage(image_,size_t(tiles)*sizeof(uint2),xyz.options()));
        C10_CUDA_CHECK(cudaMemsetAsync(ranges,0,size_t(tiles)*sizeof(uint2),stream));
        rangesKernel<<<(L+255)/256,256,0,stream>>>(L,sorted_keys,ranges,tiles);
        if(short_lists)sparse_render_short(grid,ranges,sorted_ids,W,H,xy,feature,conic,b.data_ptr<float>(),out.data_ptr<float>());
        else FORWARD::render(grid,dim3(16,16),ranges,sorted_ids,nullptr,nullptr,nullptr,nullptr,W,H,xy,feature,conic,
            nullptr,nullptr,nullptr,b.data_ptr<float>(),out.data_ptr<float>(),0,nullptr,nullptr,true);
      } else fillBackground<<<(W*H*3+255)/256,256,0,stream>>>(W*H*3,W*H,b.data_ptr<float>(),out.data_ptr<float>());
    } else {
      size_t bytes=size_t(tiles)*words*4;TORCH_CHECK(bytes<=max_bytes_,"Tile Bitset exceeds 256 MiB");
      size_t bit_bytes=bytes+2*sizeof(uint32_t);
      bits=reinterpret_cast<uint32_t*>(storage(bits_,bit_bytes,xyz.options()));C10_CUDA_CHECK(cudaMemsetAsync(bits,0,bit_bytes,stream));
      if(!fused_keys) {
        allocate_depth_sort();
        depthKeys<<<(P+255)/256,256,0,stream>>>(P,depths,radii.data_ptr<int>(),key_in,id_in);
      }
      if(sort_variant==0)C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(depth_sort_temp,depth_sort_bytes,key_in,key_out,id_in,sorted_ids,P,0,32,stream));
      else {
        const int block_size=sort_variant==1?1024:512;
        if(preprocess_sort){} // 首级Block排序已在预处理CTA内完成；后续归并不变。
        else if(sort_variant==2)blockDepthSort512<<<(P+511)/512,128,0,stream>>>(P,key_in,merge_in,key_out,sorted_ids,P<=512);
        else if(sort_variant==3)blockDepthSort512Radix<6><<<(P+511)/512,128,0,stream>>>(P,key_in,merge_in,key_out,sorted_ids,P<=512);
        else if(sort_variant==4)blockDepthSort512Radix<7><<<(P+511)/512,128,0,stream>>>(P,key_in,merge_in,key_out,sorted_ids,P<=512);
        else if(sort_variant==5)blockDepthMerge512<<<(P+511)/512,128,0,stream>>>(P,key_in,merge_in,key_out,sorted_ids,P<=512);
        else blockDepthSort<<<(P+1023)/1024,256,0,stream>>>(P,key_in,merge_in,key_out,sorted_ids,P<=1024);
        for(int width=block_size;width<P;width*=2) {
          mergeDepthSort<<<(P+255)/256,256,0,stream>>>(P,width,merge_in,merge_out,key_out,sorted_ids,width*2>=P);
          std::swap(merge_in,merge_out);
        }
      }
      if(build_variant==2)buildBitsGaussianBlock<<<P,128,0,stream>>>(P,sorted_ids,xy,conic,radii.data_ptr<int>(),bits,words,grid);
      else if(build_variant==0) {
        if(footprint_metadata)buildBits<true><<<(P+3)/4,128,0,stream>>>(P,sorted_ids,xy,conic,radii.data_ptr<int>(),bits,words,grid,rectangles,limits);
        else buildBits<false><<<(P+3)/4,128,0,stream>>>(P,sorted_ids,xy,conic,radii.data_ptr<int>(),bits,words,grid);
      }
      else {
        size_t task_bytes=size_t(P)*((tiles+255)/256)*sizeof(uint2);
        TORCH_CHECK(task_bytes<=max_bytes_,"Footprint task capacity exceeds 256 MiB");
        auto tasks=reinterpret_cast<uint2*>(storage(tasks_,task_bytes,xyz.options()));auto task_count=bits+size_t(tiles)*words;
        int workers=at::cuda::getCurrentDeviceProperties()->multiProcessorCount*4;
        if(footprint_metadata) {
          scheduleBits<true><<<(P+3)/4,128,0,stream>>>(P,sorted_ids,xy,conic,radii.data_ptr<int>(),bits,words,grid,tasks,task_count,rectangles,limits);
          executeBitTasks<true><<<workers,128,0,stream>>>(tasks,task_count,sorted_ids,xy,conic,radii.data_ptr<int>(),bits,words,grid,rectangles,limits);
        } else if(ordered_coverage) {
          scheduleBits<false,true><<<(P+3)/4,128,0,stream>>>(P,sorted_ids,xy,conic,radii.data_ptr<int>(),bits,words,grid,tasks,task_count);
          executeBitTasks<false,true><<<workers,128,0,stream>>>(tasks,task_count,sorted_ids,xy,conic,radii.data_ptr<int>(),bits,words,grid);
        } else {
          scheduleBits<false><<<(P+3)/4,128,0,stream>>>(P,sorted_ids,xy,conic,radii.data_ptr<int>(),bits,words,grid,tasks,task_count);
          executeBitTasks<false><<<workers,128,0,stream>>>(tasks,task_count,sorted_ids,xy,conic,radii.data_ptr<int>(),bits,words,grid);
        }
      }
      if(mode==2)renderBits<<<grid,dim3(16,16),0,stream>>>(P,W,H,bits,words,sorted_ids,xy,conic,feature,b.data_ptr<float>(),out.data_ptr<float>());
      else {
        const int groups=(words+31)/32,capacity=list_capacity;
        auto directory=reinterpret_cast<uint32_t*>(storage(directory_,size_t(tiles)*groups*4,xyz.options()));
        if(mode==3) {
          summarizeBits<<<(tiles*groups+7)/8,256,0,stream>>>(bits,words,tiles,directory,groups);
          renderHierarchical<false><<<grid,dim3(16,16),0,stream>>>(W,H,bits,words,directory,groups,sorted_ids,
              nullptr,nullptr,0,xy,conic,feature,b.data_ptr<float>(),out.data_ptr<float>(),key_out);
        } else {
          auto lists=reinterpret_cast<uint32_t*>(storage(lists_,size_t(tiles)*capacity*4,xyz.options()));
          auto counts=reinterpret_cast<uint32_t*>(storage(counts_,size_t(tiles)*4,xyz.options()));
          if(inspect)C10_CUDA_CHECK(cudaMemsetAsync(lists,0,size_t(tiles)*capacity*4,stream));
          if(render_variant!=7)compactTileBits<<<(tiles+3)/4,128,0,stream>>>(bits,words,tiles,sorted_ids,directory,groups,lists,counts,capacity,P,key_out,touched);
          if(render_variant==0)renderHierarchical<true><<<grid,dim3(16,16),0,stream>>>(W,H,bits,words,directory,groups,sorted_ids,
              lists,counts,capacity,xy,conic,feature,b.data_ptr<float>(),out.data_ptr<float>(),key_out);
          else if(render_variant==1)renderCached<1><<<grid,dim3(16,16),0,stream>>>(W,H,bits,words,directory,groups,sorted_ids,
              lists,counts,capacity,xy,conic,feature,b.data_ptr<float>(),out.data_ptr<float>(),key_out);
          else if(render_variant==2)renderCached<2><<<grid,dim3(16,16),0,stream>>>(W,H,bits,words,directory,groups,sorted_ids,
              lists,counts,capacity,xy,conic,feature,b.data_ptr<float>(),out.data_ptr<float>(),key_out);
          else if(render_variant==3)renderCached<3><<<grid,dim3(16,16),0,stream>>>(W,H,bits,words,directory,groups,sorted_ids,
              lists,counts,capacity,xy,conic,feature,b.data_ptr<float>(),out.data_ptr<float>(),key_out);
          else if(render_variant==4)renderCached<4><<<grid,dim3(16,16),0,stream>>>(W,H,bits,words,directory,groups,sorted_ids,
              lists,counts,capacity,xy,conic,feature,b.data_ptr<float>(),out.data_ptr<float>(),key_out);
          else if(render_variant==5)renderMultiPixel<2><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted_ids,
              lists,counts,capacity,xy,conic,feature,b.data_ptr<float>(),out.data_ptr<float>(),key_out);
          else if(render_variant==6)renderMultiPixel<4><<<grid,dim3(16,4),0,stream>>>(W,H,bits,words,directory,groups,sorted_ids,
              lists,counts,capacity,xy,conic,feature,b.data_ptr<float>(),out.data_ptr<float>(),key_out);
          else if(render_variant==10)renderMicrotile<2><<<dim3(grid.x,grid.y*2),dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted_ids,
              lists,counts,capacity,xy,conic,feature,b.data_ptr<float>(),out.data_ptr<float>(),key_out);
          else if(render_variant==11)renderMicrotile<4><<<dim3(grid.x,grid.y*4),dim3(16,4),0,stream>>>(W,H,bits,words,directory,groups,sorted_ids,
              lists,counts,capacity,xy,conic,feature,b.data_ptr<float>(),out.data_ptr<float>(),key_out);
          else if(render_variant==8)renderChunkedPixels<2,128><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted_ids,
              lists,counts,capacity,xy,conic,feature,b.data_ptr<float>(),out.data_ptr<float>(),key_out);
          else if(render_variant==9)renderChunkedPixels<2,64><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted_ids,
              lists,counts,capacity,xy,conic,feature,b.data_ptr<float>(),out.data_ptr<float>(),key_out);
          else renderTileFused<<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted_ids,
              lists,counts,capacity,xy,conic,feature,b.data_ptr<float>(),out.data_ptr<float>(),key_out,inspect);
        }
      }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if(inspect) {
      std::vector<Tensor> result={out,radii,snapshot(xy,{P,2},at::kFloat,geom_),snapshot(conic,{P,4},at::kFloat,geom_),
          snapshot(depths,{P},at::kFloat,geom_),snapshot(touched,{P},at::kInt,geom_)};
      auto empty=torch::empty({0},xyz.options().dtype(at::kInt));
      result.push_back(sorted_keys?snapshot(sorted_keys,{L},at::kLong,bin_):empty);
      result.push_back(sorted_ids?snapshot(sorted_ids,{mode>=2?P:L},at::kInt,mode>=2?sort_:bin_):empty);
      result.push_back(ranges?snapshot(ranges,{tiles,2},at::kInt,image_):empty);
      result.push_back(bits?snapshot(bits,{tiles,words},at::kInt,bits_):empty);
      result.push_back(mode>=3?snapshot(reinterpret_cast<uint32_t*>(directory_.data_ptr()),{tiles,(words+31)/32},at::kInt,directory_):empty);
      result.push_back(mode==4?snapshot(reinterpret_cast<uint32_t*>(lists_.data_ptr()),{tiles,list_capacity},at::kInt,lists_):empty);
      result.push_back(mode==4?snapshot(reinterpret_cast<uint32_t*>(counts_.data_ptr()),{tiles},at::kInt,counts_):empty);
      result.push_back(snapshot(const_cast<float*>(feature),{P,3},at::kFloat,c.numel()?c:geom_));
      result.push_back(mode>=2?snapshot(key_out,{P},at::kInt,sort_):empty);
      result.push_back(mode>=2?snapshot(bits+size_t(tiles)*words+1,{1},at::kInt,bits_):empty);
      return result;
    }
    return {out,radii};
  }
  int64_t workspace_bytes() const {int64_t n=0;for(const auto& t:{geom_,bin_,image_,bits_,sort_,directory_,lists_,counts_,tasks_,metadata_})if(t.defined())n+=t.storage().nbytes();return n;}
};

std::vector<Tensor> diagnose_alpha(const Tensor& bg,const Tensor& xy,const Tensor& conic,const Tensor& colors,
    const Tensor& bits,const Tensor& sorted,const Tensor& keys,int H,int W) {
  TORCH_CHECK(!torch::GradMode::is_enabled()&&xy.is_cuda()&&H>0&&W>0,"Frozen CUDA diagnostics only");
  c10::cuda::CUDAGuard guard(xy.device());auto stream=c10::cuda::getCurrentCUDAStream();
  int P=xy.size(0);auto out=torch::empty({3,H,W},xy.options());
  auto flags=torch::zeros({P},xy.options().dtype(at::kInt));auto count=torch::zeros({1},xy.options().dtype(at::kLong));
  if(!P){out.zero_();return {out,flags,count};}
  diagnoseAlpha<<<dim3((W+15)/16,(H+15)/16),dim3(16,16),0,stream>>>(W,H,
      reinterpret_cast<const uint32_t*>(bits.data_ptr<int>()),bits.size(1),reinterpret_cast<const uint32_t*>(sorted.data_ptr<int>()),
      reinterpret_cast<const float2*>(xy.data_ptr<float>()),reinterpret_cast<const float4*>(conic.data_ptr<float>()),colors.data_ptr<float>(),
      bg.data_ptr<float>(),reinterpret_cast<const uint32_t*>(keys.data_ptr<int>()),out.data_ptr<float>(),
      reinterpret_cast<uint32_t*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(count.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,flags,count};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("diagnose_alpha",&diagnose_alpha,pybind11::call_guard<pybind11::gil_scoped_release>());
  pybind11::class_<MicrotileStudyRasterizer>(m,"Rasterizer").def(pybind11::init<>())
      .def("render",&MicrotileStudyRasterizer::render,pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("workspace_bytes",&MicrotileStudyRasterizer::workspace_bytes);
}

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include "deployment_device_control.h"

namespace {
constexpr int threads=256;
struct Range {
  unsigned lo,hi;
};
struct MergeRange {
  __device__ Range operator()(Range a,Range b) const {return {min(a.lo,b.lo),max(a.hi,b.hi)};}
};
__device__ unsigned keyOf(float score) {
  unsigned x=score==0.f?0u:__float_as_uint(score);
  return x^(x&0x80000000u?0xffffffffu:0x80000000u);
}

struct Counts {int greater,equal;};
struct SumCounts {
  __device__ Counts operator()(Counts a,Counts b) const {return {a.greater+b.greater,a.equal+b.equal};}
};

// All CTAs are resident together. Temporary candidate order is irrelevant: only
// the exact threshold/rank is returned; final IDs use the original stable order.
__global__ void shrinkingRadix(int n,int k,const float* scores,const int64_t* rows,const bool* eligible,
    unsigned* first,unsigned* second,unsigned* state,const int* status,const DeploymentFrame* frame,
    bool saturated_fast) {
  if(status && *status)return;
  if(frame){n=frame->n;k=frame->k;if(!k)return;}
  if(saturated_fast && frame && k==frame->available) {
    if(!blockIdx.x && !threadIdx.x) {
      state[0]=n;
      state[1]=0;
      state[2]=0;
      state[5]=0;
    }
    return;
  }
  namespace cg=cooperative_groups;
  const auto grid=cg::this_grid();
  using Reduce=cub::BlockReduce<Range,threads>;
  using Scan=cub::BlockScan<int,threads>;
  __shared__ union {typename Reduce::TempStorage reduce;typename Scan::TempStorage scan;} temp;
  __shared__ unsigned hist[256],reservation;
  const int tid=threadIdx.x;
  unsigned* histogram=state+8;
  if(!blockIdx.x && !tid){state[0]=n;state[1]=k;state[2]=0;state[3]=0;state[4]=0;state[5]=0;}
  grid.sync();
  unsigned* input=first;unsigned* output=second;
  for(int pass=0;pass<5;++pass) {
    if(!blockIdx.x) {
      histogram[tid]=0;
      if(!tid){state[6]=~0u;state[7]=0;}
    }
    grid.sync();
    const int length=state[0];Range range{~0u,0};
    for(int i=blockIdx.x*threads+tid;i<length;i+=gridDim.x*threads) {
      unsigned key;
      if(!pass){key=eligible[rows[i]]?keyOf(scores[i]):0u;input[i]=key;}
      else key=input[i];
      if(key){range.lo=min(range.lo,key);range.hi=max(range.hi,key);}
    }
    const Range all=Reduce(temp.reduce).Reduce(range,MergeRange());
    if(!tid){atomicMin(state+6,all.lo);atomicMax(state+7,all.hi);}
    grid.sync();
    const unsigned lo=state[6],hi=state[7];
    if(lo==hi){if(!blockIdx.x && !tid)state[2]=lo;break;}
    // Skip shared leading bits. At most four byte-sized refinements are needed.
    const int top=31-__clz(lo^hi),shift=max(0,top-7);
    hist[tid]=0;__syncthreads();
    for(int base=blockIdx.x*threads;base<length;base+=gridDim.x*threads) {
      const int i=base+tid;const unsigned key=i<length?input[i]:0u;
      const unsigned digit=key?(key>>shift)&255u:256u;
#if __CUDA_ARCH__ >= 700
      const unsigned group=__match_any_sync(0xffffffffu,digit);
      if(key && (tid&31)==__ffs(group)-1)atomicAdd(hist+digit,__popc(group));
#else
      if(key)atomicAdd(hist+digit,1u);
#endif
    }
    __syncthreads();
    if(hist[tid])atomicAdd(histogram+tid,hist[tid]);
    grid.sync();
    if(!blockIdx.x) {
      const int count=histogram[255-tid];int before;
      const int wanted=state[1];
      Scan(temp.scan).ExclusiveSum(count,before);__syncthreads();
      if(wanted>before && wanted<=before+count) {
        const unsigned upper=shift<24?(lo&(~0u<<(shift+8))):0u;
        state[2]=upper|(unsigned(255-tid)<<shift);state[1]=wanted-before;
        state[3]=shift;state[4]=0;state[5]=pass+1;
      }
    }
    grid.sync();
    if(!shift)break;
    const unsigned prefix=state[2],mask=~0u<<shift;
    for(int base=blockIdx.x*threads;base<length;base+=gridDim.x*threads) {
      const int i=base+tid;const unsigned key=i<length?input[i]:0u;
      const int keep=key && (key&mask)==prefix;int before,total;
      Scan(temp.scan).ExclusiveSum(keep,before,total);__syncthreads();
      if(!tid)reservation=atomicAdd(state+4,unsigned(total));
      __syncthreads();
      if(keep)output[reservation+before]=key;
      __syncthreads();
    }
    grid.sync();
    if(!blockIdx.x && !tid)state[0]=state[4];
    grid.sync();
    auto swap=input;input=output;output=swap;
  }
}

__global__ void shrinkingRadixFused(int n,int k,const float* scores,const int64_t* rows,const bool* eligible,
    unsigned* first,unsigned* second,unsigned* state,const int* status,const DeploymentFrame* frame,
    bool saturated_fast,Counts* counter,int64_t* selected) {
  if(status && *status)return;
  if(frame){n=frame->n;k=frame->k;if(!k)return;}
  namespace cg=cooperative_groups;
  const auto grid=cg::this_grid();
  using Reduce=cub::BlockReduce<Range,threads>;
  using Scan=cub::BlockScan<int,threads>;
  __shared__ union {typename Reduce::TempStorage reduce;typename Scan::TempStorage scan;} temp;
  __shared__ unsigned hist[256],reservation;
  const int tid=threadIdx.x;
  unsigned* histogram=state+8;
  const bool saturated=saturated_fast && frame && k==frame->available;
  if(!blockIdx.x && !tid) {
    state[0]=n;state[1]=saturated?0:k;state[2]=0;state[3]=0;state[4]=0;state[5]=0;
  }
  grid.sync();
  if(!saturated) {
    unsigned* input=first;unsigned* output=second;
    for(int pass=0;pass<5;++pass) {
      if(!blockIdx.x) {
        histogram[tid]=0;
        if(!tid){state[6]=~0u;state[7]=0;}
      }
      grid.sync();
      const int length=state[0];Range range{~0u,0};
      for(int i=blockIdx.x*threads+tid;i<length;i+=gridDim.x*threads) {
        unsigned key;
        if(!pass){key=eligible[rows[i]]?keyOf(scores[i]):0u;input[i]=key;}
        else key=input[i];
        if(key){range.lo=min(range.lo,key);range.hi=max(range.hi,key);}
      }
      const Range all=Reduce(temp.reduce).Reduce(range,MergeRange());
      if(!tid){atomicMin(state+6,all.lo);atomicMax(state+7,all.hi);}
      grid.sync();
      const unsigned lo=state[6],hi=state[7];
      if(lo==hi){if(!blockIdx.x && !tid)state[2]=lo;break;}
      // Skip shared leading bits. At most four byte-sized refinements are needed.
      const int top=31-__clz(lo^hi),shift=max(0,top-7);
      hist[tid]=0;__syncthreads();
      for(int base=blockIdx.x*threads;base<length;base+=gridDim.x*threads) {
        const int i=base+tid;const unsigned key=i<length?input[i]:0u;
        const unsigned digit=key?(key>>shift)&255u:256u;
  #if __CUDA_ARCH__ >= 700
        const unsigned group=__match_any_sync(0xffffffffu,digit);
        if(key && (tid&31)==__ffs(group)-1)atomicAdd(hist+digit,__popc(group));
  #else
        if(key)atomicAdd(hist+digit,1u);
  #endif
      }
      __syncthreads();
      if(hist[tid])atomicAdd(histogram+tid,hist[tid]);
      grid.sync();
      if(!blockIdx.x) {
        const int count=histogram[255-tid];int before;
        const int wanted=state[1];
        Scan(temp.scan).ExclusiveSum(count,before);__syncthreads();
        if(wanted>before && wanted<=before+count) {
          const unsigned upper=shift<24?(lo&(~0u<<(shift+8))):0u;
          state[2]=upper|(unsigned(255-tid)<<shift);state[1]=wanted-before;
          state[3]=shift;state[4]=0;state[5]=pass+1;
        }
      }
      grid.sync();
      if(!shift)break;
      const unsigned prefix=state[2],mask=~0u<<shift;
      for(int base=blockIdx.x*threads;base<length;base+=gridDim.x*threads) {
        const int i=base+tid;const unsigned key=i<length?input[i]:0u;
        const int keep=key && (key&mask)==prefix;int before,total;
        Scan(temp.scan).ExclusiveSum(keep,before,total);__syncthreads();
        if(!tid)reservation=atomicAdd(state+4,unsigned(total));
        __syncthreads();
        if(keep)output[reservation+before]=key;
        __syncthreads();
      }
      grid.sync();
      if(!blockIdx.x && !tid)state[0]=state[4];
      grid.sync();
      auto swap=input;input=output;output=swap;
    }

  }
  grid.sync();
  // Each CTA owns a contiguous range, preserving the original row order.
  const int chunk=((n+int(gridDim.x)*threads-1)/(int(gridDim.x)*threads))*threads;
  const int begin=int(blockIdx.x)*chunk,end=min(n,begin+chunk);
  Counts count{0,0};
  for(int i=begin+tid;i<end;i+=threads) {
    const bool valid=eligible[rows[i]];
    const unsigned key=valid?keyOf(scores[i]):0u;
    count=SumCounts()(count,Counts{int(valid && key>state[2]),int(valid && key==state[2])});
  }
  using CountReduce=cub::BlockReduce<Counts,threads>;
  using CountScan=cub::BlockScan<Counts,threads>;
  __shared__ union {typename CountReduce::TempStorage reduce;typename CountScan::TempStorage scan;} count_temp;
  __shared__ Counts before;
  count=CountReduce(count_temp.reduce).Reduce(count,SumCounts());
  if(!tid)counter[blockIdx.x]=count;
  grid.sync();
  count={0,0};
  for(int i=tid;i<blockIdx.x;i+=threads)count=SumCounts()(count,counter[i]);
  count=CountReduce(count_temp.reduce).Reduce(count,SumCounts());
  if(!tid)before=count;
  __syncthreads();
  for(int base=begin;base<end;base+=threads) {
    const int i=base+tid;
    const bool valid=i<end && eligible[rows[i]];
    const unsigned key=valid?keyOf(scores[i]):0u;
    const Counts value{int(valid && key>state[2]),int(valid && key==state[2])};
    Counts local,total;
    CountScan(count_temp.scan).ExclusiveScan(value,local,Counts{0,0},SumCounts(),total);
    const int equal=before.equal+local.equal,ties=state[1];
    if(value.greater || (value.equal && equal<ties))
      selected[before.greater+local.greater+min(equal,ties)]=rows[i];
    __syncthreads();
    if(!tid)before=SumCounts()(before,total);
    __syncthreads();
  }
}

__global__ void countThreshold(int n,const float* scores,const int64_t* rows,const bool* eligible,
    const unsigned* state,Counts* counts,const int* status,const DeploymentFrame* frame) {
  if(status && *status)return;
  if(frame){n=frame->n;if(!frame->k)return;}
  using Reduce=cub::BlockReduce<Counts,threads>;__shared__ typename Reduce::TempStorage temp;
  const int i=blockIdx.x*threads+threadIdx.x;
  const bool valid=i<n && eligible[rows[i]];
  const unsigned key=valid?keyOf(scores[i]):0u,threshold=state[2];
  const auto sum=Reduce(temp).Reduce(Counts{int(valid && key>threshold),int(valid && key==threshold)},SumCounts());
  if(!threadIdx.x)counts[blockIdx.x]=sum;
}
__global__ void emitSelected(int n,const float* scores,const int64_t* rows,const bool* eligible,
    const unsigned* state,const Counts* counts,int64_t* output,const int* status,const DeploymentFrame* frame) {
  if(status && *status)return;
  if(frame){n=frame->n;if(!frame->k || int64_t(blockIdx.x)*threads>=n)return;}
  using Reduce=cub::BlockReduce<Counts,threads>;using Scan=cub::BlockScan<Counts,threads>;
  __shared__ union {typename Reduce::TempStorage reduce;typename Scan::TempStorage scan;} temp;
  __shared__ Counts before;
  Counts sum{0,0};
  for(int i=threadIdx.x;i<blockIdx.x;i+=threads)sum=SumCounts()(sum,counts[i]);
  sum=Reduce(temp.reduce).Reduce(sum,SumCounts());
  if(!threadIdx.x)before=sum;
  __syncthreads();
  const int i=blockIdx.x*threads+threadIdx.x;
  const bool valid=i<n && eligible[rows[i]];
  const unsigned key=valid?keyOf(scores[i]):0u;
  const Counts value{int(valid && key>state[2]),int(valid && key==state[2])};Counts local;
  Scan(temp.scan).ExclusiveScan(value,local,Counts{0,0},SumCounts());
  const int equal=before.equal+local.equal,ties=state[1];
  if(value.greater || (value.equal && equal<ties))output[before.greater+local.greater+min(equal,ties)]=rows[i];
}

void launchRadix(const torch::Tensor& scores,const torch::Tensor& rows,const torch::Tensor& eligible,
    int64_t k,const std::vector<torch::Tensor>& scratch,const torch::Tensor& status,
    const torch::Tensor& output,const DeploymentFrame* frame,bool saturated_fast,bool fused=false) {
  const int n=scores.numel();
  TORCH_CHECK(!torch::GradMode::is_enabled() && k>=0 && k<=n && scratch.size()==4,"Shrinking Radix request");
  c10::cuda::CUDAGuard guard(scores.device());if(!n)return;
  TORCH_CHECK(scores.scalar_type()==at::kFloat && scores.is_contiguous() && rows.numel()==n &&
      rows.scalar_type()==at::kLong && eligible.scalar_type()==at::kBool && scratch[0].numel()>=n &&
      scratch[1].numel()>=n && scratch[2].numel()>=264 && scratch[3].numel()>=2*((n+255)/256),"Shrinking Radix layout");
  // Cache immutable launch metadata only, never scores, membership or ratio data.
  static thread_local int cached_device=-1,sm_count=0,blocks_per_sm=0;
  int device=0;C10_CUDA_CHECK(cudaGetDevice(&device));
  if(device!=cached_device) {
    cudaDeviceProp prop{};C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
    TORCH_CHECK(prop.cooperativeLaunch,"Cooperative launch unavailable");
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm,shrinkingRadix,threads,0));
    sm_count=prop.multiProcessorCount;cached_device=device;
  }
  const int blocks=std::min((n+255)/256,std::min(sm_count,blocks_per_sm*sm_count));
  const float* score_ptr=scores.data_ptr<float>();const int64_t* row_ptr=rows.data_ptr<int64_t>();
  const bool* eligible_ptr=eligible.data_ptr<bool>();
  auto first=reinterpret_cast<unsigned*>(scratch[0].data_ptr<int>()),second=reinterpret_cast<unsigned*>(scratch[1].data_ptr<int>()),
      state=reinterpret_cast<unsigned*>(scratch[2].data_ptr<int>());
  const int* failed=frame?&frame->status:(status.defined()?status.data_ptr<int>():nullptr);int count=k;
  void* args[]={const_cast<int*>(&n),&count,&score_ptr,&row_ptr,&eligible_ptr,&first,&second,&state,&failed,&frame,&saturated_fast};
  const auto stream=c10::cuda::getCurrentCUDAStream();
  auto counts=reinterpret_cast<Counts*>(scratch[3].data_ptr<int>());
  if(fused) {
    auto selected=output.data_ptr<int64_t>();
    void* fused_args[]={const_cast<int*>(&n),&count,&score_ptr,&row_ptr,&eligible_ptr,&first,&second,&state,&failed,&frame,&saturated_fast,&counts,&selected};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(shrinkingRadixFused),blocks,threads,fused_args,0,stream));
  } else {
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(shrinkingRadix),blocks,threads,args,0,stream));
    countThreshold<<<(n+255)/256,threads,0,stream>>>(n,score_ptr,row_ptr,eligible_ptr,state,counts,failed,frame);
    emitSelected<<<(n+255)/256,threads,0,stream>>>(n,score_ptr,row_ptr,eligible_ptr,state,counts,output.data_ptr<int64_t>(),failed,frame);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}

torch::Tensor deployment_radix_select(const torch::Tensor& scores,const torch::Tensor& rows,
    const torch::Tensor& eligible,int64_t k,const std::vector<torch::Tensor>& scratch,const torch::Tensor& status) {
  auto output=torch::empty({k},rows.options());if(!k)return output;
  launchRadix(scores,rows,eligible,k,scratch,status,output,nullptr,false);return output;
}

void deployment_radix_device(const torch::Tensor& scores,const torch::Tensor& rows,const torch::Tensor& eligible,
    const torch::Tensor& output,const torch::Tensor& control,const std::vector<torch::Tensor>& scratch,
    bool saturated_fast,bool fused) {
  launchRadix(scores,rows,eligible,0,scratch,torch::Tensor(),output,
      reinterpret_cast<const DeploymentFrame*>(control.data_ptr<uint8_t>()),saturated_fast,fused);
}

torch::Tensor deployment_radix_fused(const torch::Tensor& scores,const torch::Tensor& rows,
    const torch::Tensor& eligible,int64_t k,const std::vector<torch::Tensor>& scratch,const torch::Tensor& status) {
  auto output=torch::empty({k},rows.options());if(!k)return output;
  launchRadix(scores,rows,eligible,k,scratch,status,output,nullptr,false,true);return output;
}

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

torch::Tensor exact_select(const torch::Tensor&,const torch::Tensor&,int64_t,
    const std::vector<torch::Tensor>&,bool);

namespace {
constexpr int threads=256;
struct AddPair {
  __host__ __device__ int2 operator()(int2 a,int2 b) const {return make_int2(a.x+b.x,a.y+b.y);}
};

// Match CUB float ordering, including equal signed zero. The caller has checked finite scores.
__device__ unsigned scoreKey(float score) {
  unsigned bits=score==0.f?0u:__float_as_uint(score);
  return bits^(bits&0x80000000u?0xffffffffu:0x80000000u);
}

// Three exact radix passes find the Kth key and the number of ties to retain.
// A single CTA avoids per-pass launches and never reorders Gaussian rows.
template<int Items> __global__ void selectThreshold(int n,int k,const float* scores,const int64_t* rows,
    const bool* eligible,unsigned* result,const int* status) {
  if(status && status[0]==1)return;
  using Scan=cub::BlockScan<int,threads>;
  __shared__ typename Scan::TempStorage temp;
  __shared__ unsigned histogram[2048],prefix,mask;
  __shared__ int rank;
  const int tid=threadIdx.x;
  unsigned keys[Items];
  #pragma unroll
  for(int j=0;j<Items;++j) {
    const int i=j*threads+tid;
    keys[j]=i<n && eligible[rows[i]]?scoreKey(scores[i]):0u;
  }
  if(!tid){prefix=0;mask=0;rank=k;}
  __syncthreads();
  for(int shift=22;shift>=0;shift-=11) {
    #pragma unroll
    for(int j=0;j<8;++j)histogram[tid+j*threads]=0;
    __syncthreads();
    #pragma unroll
    for(int j=0;j<Items;++j) {
      if(j*threads<n) {
      const unsigned key=keys[j];
      const bool active=key && (key&mask)==prefix;
      const unsigned digit=active?(key>>shift)&2047u:2048u;
#if __CUDA_ARCH__ >= 700
      const unsigned group=__match_any_sync(0xffffffffu,digit);
      if(active && (tid&31)==__ffs(group)-1)atomicAdd(histogram+digit,__popc(group));
#else
      if(active)atomicAdd(histogram+digit,1u);
#endif
      }
    }
    __syncthreads();
    int sum=0;
    #pragma unroll
    for(int j=0;j<8;++j)sum+=histogram[2047-tid*8-j];
    int before;const int wanted=rank;
    Scan(temp).ExclusiveSum(sum,before);
    __syncthreads();
    if(wanted>before && wanted<=before+sum) {
      int remaining=wanted-before;
      for(int j=0;j<8;++j) {
        const int digit=2047-tid*8-j,count=histogram[digit];
        if(remaining>count)remaining-=count;
        else{prefix|=unsigned(digit)<<shift;rank=remaining;break;}
      }
    }
    if(!tid)mask|=2047u<<shift;
    __syncthreads();
  }
  if(!tid){result[0]=prefix;result[1]=rank;}
}

__global__ void thresholdCounts(int n,const float* scores,const int64_t* rows,
    const bool* eligible,const unsigned* threshold,int2* counts,const int* status) {
  if(status && status[0]==1)return;
  using Reduce=cub::BlockReduce<int2,threads>;
  __shared__ typename Reduce::TempStorage temp;
  const int i=blockIdx.x*threads+threadIdx.x;
  const bool valid=i<n && eligible[rows[i]];
  const unsigned key=valid?scoreKey(scores[i]):0u;
  const int2 value=make_int2(valid && key>threshold[0],valid && key==threshold[0]);
  const int2 sum=Reduce(temp).Reduce(value,AddPair());
  if(!threadIdx.x)counts[blockIdx.x]=sum;
}

__global__ void compactSelected(int n,const float* scores,const int64_t* rows,
    const bool* eligible,const unsigned* threshold,const int2* counts,int64_t* output,const int* status) {
  if(status && status[0]==1)return;
  using Reduce=cub::BlockReduce<int2,threads>;
  using Scan=cub::BlockScan<int2,threads>;
  __shared__ union {typename Reduce::TempStorage reduce;typename Scan::TempStorage scan;} temp;
  __shared__ int2 before;
  int2 sum=make_int2(0,0);
  for(int j=threadIdx.x;j<blockIdx.x;j+=threads)sum=AddPair()(sum,counts[j]);
  sum=Reduce(temp.reduce).Reduce(sum,AddPair());
  if(!threadIdx.x)before=sum;
  __syncthreads();
  const int i=blockIdx.x*threads+threadIdx.x;
  const bool valid=i<n && eligible[rows[i]];
  const unsigned key=valid?scoreKey(scores[i]):0u;
  const int2 value=make_int2(valid && key>threshold[0],valid && key==threshold[0]);
  int2 local;
  Scan(temp.scan).ExclusiveScan(value,local,make_int2(0,0),AddPair());
  const int equal_before=before.y+local.y,keep_equal=threshold[1];
  if(value.x || (value.y && equal_before<keep_equal)) {
    const int position=before.x+local.x+min(equal_before,keep_equal);
    output[position]=rows[i];
  }
}
}

torch::Tensor deployment_select_checked(const torch::Tensor& scores,const torch::Tensor& rows,
    const torch::Tensor& eligible,int64_t k,const torch::Tensor& scratch,
    const std::vector<torch::Tensor>& sort_workspace,const torch::Tensor& status) {
  const auto n=scores.numel();
  TORCH_CHECK(!torch::GradMode::is_enabled() && scores.is_cuda() && scores.is_contiguous() &&
      scores.scalar_type()==at::kFloat && rows.is_contiguous() && rows.scalar_type()==at::kLong &&
      eligible.is_contiguous() && eligible.scalar_type()==at::kBool && rows.numel()==n &&
      rows.device()==scores.device() && eligible.device()==scores.device() && k>=0 && k<=n,"Deployment selection layout");
  c10::cuda::CUDAGuard guard(scores.device());
  if(n>16384) {
    // Large maps retain CUB's scalable exact implementation. No Python scheduling is used.
    auto selected=exact_select(scores,eligible.index_select(0,rows),k,sort_workspace,true);
    return rows.index_select(0,selected);
  }
  auto output=torch::empty({k},rows.options());
  if(!k)return output;
  const int blocks=(n+threads-1)/threads;
  TORCH_CHECK(scratch.device()==scores.device() && scratch.scalar_type()==at::kInt &&
      scratch.is_contiguous() && scratch.numel()>=2+2*blocks,"Deployment selection workspace");
  auto threshold=reinterpret_cast<unsigned*>(scratch.data_ptr<int>());
  auto counts=reinterpret_cast<int2*>(scratch.data_ptr<int>()+2);
  const int* failed=status.defined()?status.data_ptr<int>():nullptr;
  const auto stream=c10::cuda::getCurrentCUDAStream();
  if(n<=4096)selectThreshold<16><<<1,threads,0,stream>>>(n,k,scores.data_ptr<float>(),rows.data_ptr<int64_t>(),eligible.data_ptr<bool>(),threshold,failed);
  else selectThreshold<64><<<1,threads,0,stream>>>(n,k,scores.data_ptr<float>(),rows.data_ptr<int64_t>(),eligible.data_ptr<bool>(),threshold,failed);
  thresholdCounts<<<blocks,threads,0,stream>>>(n,scores.data_ptr<float>(),rows.data_ptr<int64_t>(),eligible.data_ptr<bool>(),threshold,counts,failed);
  compactSelected<<<blocks,threads,0,stream>>>(n,scores.data_ptr<float>(),rows.data_ptr<int64_t>(),eligible.data_ptr<bool>(),threshold,counts,output.data_ptr<int64_t>() ,failed);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

torch::Tensor deployment_select(const torch::Tensor& scores,const torch::Tensor& rows,
    const torch::Tensor& eligible,int64_t k,const torch::Tensor& scratch,
    const std::vector<torch::Tensor>& sort_workspace) {
  return deployment_select_checked(scores,rows,eligible,k,scratch,sort_workspace,torch::Tensor());
}

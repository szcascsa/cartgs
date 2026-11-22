#include "include/anchor_si.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

namespace anchor_si {
namespace {
__device__ bool allowed(int64_t i,const float* opacity,int os,const bool* inactive) {
  return !inactive || !(inactive[i] && opacity[i*os]<1.f/255.f);
}
__global__ void meanKernel(int64_t a,const float* values,int stride,const int64_t* members,
    const int64_t* counts,const float* opacity,int os,const bool* inactive,float* mean,float* count,float* activity) {
  int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=a)return;
  float sum=0,n=0;int bits=0;for(int j=0;j<counts[row];++j){auto i=members[row*10+j];if(allowed(i,opacity,os,inactive)){sum+=values[i*stride];n+=1;bits|=1<<j;}}
  mean[row]=n?sum/n:0;count[row]=n;activity[row]=float(bits);
}
__global__ void meanVJP(int64_t n,const int64_t* owner,const float* count,
    const float* opacity,int os,const bool* inactive,const float* gradient,float* output) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  auto a=owner[i];output[i]=allowed(i,opacity,os,inactive)&&count[a]>0?gradient[a]/count[a]:0;
}
__device__ float blockSum(float value,float* scratch) {
  scratch[threadIdx.x]=value;__syncthreads();
  for(int s=128;s;s>>=1){if(threadIdx.x<s)scratch[threadIdx.x]+=scratch[threadIdx.x+s];__syncthreads();}
  return scratch[0];
}
__global__ void targetKernel(int64_t a,const float* ema,const int64_t* visits,const int64_t* last,
    const float* active,const float* ratios,int64_t clock,int64_t n,int min_visits,int max_age,float temperature,float* output) {
  __shared__ float scratch[256];float total=0,valid_total=0;
  for(int64_t i=threadIdx.x;i<a;i+=256){total+=active[i];if(visits[i]>=min_visits&&clock-last[i]<=max_age&&active[i]>0)valid_total+=active[i];}
  total=blockSum(total,scratch);__syncthreads();valid_total=blockSum(valid_total,scratch);__syncthreads();
  const float rho=total>0?ratios[blockIdx.x]*float(n)/total:2.f;
  float low=-80,high=80;
  for(int step=0;step<32;++step){float mass=0;const float mid=(low+high)*.5f;
    for(int64_t i=threadIdx.x;i<a;i+=256)if(visits[i]>=min_visits&&clock-last[i]<=max_age&&active[i]>0)
      mass+=active[i]/(1.f+expf(-(logf(fmaxf(ema[i],1e-12f))-mid)/temperature));
    mass=blockSum(mass,scratch);__syncthreads();if(mass>rho*valid_total)low=mid;else high=mid;
  }
  for(int64_t i=threadIdx.x;i<a;i+=256)output[blockIdx.x*a+i]=rho>1||rho<0||valid_total==0?-1.f:
    1.f/(1.f+expf(-(logf(fmaxf(ema[i],1e-12f))-(low+high)*.5f)/temperature));
}
void check(const std::vector<torch::Tensor>& input) {
  TORCH_CHECK(input[0].is_cuda()&&input[0].scalar_type()==torch::kFloat32&&input[0].dim()==1,"Anchor SI values require CUDA FP32 rows");
  const auto device=input[0].device();
  for(int j:{1,2,3})TORCH_CHECK(input[j].device()==device&&input[j].scalar_type()==torch::kLong&&input[j].is_contiguous(),"Anchor SI indices require contiguous CUDA Int64");
  TORCH_CHECK(input[1].numel()==input[0].numel()&&input[2].sizes()==torch::IntArrayRef({input[3].numel(),10}),"Anchor SI layout mismatch");
  TORCH_CHECK(input[4].device()==device&&input[4].scalar_type()==torch::kFloat32&&input[4].numel()==input[0].numel(),"Anchor SI opacity layout mismatch");
  TORCH_CHECK(!input[5].defined()||!input[5].numel()||(input[5].device()==device&&input[5].scalar_type()==torch::kBool&&input[5].is_contiguous()&&input[5].numel()==input[0].numel()),"Anchor SI inactive layout mismatch");
}
}
std::vector<torch::Tensor> reduceForward(const std::vector<torch::Tensor>& x) {
  check(x);c10::cuda::CUDAGuard guard(x[0].device());auto a=x[3].numel();
  auto out=torch::empty({a},x[0].options()),count=torch::empty_like(out),activity=torch::empty_like(out);
  if(a)meanKernel<<<(a+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(a,x[0].data_ptr<float>(),x[0].stride(0),x[2].data_ptr<int64_t>(),x[3].data_ptr<int64_t>(),x[4].data_ptr<float>(),x[4].stride(0),x[5].defined()&&x[5].numel()?x[5].data_ptr<bool>():nullptr,out.data_ptr<float>(),count.data_ptr<float>(),activity.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,count,activity};
}
torch::Tensor reduceBackward(const std::vector<torch::Tensor>& x,const torch::Tensor& count,const torch::Tensor& gradient) {
  c10::cuda::CUDAGuard guard(x[0].device());auto out=torch::empty(x[0].sizes(),x[0].options());auto n=out.numel();
  if(n)meanVJP<<<(n+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,x[1].data_ptr<int64_t>(),count.data_ptr<float>(),x[4].data_ptr<float>(),x[4].stride(0),x[5].defined()&&x[5].numel()?x[5].data_ptr<bool>():nullptr,gradient.data_ptr<float>(),out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor targets(const torch::Tensor& ema,const torch::Tensor& visits,const torch::Tensor& last,
    const torch::Tensor& active,const torch::Tensor& ratios,int64_t clock,int64_t n,int64_t min_visits,int64_t max_age,float temperature) {
  c10::cuda::CUDAGuard guard(ema.device());auto out=torch::empty({ratios.numel(),ema.numel()},ema.options());
  if(ema.numel())targetKernel<<<ratios.numel(),256,0,c10::cuda::getCurrentCUDAStream()>>>(ema.numel(),ema.data_ptr<float>(),visits.data_ptr<int64_t>(),last.data_ptr<int64_t>(),active.data_ptr<float>(),ratios.data_ptr<float>(),clock,n,min_visits,max_age,temperature,out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
}

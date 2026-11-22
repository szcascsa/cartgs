#include "include/selector_sampling_cuda.h"
#include <cub/cub.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>
namespace mapping_cuda {
namespace {
__global__ void sample(int64_t n,const float* p,int64_t ps,const float* u,int64_t us,const float* opacity,int64_t os,const bool* inactive,
    float* effective,float* mask,bool* eligible,bool* flags){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  const bool allowed=!inactive||!(inactive[i]&&opacity[i*os]<1.f/255.f);eligible[i]=allowed;
  float value=p[i*ps];if(inactive){value*=float(allowed);effective[i]=value;}
  const float hard=u[i*us]<value?1.f:0.f;mask[i]=hard+(value-value);flags[i]=mask[i]>0.f;
}
__global__ void backward(int64_t n,const bool* eligible,const float* gm,const float* gp,float* out){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)out[i]=((gm?gm[i]:0.f)+(gp?gp[i]:0.f))*float(eligible[i]);
}
}
std::vector<torch::Tensor> selectorSamplingForward(const torch::Tensor& p,const torch::Tensor& u,const torch::Tensor& opacity,const torch::Tensor& inactive){
  torch::NoGradGuard guard;TORCH_CHECK(p.is_cuda()&&p.scalar_type()==torch::kFloat32&&p.dim()==1&&u.sizes()==p.sizes()&&u.device()==p.device()&&u.scalar_type()==torch::kFloat32,"Sampling requires row aligned CUDA FP32");
  c10::cuda::CUDAGuard device(p.device());const auto n=p.numel();TORCH_CHECK(n<=INT_MAX,"CUB compaction row count exceeds Int32");
  const bool slots=inactive.defined()&&inactive.numel()>0;
  if(slots)TORCH_CHECK(inactive.is_cuda()&&inactive.device()==p.device()&&inactive.scalar_type()==torch::kBool&&inactive.is_contiguous()&&inactive.numel()==n&&opacity.is_cuda()&&opacity.device()==p.device()&&opacity.scalar_type()==torch::kFloat32&&opacity.numel()==n&&(opacity.dim()==1||(opacity.dim()==2&&opacity.size(1)==1)),"Sampling slot layout mismatch");
  auto effective=slots?torch::empty({n},p.options()):p;
  auto mask=torch::empty({n},p.options()),eligible=torch::empty({n},p.options().dtype(torch::kBool)),flags=torch::empty_like(eligible);
  auto ids=torch::empty({n},p.options().dtype(torch::kLong));if(!n)return {effective,mask,ids,eligible};
  auto count=torch::empty({},p.options().dtype(torch::kInt32));auto stream=c10::cuda::getCurrentCUDAStream();
  sample<<<(n+255)/256,256,0,stream>>>(n,p.data_ptr<float>(),p.stride(0),u.data_ptr<float>(),u.stride(0),slots?opacity.data_ptr<float>():nullptr,
      slots?opacity.stride(0):0,slots?inactive.data_ptr<bool>():nullptr,effective.data_ptr<float>(),mask.data_ptr<float>(),eligible.data_ptr<bool>(),flags.data_ptr<bool>());
  size_t bytes=0;cub::CountingInputIterator<int64_t> rows(0);
  C10_CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr,bytes,rows,flags.data_ptr<bool>(),ids.data_ptr<int64_t>(),count.data_ptr<int>(),n,stream));
  auto temp=torch::empty({static_cast<int64_t>(bytes)},p.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceSelect::Flagged(temp.data_ptr(),bytes,rows,flags.data_ptr<bool>(),ids.data_ptr<int64_t>(),count.data_ptr<int>(),n,stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();const int k=count.item<int>(); // One required dynamic-shape synchronization.
  return {effective,mask,ids.slice(0,0,k),eligible};
}
torch::Tensor selectorSamplingBackward(const torch::Tensor& eligible,const torch::Tensor& gm,const torch::Tensor& gp){
  if(!gm.defined()&&!gp.defined())return {};
  c10::cuda::CUDAGuard device(eligible.device());auto out=torch::empty(eligible.sizes(),eligible.options().dtype(torch::kFloat32));const auto n=eligible.numel();
  if(n)backward<<<(n+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,eligible.data_ptr<bool>(),gm.defined()?gm.data_ptr<float>():nullptr,gp.defined()?gp.data_ptr<float>():nullptr,out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
}

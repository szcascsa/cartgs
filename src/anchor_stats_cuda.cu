#include "include/anchor_stats_cuda.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>

namespace mapping_cuda {
namespace {
__global__ void init(int64_t a,int* counts,long long* error){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<a)counts[i]=0;if(i<6)error[i]=LLONG_MAX;
}
__device__ void fail(long long* error,int kind,int64_t index){atomicMin(reinterpret_cast<unsigned long long*>(error+kind),static_cast<unsigned long long>(index));}
__global__ void owners(int64_t n,int64_t a,const int64_t* owner,int* counts,int64_t* members,long long* error){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  int64_t id=owner[i];if(id<0||id>=a){fail(error,0,i);return;}
  int slot=atomicAdd(counts+id,1);if(slot<10)members[id*10+slot]=i;
}
__global__ void finish(int64_t a,const int* actual,const int64_t* counts,const int64_t* levels,
    const float* centers,const float* scales,int64_t* members,long long* error){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=a)return;
  int count=actual[i];
  if(count<1||count>10||counts[i]<1||counts[i]>10)fail(error,1,i);
  if(count!=counts[i])fail(error,2,i);
  if(levels[i]<0||levels[i]>2)fail(error,3,i);
  if(!isfinite(centers[3*i])||!isfinite(centers[3*i+1])||!isfinite(centers[3*i+2]))fail(error,4,i);
  if(!isfinite(scales[i])||scales[i]<=0)fail(error,5,i);
  if(count<=10)for(int j=1;j<count;++j){int64_t x=members[10*i+j];int k=j;
    while(k&&members[10*i+k-1]>x){members[10*i+k]=members[10*i+k-1];--k;}members[10*i+k]=x;}
}
__global__ void visibility(int64_t v,const int64_t* visible,const int64_t* owner,int* seen){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<v)atomicExch(seen+owner[visible[i]],1);
}
__global__ void stats(int64_t a,const int64_t* members,const int64_t* counts,const int* seen,const float* opacity,
    double* observed_values,int64_t* visits,double* slots,int64_t* low,int64_t* cutoff,bool* inactive,bool* recovered){
  int64_t id=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=a)return;
  const bool observed=seen[id];if(observed)++visits[id];
  // Each slot is visited once; each visible Anchor increments its count once.
  for(int j=0;j<counts[id];++j){const int64_t i=members[10*id+j];const float alpha=opacity[i];
    if(inactive){bool recover=inactive[i]&&!(alpha<1.f/255.f);recovered[i]=recovered[i]||recover;inactive[i]=inactive[i]&&!recover;}
    const double value=static_cast<double>(alpha);observed_values[i]=observed?value:0.;
    if(observed){slots[i]+=value;
      low[i]+=value<.02;cutoff[i]+=value<1./255.;}
  }
}
void fp32(const torch::Tensor& t,const torch::Device& device){TORCH_CHECK(t.is_cuda()&&t.device()==device&&t.scalar_type()==torch::kFloat32&&t.is_contiguous(),"Anchor fusion needs contiguous CUDA FP32");}
void index(const torch::Tensor& t,const torch::Device& device){TORCH_CHECK(t.is_cuda()&&t.device()==device&&t.scalar_type()==torch::kLong&&t.is_contiguous(),"Anchor fusion needs contiguous CUDA Int64");}
}
torch::Tensor validateAnchors(const torch::Tensor& owner,const torch::Tensor& counts,const torch::Tensor& levels,
    const torch::Tensor& centers,const torch::Tensor& scales){
  torch::NoGradGuard no_grad;c10::cuda::CUDAGuard guard(owner.device());
  const auto a=counts.numel(),n=owner.numel();index(owner,owner.device());index(counts,owner.device());index(levels,owner.device());fp32(centers,owner.device());fp32(scales,owner.device());
  TORCH_CHECK(levels.numel()==a&&centers.sizes()==torch::IntArrayRef({a,3})&&scales.numel()==a,"Invalid variable Anchor buffer shape");
  auto members=torch::empty({a,10},owner.options()),actual=torch::empty({a},owner.options().dtype(torch::kInt32));
  auto error=torch::empty({6},owner.options());const auto stream=c10::cuda::getCurrentCUDAStream();
  init<<<(std::max<int64_t>(a,6)+127)/128,128,0,stream>>>(a,actual.data_ptr<int>(),reinterpret_cast<long long*>(error.data_ptr<int64_t>()));
  if(n)owners<<<(n+127)/128,128,0,stream>>>(n,a,owner.data_ptr<int64_t>(),actual.data_ptr<int>(),members.data_ptr<int64_t>(),reinterpret_cast<long long*>(error.data_ptr<int64_t>()));
  if(a)finish<<<(a+127)/128,128,0,stream>>>(a,actual.data_ptr<int>(),counts.data_ptr<int64_t>(),levels.data_ptr<int64_t>(),centers.data_ptr<float>(),scales.data_ptr<float>(),members.data_ptr<int64_t>(),reinterpret_cast<long long*>(error.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();auto cpu=error.cpu();const char* names[]={"invalid owner/Gaussian row","capacity/Anchor","member count mismatch/Anchor","level/Anchor","center nonfinite/Anchor","scale invalid/Anchor"};
  for(int i=0;i<6;++i)TORCH_CHECK(cpu.data_ptr<int64_t>()[i]==LLONG_MAX,"Anchor validation: ",names[i]," index=",cpu.data_ptr<int64_t>()[i]);
  return members;
}
void anchorFullStats(const torch::Tensor& owner,const torch::Tensor& members,const torch::Tensor& counts,
    const torch::Tensor& visible,const torch::Tensor& opacity,const std::vector<torch::Tensor>& s,
    const torch::Tensor& inactive,const torch::Tensor& recovered){
  torch::NoGradGuard no_grad;c10::cuda::CUDAGuard guard(owner.device());
  const auto a=counts.numel(),n=owner.numel();TORCH_CHECK(s.size()==5,"Anchor statistic count");
  index(owner,owner.device());index(members,owner.device());index(counts,owner.device());index(visible,owner.device());fp32(opacity,owner.device());
  TORCH_CHECK(members.sizes()==torch::IntArrayRef({a,10})&&opacity.numel()==n,"Anchor statistics layout mismatch");
  for(size_t i=0;i<s.size();++i)TORCH_CHECK(s[i].is_cuda()&&s[i].device()==owner.device()&&s[i].is_contiguous()&&
      s[i].numel()==(i<2?a:n)&&s[i].scalar_type()==(i==0||i==2?torch::kFloat64:torch::kLong),"Anchor statistic layout/dtype");
  if(inactive.defined())TORCH_CHECK(recovered.defined()&&inactive.numel()==n&&recovered.numel()==n&&inactive.scalar_type()==torch::kBool&&recovered.scalar_type()==torch::kBool&&inactive.is_contiguous()&&recovered.is_contiguous(),"Slot state mismatch");
  auto seen=torch::zeros({a},owner.options().dtype(torch::kInt32));auto observed=torch::empty({n},s[0].options());auto stream=c10::cuda::getCurrentCUDAStream();
  if(visible.numel())visibility<<<(visible.numel()+127)/128,128,0,stream>>>(visible.numel(),visible.data_ptr<int64_t>(),owner.data_ptr<int64_t>(),seen.data_ptr<int>());
  if(a)stats<<<(a+127)/128,128,0,stream>>>(a,members.data_ptr<int64_t>(),counts.data_ptr<int64_t>(),seen.data_ptr<int>(),opacity.data_ptr<float>(),observed.data_ptr<double>(),s[1].data_ptr<int64_t>(),s[2].data_ptr<double>(),s[3].data_ptr<int64_t>(),s[4].data_ptr<int64_t>(),inactive.defined()?inactive.data_ptr<bool>():nullptr,recovered.defined()?recovered.data_ptr<bool>():nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Preserve the original FP64 accumulation kernel, including its rounding and
  // atomic semantics at a prune threshold. Fusing this reduction changes the
  // addition order; even a one-ULP difference can change a strict '<' decision.
  s[0].index_add_(0,owner,observed);
}
}

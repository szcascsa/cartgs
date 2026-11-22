#include "include/transform_field_cuda_optimizer.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>

namespace tf_cuda {
namespace {
constexpr int kBatch=32;
struct Batch {
  const float *p[kBatch],*g[kBatch],*old_m[kBatch],*old_v[kBatch];
  float *m[kBatch],*v[kBatch],*proposed[kBatch],*delta[kBatch];
  int64_t end[kBatch];
  float b1[kBatch],b2[kBatch],a1[kBatch],a2[kBatch],inv[kBatch],step[kBatch],eps[kBatch],decay[kBatch];
  int count;
};
__global__ void adamKernel(Batch a,int64_t count) {
  for(int64_t flat=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;flat<count;flat+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
    int tensor=0;while(flat>=a.end[tensor])++tensor;
    int64_t i=flat-(tensor?a.end[tensor-1]:0);
    float p=a.p[tensor][i],g=a.g[tensor][i];
    if(a.decay[tensor]!=0)g=fmaf(a.decay[tensor],p,g);
    float m=a.old_m[tensor]?a.old_m[tensor][i]:0,v=a.old_v[tensor]?a.old_v[tensor][i]:0;
    // Separate mul_, then ATen's alpha-add/addcmul; explicit FMA matches 2.3.1.
    m=fmaf(a.a1[tensor],g,a.b1[tensor]*m);
    v=fmaf(a.a2[tensor],g*g,a.b2[tensor]*v);
    float denom=sqrtf(v)*a.inv[tensor];denom+=a.eps[tensor];
    float proposed=fmaf(a.step[tensor],m/denom,p);
    a.m[tensor][i]=m;a.v[tensor][i]=v;a.proposed[tensor][i]=proposed;
    if(a.delta[tensor])a.delta[tensor][i]=proposed-p;
  }
}
void check(const torch::Tensor& t,const torch::Tensor& reference) {
  TORCH_CHECK(t.is_cuda() && t.device()==reference.device() && t.scalar_type()==torch::kFloat32 &&
              t.sizes()==reference.sizes() && t.is_contiguous(),"Invalid CUDA Adam tensor");
}
struct CopyBatch {
  const float *p[kBatch],*d[kBatch];float* out[kBatch];int64_t end[kBatch];
};
template<bool Add>
__global__ void copyKernel(CopyBatch b,int64_t count,float alpha) {
  for(int64_t flat=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;flat<count;flat+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
    int t=0;while(flat>=b.end[t])++t;int64_t i=flat-(t?b.end[t-1]:0);
    b.out[t][i]=Add?fmaf(alpha,b.d[t][i],b.p[t][i]):b.d[t][i];
  }
}
template<bool Add>
void copy(const std::vector<torch::Tensor>& parameters,const std::vector<torch::Tensor>& direction,
          const std::vector<torch::Tensor>& output,float alpha) {
  TORCH_CHECK(parameters.size()==direction.size() && parameters.size()==output.size(),"CUDA Adam block count");
  if(parameters.empty())return;
  c10::cuda::CUDAGuard guard(parameters[0].device());
  for(size_t i=0;i<parameters.size();++i) {
    TORCH_CHECK(parameters[i].device()==parameters[0].device(),"CUDA Adam requires one device");
    check(parameters[i],parameters[i]);check(direction[i],parameters[i]);check(output[i],parameters[i]);
  }
  for(size_t first=0;first<parameters.size();first+=kBatch) {
    CopyBatch b{};int64_t total=0;int count=std::min<size_t>(kBatch,parameters.size()-first);
    for(int i=0;i<count;++i) {
      b.p[i]=parameters[first+i].data_ptr<float>();b.d[i]=direction[first+i].data_ptr<float>();b.out[i]=output[first+i].data_ptr<float>();
      total+=parameters[first+i].numel();b.end[i]=total;
    }
    if(total){copyKernel<Add><<<std::min<int64_t>((total+255)/256,65535),256,0,c10::cuda::getCurrentCUDAStream()>>>(b,total,alpha);C10_CUDA_KERNEL_LAUNCH_CHECK();}
  }
}
}  // namespace
void adamUpdate(const std::vector<AdamUpdate>& updates) {
  if(updates.empty())return;
  c10::cuda::CUDAGuard guard(updates[0].parameter.device());
  for(const auto& u:updates) {
    TORCH_CHECK(u.parameter.device()==updates[0].parameter.device(),"TF Adam requires one CUDA device");
    for(auto t:{u.parameter,u.gradient,u.mean,u.variance,u.proposed})check(t,u.parameter);
    if(u.old_mean.defined()){check(u.old_mean,u.parameter);check(u.old_variance,u.parameter);}
    TORCH_CHECK(u.old_mean.defined()==u.old_variance.defined(),"Incomplete TF Adam state");
    if(u.delta.defined())check(u.delta,u.parameter);
  }
  for(size_t first=0;first<updates.size();first+=kBatch) {
    Batch b{};b.count=std::min<size_t>(kBatch,updates.size()-first);int64_t total=0;
    for(int i=0;i<b.count;++i) {
      const auto& u=updates[first+i];
      b.p[i]=u.parameter.data_ptr<float>();b.g[i]=u.gradient.data_ptr<float>();
      b.old_m[i]=u.old_mean.defined()?u.old_mean.data_ptr<float>():nullptr;b.old_v[i]=u.old_variance.defined()?u.old_variance.data_ptr<float>():nullptr;
      b.m[i]=u.mean.data_ptr<float>();b.v[i]=u.variance.data_ptr<float>();b.proposed[i]=u.proposed.data_ptr<float>();b.delta[i]=u.delta.defined()?u.delta.data_ptr<float>():nullptr;
      total+=u.parameter.numel();b.end[i]=total;b.b1[i]=u.beta1;b.b2[i]=u.beta2;b.a1[i]=u.alpha1;b.a2[i]=u.alpha2;
      b.inv[i]=u.inverse_bias2;b.step[i]=u.step_size;b.eps[i]=u.eps;b.decay[i]=u.weight_decay;
    }
    if(total){adamKernel<<<std::min<int64_t>((total+255)/256,65535),256,0,c10::cuda::getCurrentCUDAStream()>>>(b,total);C10_CUDA_KERNEL_LAUNCH_CHECK();}
  }
}
std::vector<torch::Tensor> adamCandidate(const std::vector<torch::Tensor>& parameters,
                                       const std::vector<torch::Tensor>& direction,double alpha) {
  torch::NoGradGuard no_grad;std::vector<torch::Tensor> result;
  for(const auto& p:parameters)result.push_back(torch::empty_like(p));
  copy<true>(parameters,direction,result,alpha);return result;
}
void adamCopy(const std::vector<torch::Tensor>& parameters,const std::vector<torch::Tensor>& candidate) {
  copy<false>(parameters,candidate,parameters,0);
}
}  // namespace tf_cuda

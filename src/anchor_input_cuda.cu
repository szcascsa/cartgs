#include "include/anchor_input_cuda.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace mapping_cuda {
namespace {
__global__ void pack(int64_t n,const float* xyz,const float* feature,const int64_t* owner,
    const float* center,const float* reference,const float* scale,const float* dc,const float* opacity,
    float ratio,float* out) {
  const int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  const int64_t a=owner[i];float* row=out+27*i;
  for(int j=0;j<16;++j)row[j]=feature[16*a+j];
  for(int j=0;j<3;++j){row[16+j]=(xyz[3*i+j]-center[3*a+j])/reference[a];
    row[19+j]=scale[3*i+j];row[22+j]=dc[3*i+j];}
  row[25]=opacity[i];row[26]=logf(ratio);
}
__global__ void vjp(int64_t n,const int64_t* owner,const float* reference,const float* g,
    float* gx,float* gf,float* gs,float* gd,float* go) {
  const int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  const int64_t a=owner[i];const float* row=g+27*i;
  if(gf)for(int j=0;j<16;++j)atomicAdd(gf+16*a+j,row[j]);
  for(int j=0;j<3;++j){if(gx)gx[3*i+j]=row[16+j]/reference[a];
    if(gs)gs[3*i+j]=row[19+j];if(gd)gd[3*i+j]=row[22+j];}
  if(go)go[i]=row[25];
}
void check(const std::vector<torch::Tensor>& a) {
  TORCH_CHECK(a.size()==8,"Input27 input count");const int64_t n=a[0].size(0),m=a[1].size(0);
  for(size_t i=0;i<a.size();++i)TORCH_CHECK(a[i].is_cuda()&&a[i].device()==a[0].device()&&a[i].is_contiguous()&&
      a[i].scalar_type()==(i==2?torch::kInt64:torch::kFloat32),"Input27 requires contiguous CUDA FP32 and Int64 owner");
  TORCH_CHECK(a[0].sizes()==torch::IntArrayRef({n,3})&&a[1].sizes()==torch::IntArrayRef({m,16})&&
      a[2].numel()==n&&a[3].sizes()==torch::IntArrayRef({m,3})&&a[4].numel()==m&&
      a[5].sizes()==a[0].sizes()&&a[6].numel()==n*3&&a[7].sizes()==torch::IntArrayRef({n,1}),"Input27 attribute shapes");
  TORCH_CHECK(!a[3].requires_grad()&&!a[4].requires_grad(),"Anchor geometry buffers must be non-trainable");
}
const float* ptr(const torch::Tensor& a){return a.data_ptr<float>();}
float* outptr(torch::Tensor& a){return a.defined()?a.data_ptr<float>():nullptr;}
}
torch::Tensor anchorInput27Forward(const std::vector<torch::Tensor>& a,double ratio) {
  check(a);TORCH_CHECK(std::isfinite(ratio)&&ratio>0&&ratio<=1,"Invalid Input27 ratio");
  c10::cuda::CUDAGuard guard(a[0].device());const auto n=a[0].size(0);
  auto out=torch::empty({n,27},a[0].options());
  if(n){pack<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(n,ptr(a[0]),ptr(a[1]),a[2].data_ptr<int64_t>(),
      ptr(a[3]),ptr(a[4]),ptr(a[5]),ptr(a[6]),ptr(a[7]),static_cast<float>(ratio),out.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return out;
}
std::vector<torch::Tensor> anchorInput27Backward(const std::vector<torch::Tensor>& a,
    const torch::Tensor& g,const std::vector<bool>& needs) {
  check(a);c10::cuda::CUDAGuard guard(a[0].device());const auto n=a[0].size(0);
  std::vector<torch::Tensor> out(8);
  for(int i:{0,1,5,6,7})if(needs[i])out[i]=i==1?torch::zeros_like(a[i]):torch::empty_like(a[i]);
  if(n){vjp<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(n,a[2].data_ptr<int64_t>(),ptr(a[4]),ptr(g),
      outptr(out[0]),outptr(out[1]),outptr(out[5]),outptr(out[6]),outptr(out[7]));C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return out;
}
}

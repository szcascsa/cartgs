#include "include/spatial_compensation_cuda.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
namespace mapping_cuda {
namespace {
void check(const std::vector<torch::Tensor>& a){for(const auto& v:a)TORCH_CHECK(v.is_cuda()&&v.scalar_type()==torch::kFloat32&&v.is_contiguous()&&v.device()==a[0].device(),"Spatial CUDA requires same-device contiguous FP32");}
const float* p(const torch::Tensor& v){return v.data_ptr<float>();}
__global__ void pack(int64_t n,const float* raw,const float* coords,const float* mean,const float* stddev,const float* freq,float* out){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n*55)return;
  const int64_t row=i/55,col=i%55;
  if(col<19){float v=(raw[row*19+col]-mean[col])/stddev[col];out[i]=fminf(15.f,fmaxf(-15.f,v));}
  else {int k=(col-19)%18;float angle=coords[row*3+k/6]*freq[k%6];out[i]=col<37?sinf(angle):cosf(angle);}
}
__global__ void packVjp(int64_t n,const float* raw,const float* coords,const float* mean,const float* stddev,const float* freq,const float* g,float* gr,float* gc){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n*19)return;
  int64_t row=i/19,col=i%19;float v=(raw[i]-mean[col])/stddev[col];gr[i]=(v>=-15.f&&v<=15.f)?g[row*55+col]/stddev[col]:0.f;
  if(col<3){float sum=0;for(int j=0;j<6;++j){float angle=coords[row*3+col]*freq[j];sum+=(g[row*55+19+col*6+j]*cosf(angle)+g[row*55+37+col*6+j]*(-sinf(angle)))*freq[j];}gc[row*3+col]=sum;}
}
__global__ void film(int64_t n,const float* h,const float* gain,const float* bias,float* out){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)out[i]=fmaxf((1.f+gain[i%64])*h[i]+bias[i%64],0.f);
}
__global__ void filmVjp(int64_t n,const float* h,const float* gain,const float* out,const float* g,float* gh,float* gg,float* gb){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;float v=out[i]>0?g[i]:0.f;gh[i]=v*(1.f+gain[i%64]);gg[i]=v*h[i];gb[i]=v;
}
}
torch::Tensor spatialInputForward(const std::vector<torch::Tensor>& a){
  check(a);TORCH_CHECK(a[0].dim()==2&&a[0].size(1)==19&&a[1].sizes()==torch::IntArrayRef({a[0].size(0),3})&&a[2].numel()==19&&a[3].numel()==19&&a[4].numel()==6,"Spatial input shape mismatch");
  c10::cuda::CUDAGuard guard(a[0].device());auto n=a[0].size(0);auto out=torch::empty({n,55},a[0].options());
  if(n){pack<<<(n*55+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,p(a[0]),p(a[1]),p(a[2]),p(a[3]),p(a[4]),out.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}return out;
}
std::vector<torch::Tensor> spatialInputBackward(const std::vector<torch::Tensor>& a,const torch::Tensor& g){
  check(a);c10::cuda::CUDAGuard guard(a[0].device());auto n=a[0].size(0);auto gr=torch::empty_like(a[0]),gc=torch::empty_like(a[1]);
  if(n){packVjp<<<(n*19+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,p(a[0]),p(a[1]),p(a[2]),p(a[3]),p(a[4]),p(g),gr.data_ptr<float>(),gc.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}return {gr,gc};
}
torch::Tensor spatialFilmForward(const std::vector<torch::Tensor>& a){
  check(a);TORCH_CHECK(a[0].dim()==2&&a[0].size(1)==64&&a[1].numel()==64&&a[2].numel()==64,"Spatial FiLM shape mismatch");
  c10::cuda::CUDAGuard guard(a[0].device());auto out=torch::empty_like(a[0]);auto n=out.numel();
  if(n){film<<<(n+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,p(a[0]),p(a[1]),p(a[2]),out.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}return out;
}
std::vector<torch::Tensor> spatialFilmBackward(const std::vector<torch::Tensor>& a,const torch::Tensor& out,const torch::Tensor& g){
  check(a);c10::cuda::CUDAGuard guard(a[0].device());auto gh=torch::empty_like(a[0]),gg=torch::empty_like(gh),gb=torch::empty_like(gh);auto n=gh.numel();
  if(n){filmVjp<<<(n+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,p(a[0]),p(a[1]),p(out),p(g),gh.data_ptr<float>(),gg.data_ptr<float>(),gb.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}return {gh,gg,gb};
}
}

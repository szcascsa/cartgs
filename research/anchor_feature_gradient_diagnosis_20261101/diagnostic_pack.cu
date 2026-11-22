#include "diagnostic_pack.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
namespace diagnostic_cuda {
namespace {
struct Features {const float* table[6];float weight[6];int count;};
__global__ void pack(int64_t n,const float* x,const float* c,const float* reference,
    const float* mean,const float* stddev,Features features,float* output,float* saved,const int64_t* owners) {
  auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n*51)return;
  const auto row=i/51,col=i%51;
  if(col<3) {
    float q=__fdiv_rn(__fsub_rn(x[row*3+col],c[row*3+col]),reference[row]);
    float v=__fdiv_rn(__fsub_rn(asinhf(q),mean[3+col]),stddev[3+col]);
    output[i]=v<-15.f?-15.f:(v>15.f?15.f:v);
    if(saved){saved[row*6+col]=q;saved[row*6+3+col]=v;}
  } else {
    const auto index=(owners?owners[row]:row)*48+col-3;float value=features.table[0][index];
    for(int j=1;j<features.count;++j)value=__fadd_rn(value,__fmul_rn(features.weight[j],features.table[j][index]));
    output[i]=value;
  }
}
__global__ void xyzVjp(int64_t n,const float* reference,const float* stddev,const float* saved,
    const float* grad,int64_t stride0,int64_t stride1,float* output) {
  auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n*3)return;
  const auto row=i/3,col=i%3;float q=saved[row*6+col],v=saved[row*6+3+col];
  float g=__fdiv_rn(v>=-15.f&&v<=15.f?grad[row*stride0+col*stride1]:0.f,stddev[3+col]);
  g=__fmul_rn(g,rsqrtf(__fadd_rn(__fmul_rn(q,q),1.f)));
  output[i]=__fdiv_rn(g,reference[row]);
}
}
std::vector<torch::Tensor> residual48PackForward(const torch::Tensor& x,const torch::Tensor& c,
    const torch::Tensor& r,const torch::Tensor& mean,const torch::Tensor& stddev,
    const std::vector<torch::Tensor>& features,const std::vector<double>& weights) {
  c10::cuda::CUDAGuard guard(x.device());const auto n=x.size(0);
  TORCH_CHECK(x.sizes()==torch::IntArrayRef({n,3})&&c.sizes()==x.sizes()&&r.numel()==n&&mean.numel()==19&&stddev.numel()==19,"Residual48 pack shape");
  Features f{};f.count=features.size();
  for(int j=0;j<f.count;++j) {
    TORCH_CHECK(features[j].is_cuda()&&features[j].device()==x.device()&&features[j].scalar_type()==torch::kFloat32&&
                features[j].is_contiguous()&&features[j].sizes()==torch::IntArrayRef({n,48}),"Residual48 gathered Feature shape");
    f.table[j]=features[j].data_ptr<float>();f.weight[j]=float(weights[j]);
  }
  auto output=torch::empty({n,51},x.options()),saved=torch::empty({n,6},x.options());
  if(n){pack<<<(n*51+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,x.data_ptr<float>(),c.data_ptr<float>(),r.data_ptr<float>(),
      mean.data_ptr<float>(),stddev.data_ptr<float>(),f,output.data_ptr<float>(),saved.data_ptr<float>(),nullptr);C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return {output,saved};
}
torch::Tensor residual48XYZBackward(const torch::Tensor& reference,const torch::Tensor& stddev,
    const torch::Tensor& saved,const torch::Tensor& gradient) {
  c10::cuda::CUDAGuard guard(reference.device());const auto n=reference.numel();
  auto output=torch::empty({n,3},gradient.options());
  if(n){xyzVjp<<<(n*3+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,reference.data_ptr<float>(),stddev.data_ptr<float>(),saved.data_ptr<float>(),
      gradient.data_ptr<float>(),gradient.stride(0),gradient.stride(1),output.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return output;
}
}

#include "include/residual48_fusion.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace mapping_cuda {
namespace {
struct Features {const float* table[6];float weight[6];int count;};
struct HeadPointers {const float* input[4];};
__global__ void trainingPack(int64_t n,const float* x,const float* c,const float* reference,
    const float* mean,const float* stddev,Features features,float* output,float* saved) {
  auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n*51)return;
  const auto row=i/51,col=i%51;
  if(col<3) {
    float q=__fdiv_rn(__fsub_rn(x[row*3+col],c[row*3+col]),reference[row]);
    float v=__fdiv_rn(__fsub_rn(asinhf(q),mean[3+col]),stddev[3+col]);
    output[i]=v<-15.f?-15.f:(v>15.f?15.f:v);
    saved[row*6+col]=q;saved[row*6+3+col]=v;
  }else {
    const auto index=row*48+col-3;float value=features.table[0][index];
    for(int j=1;j<features.count;++j)value=__fadd_rn(value,__fmul_rn(features.weight[j],features.table[j][index]));
    output[i]=value;
  }
}
__global__ void trainingXYZVjp(int64_t n,const float* reference,const float* stddev,const float* saved,
    const float* grad,int64_t stride0,int64_t stride1,float* output) {
  auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n*3)return;
  const auto row=i/3,col=i%3;float q=saved[row*6+col],v=saved[row*6+3+col];
  float g=__fdiv_rn(v>=-15.f&&v<=15.f?grad[row*stride0+col*stride1]:0.f,stddev[3+col]);
  g=__fmul_rn(g,rsqrtf(__fadd_rn(__fmul_rn(q,q),1.f)));
  output[i]=__fdiv_rn(g,reference[row]);
}
__global__ void addRelu(int64_t n,const float* input,const float* ratio,float* output) {
  auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n*64)return;
  float v=__fadd_rn(input[i],ratio[i%64]);output[i]=v<=0.f?0.f:v;
}
__global__ void output14(int64_t n,HeadPointers p,const float* reference,float* output) {
  auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n*14)return;
  auto row=i/14,col=i%14;
  if(col<3)output[i]=__fmul_rn(p.input[0][row*3+col],reference[row]);
  else if(col<6)output[i]=p.input[1][row*3+col-3];
  else if(col<10)output[i]=p.input[2][row*4+col-6];
  else output[i]=p.input[3][row*4+col-10];
}
template<int Count> __global__ void inferencePack(int64_t n,const float* x,const float* c,const float* reference,
    const float* mean,const float* stddev,const int64_t* owners,Features features,float* output) {
  const int64_t row=int64_t(blockIdx.x)*4+(threadIdx.x>>6);const int col=threadIdx.x&63;if(row>=n)return;
  if(col<48) {
    const auto index=owners[row]*48+col;float value=features.table[0][index];
    #pragma unroll
    for(int j=1;j<Count;++j)value=__fadd_rn(value,__fmul_rn(features.weight[j],features.table[j][index]));
    output[row*51+col+3]=value;
  }else if(col<51) {
    const int axis=col-48;float q=__fdiv_rn(__fsub_rn(x[row*3+axis],c[row*3+axis]),reference[row]);
    float v=__fdiv_rn(__fsub_rn(asinhf(q),mean[3+axis]),stddev[3+axis]);
    output[row*51+axis]=v<-15.f?-15.f:(v>15.f?15.f:v);
  }
}
}
std::vector<torch::Tensor> residual48TrainingPackForward(const torch::Tensor& x,const torch::Tensor& c,
    const torch::Tensor& r,const torch::Tensor& mean,const torch::Tensor& stddev,
    const std::vector<torch::Tensor>& features,const std::vector<double>& weights) {
  c10::cuda::CUDAGuard guard(x.device());const auto n=x.size(0);
  TORCH_CHECK(x.sizes()==torch::IntArrayRef({n,3})&&c.sizes()==x.sizes()&&r.numel()==n&&mean.numel()==19&&stddev.numel()==19,"Residual48 training pack shape");
  Features f{};f.count=features.size();
  for(int j=0;j<f.count;++j) {
    TORCH_CHECK(features[j].is_cuda()&&features[j].device()==x.device()&&features[j].scalar_type()==torch::kFloat32&&
        features[j].is_contiguous()&&features[j].sizes()==torch::IntArrayRef({n,48}),"Residual48 gathered Feature shape");
    f.table[j]=features[j].data_ptr<float>();f.weight[j]=float(weights[j]);
  }
  auto output=torch::empty({n,51},x.options()),saved=torch::empty({n,6},x.options());
  if(n){trainingPack<<<(n*51+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,x.data_ptr<float>(),c.data_ptr<float>(),r.data_ptr<float>(),
      mean.data_ptr<float>(),stddev.data_ptr<float>(),f,output.data_ptr<float>(),saved.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return {output,saved};
}

torch::Tensor residual48TrainingXYZBackward(const torch::Tensor& reference,const torch::Tensor& stddev,
    const torch::Tensor& saved,const torch::Tensor& gradient) {
  c10::cuda::CUDAGuard guard(reference.device());const auto n=reference.numel();
  auto output=torch::empty({n,3},gradient.options());
  if(n){trainingXYZVjp<<<(n*3+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,reference.data_ptr<float>(),stddev.data_ptr<float>(),saved.data_ptr<float>(),
      gradient.data_ptr<float>(),gradient.stride(0),gradient.stride(1),output.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return output;
}

torch::Tensor residual48AddReluInference(const torch::Tensor& encoded,const torch::Tensor& ratio) {
  TORCH_CHECK(!torch::GradMode::is_enabled()&&encoded.is_cuda()&&encoded.scalar_type()==torch::kFloat32&&encoded.is_contiguous()&&
      encoded.dim()==2&&encoded.size(1)==64&&ratio.sizes()==torch::IntArrayRef({1,64})&&ratio.is_contiguous(),"Residual48 inference add/ReLU");
  c10::cuda::CUDAGuard guard(encoded.device());auto output=torch::empty_like(encoded);const auto n=encoded.size(0);
  if(n){addRelu<<<(n*64+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,encoded.data_ptr<float>(),ratio.data_ptr<float>(),output.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return output;
}
torch::Tensor residual48OutputInference(const std::vector<torch::Tensor>& input,const torch::Tensor& reference) {
  TORCH_CHECK(!torch::GradMode::is_enabled()&&input.size()==4&&reference.numel()==input[0].size(0),"Residual48 inference output");
  c10::cuda::CUDAGuard guard(reference.device());const auto n=input[0].size(0);HeadPointers p{};
  for(int j=0;j<4;++j){TORCH_CHECK(input[j].is_contiguous()&&input[j].sizes()==torch::IntArrayRef({n,j<2?3:4}),"Head output shape");p.input[j]=input[j].data_ptr<float>();}
  auto out=torch::empty({n,14},input[0].options());
  if(n){output14<<<(n*14+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,p,reference.data_ptr<float>(),out.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return out;
}
torch::Tensor residual48InferencePack(const torch::Tensor& x,const torch::Tensor& center,
    const torch::Tensor& reference,const torch::Tensor& mean,const torch::Tensor& stddev,
    const torch::Tensor& owners,const std::vector<torch::Tensor>& tables,const std::vector<double>& weights) {
  TORCH_CHECK(!torch::GradMode::is_enabled(),"Residual48 direct gather is inference-only; original training scatter is retained");
  c10::cuda::CUDAGuard guard(x.device());const auto n=x.size(0);
  TORCH_CHECK(x.scalar_type()==torch::kFloat32&&center.scalar_type()==torch::kFloat32&&reference.scalar_type()==torch::kFloat32&&
      mean.scalar_type()==torch::kFloat32&&stddev.scalar_type()==torch::kFloat32&&mean.is_contiguous()&&stddev.is_contiguous()&&mean.numel()==19&&stddev.numel()==19,
      "Residual48 inference FP32 normalization");
  TORCH_CHECK(weights.size()>=1&&weights.size()<=6&&tables.size()>=weights.size(),"Residual48 active tables");
  TORCH_CHECK(x.is_contiguous()&&center.is_contiguous()&&reference.is_contiguous()&&owners.is_contiguous()&&
      owners.scalar_type()==torch::kInt64&&owners.numel()==n&&x.sizes()==torch::IntArrayRef({n,3})&&center.sizes()==x.sizes()&&reference.numel()==n,
      "Residual48 inference geometry shape");
  Features f{};f.count=weights.size();
  for(int j=0;j<f.count;++j) {
    TORCH_CHECK(tables[j].is_cuda()&&tables[j].device()==x.device()&&tables[j].scalar_type()==torch::kFloat32&&
        tables[j].is_contiguous()&&tables[j].dim()==2&&tables[j].size(1)==48,"Residual48 inference table");
    f.table[j]=tables[j].data_ptr<float>();f.weight[j]=float(weights[j]);
  }
  auto output=torch::empty({n,51},x.options());
  if(n) {
    const auto stream=c10::cuda::getCurrentCUDAStream();
    #define LAUNCH_PACK(C) inferencePack<C><<<(n+3)/4,256,0,stream>>>(n,x.data_ptr<float>(),center.data_ptr<float>(),reference.data_ptr<float>(),mean.data_ptr<float>(),stddev.data_ptr<float>(),owners.data_ptr<int64_t>(),f,output.data_ptr<float>())
    switch(f.count){case 1:LAUNCH_PACK(1);break;case 2:LAUNCH_PACK(2);break;case 3:LAUNCH_PACK(3);break;
      case 4:LAUNCH_PACK(4);break;case 5:LAUNCH_PACK(5);break;case 6:LAUNCH_PACK(6);break;}
    #undef LAUNCH_PACK
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}
}

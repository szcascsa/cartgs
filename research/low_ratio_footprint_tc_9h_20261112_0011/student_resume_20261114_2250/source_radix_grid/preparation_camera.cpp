#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <algorithm>

namespace {
template<class T> void cameraInverse(const T* input,T* inverse) {
  T a[4][8]{};
  for(int r=0;r<4;++r)for(int c=0;c<4;++c){a[r][c]=input[r*4+c];a[r][c+4]=r==c?T(1):T(0);}
  for(int c=0;c<4;++c){
    int pivot=c;for(int r=c+1;r<4;++r)if(std::abs(a[r][c])>std::abs(a[pivot][c]))pivot=r;
    TORCH_CHECK(std::isfinite(a[pivot][c])&&a[pivot][c]!=T(0),"Singular or nonfinite deployment camera");
    if(pivot!=c)for(int j=0;j<8;++j)std::swap(a[pivot][j],a[c][j]);
    T denominator=a[c][c];for(int j=0;j<8;++j)a[c][j]/=denominator;
    for(int r=0;r<4;++r)if(r!=c){T weight=a[r][c];for(int j=0;j<8;++j)a[r][j]-=weight*a[c][j];}
  }
  for(int r=0;r<4;++r)for(int c=0;c<4;++c)inverse[r*4+c]=a[r][c+4];
}
}

torch::Tensor preparation_camera(const torch::Tensor& pose,const torch::Tensor& projection,
    const torch::Tensor& device_reference) {
  TORCH_CHECK(!torch::GradMode::is_enabled()&&pose.device().is_cpu()&&projection.device().is_cpu()&&
      pose.scalar_type()==at::kDouble&&projection.scalar_type()==at::kFloat&&pose.is_contiguous()&&projection.is_contiguous()&&
      pose.sizes()==at::IntArrayRef({4,4})&&projection.sizes()==pose.sizes()&&device_reference.is_cuda(),"Native camera input layout");
  const double* input=pose.data_ptr<double>();for(int j=0;j<16;++j)TORCH_CHECK(std::isfinite(input[j]),"Nonfinite camera pose");
  double w2c[16];cameraInverse(input,w2c);
  c10::cuda::CUDAGuard guard(device_reference.device());
  auto staging=torch::empty({35},torch::TensorOptions().device(at::kCPU).dtype(at::kFloat).pinned_memory(true));
  float* data=staging.data_ptr<float>();
  for(int r=0;r<4;++r)for(int c=0;c<4;++c)data[r*4+c]=float(w2c[c*4+r]);
  const float* p=projection.data_ptr<float>();
  for(int r=0;r<4;++r)for(int c=0;c<4;++c){float x=0.f;for(int k=0;k<4;++k)x=std::fma(data[r*4+k],p[k*4+c],x);data[16+r*4+c]=x;}
  float inverse[16];cameraInverse(data,inverse);for(int c=0;c<3;++c)data[32+c]=inverse[12+c];
  for(int j=0;j<35;++j)TORCH_CHECK(std::isfinite(data[j]),"Nonfinite prepared camera");
  auto output=torch::empty({35},device_reference.options().dtype(at::kFloat));
  // ATen owns the pinned allocation event and keeps staging alive until this Stream's copy completes.
  output.copy_(staging,true);
  return output;
}

std::vector<torch::Tensor> preparation_camera_parts(const torch::Tensor& pose,
    const torch::Tensor& projection,const torch::Tensor& device_reference) {
  auto flat=preparation_camera(pose,projection,device_reference);
  return {flat.slice(0,0,16).view({4,4}),flat.slice(0,16,32).view({4,4}),flat.slice(0,32,35)};
}

// 前一帧finish_pending已同步，稳定pinned缓冲可安全写入本帧。
void preparation_camera_into(const double* input,const torch::Tensor& projection,
    const torch::Tensor& staging,const torch::Tensor& output){
  TORCH_CHECK(!torch::GradMode::is_enabled()&&projection.device().is_cpu()&&projection.scalar_type()==at::kFloat&&
      projection.is_contiguous()&&projection.sizes()==at::IntArrayRef({4,4}),"Native reused camera projection");
  for(int j=0;j<16;++j)TORCH_CHECK(std::isfinite(input[j]),"Nonfinite camera pose");
  double w2c[16];cameraInverse(input,w2c);
  float* data=staging.data_ptr<float>();
  for(int r=0;r<4;++r)for(int c=0;c<4;++c)data[r*4+c]=float(w2c[c*4+r]);
  const float* p=projection.data_ptr<float>();
  for(int r=0;r<4;++r)for(int c=0;c<4;++c){float x=0.f;for(int k=0;k<4;++k)x=std::fma(data[r*4+k],p[k*4+c],x);data[16+r*4+c]=x;}
  float inverse[16];cameraInverse(data,inverse);for(int c=0;c<3;++c)data[32+c]=inverse[12+c];
  for(int j=0;j<35;++j)TORCH_CHECK(std::isfinite(data[j]),"Nonfinite prepared camera");
  C10_CUDA_CHECK(cudaMemcpyAsync(output.data_ptr<float>(),staging.data_ptr<float>(),35*sizeof(float),cudaMemcpyHostToDevice,c10::cuda::getCurrentCUDAStream()));
}

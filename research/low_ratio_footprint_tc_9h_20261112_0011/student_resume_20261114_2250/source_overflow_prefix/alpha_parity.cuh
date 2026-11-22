#include <cuda_runtime.h>
__global__ void checkMergedAlpha(int n,const float* powers,float* old_alpha,float* new_alpha){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float p=powers[i],a=0.f;
  if(!(p<-5.541263545158426f)){a=fminf(.99f,boundedExp<1>(p));if(a<1.f/255.f)a=0.f;}
  old_alpha[i]=a;new_alpha[i]=mergedFoldedAlpha(p,false);
}
std::vector<torch::Tensor> alphaParity(const torch::Tensor& powers){
  TORCH_CHECK(powers.is_cuda()&&powers.scalar_type()==at::kFloat&&powers.is_contiguous(),"Alpha parity FP32 CUDA");
  auto a=torch::empty_like(powers),b=torch::empty_like(powers);
  checkMergedAlpha<<<(powers.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(powers.numel(),powers.data_ptr<float>(),a.data_ptr<float>(),b.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {a,b};
}

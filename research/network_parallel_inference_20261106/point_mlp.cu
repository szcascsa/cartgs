#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
struct Params { const float* p[21]; };
__device__ float reluExact(float v){return v<=0.f?0.f:v;}
// Four independent Gaussian warps in a block. The trial intentionally exposes
// the accumulation order for direct comparison with the existing cuBLAS MLP.
__global__ void pointSelector(int n,int width,const float* input,Params weights,float* output) {
 const int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32,warp=threadIdx.x/32;
 if(row>=n)return;
 __shared__ float h0[4][32],h1[4][32];
 float sum=0.f;
 for(int j=0;j<width;++j)sum=__fmaf_rn(input[row*width+j],weights.p[0][lane*width+j],sum);
 h0[warp][lane]=reluExact(__fadd_rn(sum,weights.p[1][lane]));__syncwarp();
 sum=0.f;for(int j=0;j<32;++j)sum=__fmaf_rn(h0[warp][j],weights.p[2][lane*32+j],sum);
 h1[warp][lane]=reluExact(__fadd_rn(sum,weights.p[3][lane]));__syncwarp();
 if(lane<2){sum=0.f;for(int j=0;j<32;++j)sum=__fmaf_rn(h1[warp][j],weights.p[4][lane*32+j],sum);output[row*2+lane]=__fadd_rn(sum,weights.p[5][lane]);}
}
__global__ void pointComp(int n,const float* input,const float* ratio,const float* view,const float* reference,Params w,float* output) {
 const int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32,warp=threadIdx.x/32;
 if(row>=n)return;
 __shared__ float h0[4][64],h1[4][64],h2[4][64];
 for(int k=0;k<2;++k){int col=lane+32*k;float sum=0.f;for(int j=0;j<51;++j)sum=__fmaf_rn(input[row*51+j],w.p[0][col*51+j],sum);
  sum=__fadd_rn(__fadd_rn(sum,w.p[1][col]),ratio[col]);if(view)sum=__fadd_rn(sum,view[row*64+col]);h0[warp][col]=reluExact(sum);}
 __syncwarp();
 for(int k=0;k<2;++k){int col=lane+32*k;float sum=0.f;for(int j=0;j<64;++j)sum=__fmaf_rn(h0[warp][j],w.p[2][col*64+j],sum);h1[warp][col]=reluExact(__fadd_rn(sum,w.p[3][col]));}
 __syncwarp();
 for(int head=0;head<4;++head){int p=5+head*4;
  for(int k=0;k<2;++k){int col=lane+32*k;float sum=0.f;for(int j=0;j<64;++j)sum=__fmaf_rn(h1[warp][j],w.p[p][col*64+j],sum);h2[warp][col]=reluExact(__fadd_rn(sum,w.p[p+1][col]));}
  __syncwarp();const int width=head<2?3:4,offset=head==0?0:head==1?3:head==2?6:10;
  if(lane<width){float sum=0.f;for(int j=0;j<64;++j)sum=__fmaf_rn(h2[warp][j],w.p[p+2][lane*64+j],sum);sum=__fadd_rn(sum,w.p[p+3][lane]);if(head==0)sum=__fmul_rn(sum,reference[row]);output[row*14+offset+lane]=sum;}
  __syncwarp();
 }
}
torch::Tensor selector(torch::Tensor input,std::vector<torch::Tensor> weights){
 TORCH_CHECK(!torch::GradMode::is_enabled() && input.is_cuda() && input.is_contiguous() && weights.size()==6,"Frozen FP32 probe");
 c10::cuda::CUDAGuard guard(input.device());Params w{};for(int i=0;i<6;++i)w.p[i]=weights[i].data_ptr<float>();
 auto out=torch::empty({input.size(0),2},input.options());
 if(input.size(0))pointSelector<<<(input.size(0)+3)/4,128,0,c10::cuda::getCurrentCUDAStream()>>>(input.size(0),input.size(1),input.data_ptr<float>(),w,out.data_ptr<float>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor compensator(torch::Tensor input,torch::Tensor ratio,torch::Tensor view,torch::Tensor reference,std::vector<torch::Tensor> weights){
 TORCH_CHECK(!torch::GradMode::is_enabled() && input.is_cuda() && input.is_contiguous() && weights.size()==21,"Frozen FP32 probe");
 c10::cuda::CUDAGuard guard(input.device());Params w{};for(int i=0;i<21;++i)w.p[i]=weights[i].data_ptr<float>();
 auto out=torch::empty({input.size(0),14},input.options());
 if(input.size(0))pointComp<<<(input.size(0)+3)/4,128,0,c10::cuda::getCurrentCUDAStream()>>>(input.size(0),input.data_ptr<float>(),ratio.data_ptr<float>(),view.numel()?view.data_ptr<float>():nullptr,reference.data_ptr<float>(),w,out.data_ptr<float>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

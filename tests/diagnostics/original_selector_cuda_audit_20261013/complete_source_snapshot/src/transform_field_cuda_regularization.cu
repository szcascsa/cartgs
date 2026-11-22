#include "include/transform_field_cuda.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>
#include <algorithm>

namespace tf_cuda {
namespace {
constexpr int kPlanes=48;
struct Regularizer {
  const float* data[kPlanes];float* grad[kPlanes];
  int height[kPlanes],width[kPlanes],channels[kPlanes],numel[kPlanes];
  int count,blocks;
};
Regularizer describe(const std::vector<torch::Tensor>& planes) {
  TORCH_CHECK(!planes.empty() && planes.size()%6==0 && planes.size()<=kPlanes,"Invalid regularization planes");
  Regularizer a{};a.count=planes.size();int max_size=0;
  for(int i=0;i<a.count;++i) {
    const auto& p=planes[i];
    TORCH_CHECK(p.is_cuda() && p.device()==planes[0].device() && p.scalar_type()==torch::kFloat32 &&
        p.is_contiguous() && p.dim()==4 && p.size(0)==1 && p.size(1)>0 && p.size(2)>0 &&
        p.size(3)>0 && p.numel()<=INT_MAX,"Regularization requires contiguous CUDA FP32 NCHW planes");
    a.data[i]=p.data_ptr<float>();a.height[i]=p.size(2);a.width[i]=p.size(3);
    a.channels[i]=p.size(1);a.numel[i]=p.numel();max_size=std::max(max_size,a.numel[i]);
  }
  a.blocks=(max_size+255)/256;return a;
}
__device__ bool ratioPlane(int p) {return p%6==2 || p%6==4 || p%6==5;}
__device__ float second(const float* p,int index,int width) {
  const float d1a=p[index+width]-p[index];
  const float d1b=p[index+2*width]-p[index+width];
  return d1b-d1a;
}
__global__ void regularizationPartialKernel(Regularizer a,float* partial) {
  __shared__ float smooth[256],l1[256];
  int p=blockIdx.y,tid=threadIdx.x,index=blockIdx.x*256+tid;
  int h=a.height[p],w=a.width[p];float s=0,l=0;
  if(h<3) {if(index<a.numel[p])s=a.data[p][index];}
  else if(index<a.channels[p]*(h-2)*w) {
    int channel=index/((h-2)*w),j=(index/w)%(h-2),col=index%w;
    float d=second(a.data[p],(channel*h+j)*w+col,w);s=d*d;
  }
  if(ratioPlane(p) && index<a.numel[p])l=fabsf(1.0f-a.data[p][index]);
  smooth[tid]=s;l1[tid]=l;__syncthreads();
  for(int stride=128;stride;stride/=2) {
    if(tid<stride){smooth[tid]+=smooth[tid+stride];l1[tid]+=l1[tid+stride];}__syncthreads();
  }
  if(!tid){int offset=(p*a.blocks+blockIdx.x)*2;partial[offset]=smooth[0];partial[offset+1]=l1[0];}
}
__global__ void regularizationReduceKernel(Regularizer a,const float* partial,float* loss,
                                          float sw,float rw,float lw) {
  __shared__ float smooth[kPlanes],l1[kPlanes];
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  for(int p=warp;p<a.count;p+=8) {
    float s=0,l=0;
    for(int i=lane;i<a.blocks;i+=32){s+=partial[(p*a.blocks+i)*2];l+=partial[(p*a.blocks+i)*2+1];}
    for(int offset=16;offset;offset/=2){s+=__shfl_down_sync(0xffffffff,s,offset);l+=__shfl_down_sync(0xffffffff,l,offset);}
    if(!lane) {
      smooth[p]=a.height[p]<3?s*0.0f:s/(a.channels[p]*(a.height[p]-2)*a.width[p]);
      l1[p]=l/a.numel[p];
    }
  }
  __syncthreads();
  if(!threadIdx.x) {
    // Preserve per-plane mean, then the original level/plane sum order.
    float spatial=smooth[0],ratio=smooth[2],one=l1[2];
    for(int p=1;p<a.count;++p)if(!ratioPlane(p))spatial+=smooth[p];
    for(int p=3;p<a.count;++p)if(ratioPlane(p)){ratio+=smooth[p];one+=l1[p];}
    loss[0]=(sw*spatial+rw*ratio)+lw*one;
  }
}
__global__ void regularizationGradKernel(Regularizer a,const float* upstream,float sw,float rw,float lw) {
  int p=blockIdx.y,index=blockIdx.x*256+threadIdx.x;
  if(index>=a.numel[p])return;
  int h=a.height[p],w=a.width[p],row=(index/w)%h;
  bool ratio=ratioPlane(p);float grad=0,value=a.data[p][index];
  if(h>=3) {
    float factor=(upstream[0]*(ratio?rw:sw))/(a.channels[p]*(h-2)*w);
    float prev=row>0?a.data[p][index-w]:0,prev2=row>1?a.data[p][index-2*w]:0;
    float next=row+1<h?a.data[p][index+w]:0,next2=row+2<h?a.data[p][index+2*w]:0;
    float g0=row+2<h?factor*(2*((next2-next)-(next-value))):0;
    float g1=row>0 && row+1<h?factor*(2*((next-value)-(value-prev))):0;
    float g2=row>1?factor*(2*((value-prev)-(prev-prev2))):0;
    grad=(row>0?g2-g1:0)-(row+1<h?g1-g0:0);
  }
  if(ratio) {
    float v=1.0f-value,sign=(v>0?1.0f:0.0f)-(v<0?1.0f:0.0f);
    grad+=-sign*((upstream[0]*lw)/a.numel[p]);
  }
  a.grad[p][index]=grad;
}
}  // namespace
torch::Tensor regularizationForward(const std::vector<torch::Tensor>& planes,
    float spatial,float ratio,float l1) {
  auto a=describe(planes);c10::cuda::CUDAGuard guard(planes[0].device());
  auto opt=planes[0].options().requires_grad(false);
  auto partial=torch::empty({a.count,a.blocks,2},opt),loss=torch::empty({},opt);
  regularizationPartialKernel<<<dim3(a.blocks,a.count),256,0,c10::cuda::getCurrentCUDAStream()>>>(a,partial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  regularizationReduceKernel<<<1,256,0,c10::cuda::getCurrentCUDAStream()>>>(a,partial.data_ptr<float>(),loss.data_ptr<float>(),spatial,ratio,l1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return loss;
}
std::vector<torch::Tensor> regularizationBackward(const std::vector<torch::Tensor>& planes,
    const torch::Tensor& gradient,float spatial,float ratio,float l1) {
  auto a=describe(planes);c10::cuda::CUDAGuard guard(planes[0].device());
  TORCH_CHECK(gradient.is_cuda() && gradient.device()==planes[0].device() && gradient.scalar_type()==torch::kFloat32 &&
              gradient.numel()==1,"Invalid regularization scalar gradient");
  std::vector<torch::Tensor> out;for(const auto& p:planes)out.push_back(torch::empty_like(p));
  for(int i=0;i<a.count;++i)a.grad[i]=out[i].data_ptr<float>();
  regularizationGradKernel<<<dim3(a.blocks,a.count),256,0,c10::cuda::getCurrentCUDAStream()>>>(a,gradient.data_ptr<float>(),spatial,ratio,l1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
}  // namespace tf_cuda

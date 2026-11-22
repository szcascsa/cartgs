#include "include/change_distill_transport_cuda.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>

namespace cd_transport_cuda {
namespace {
struct ReadState {
  const float *x, *s, *q;
  int64_t xs0, xs1, ss0, ss1, qs0, qs1;
};
// Float metadata rows: matrix[16], wxyz[4], scale, log(scale), flags.
// The reduction trees and explicit cross-product FMAs match LibTorch 2.3.1.
// Keep --fmad=false: only the original cross product uses explicit FMA.
__global__ void compactXYZKernel(const float* x, int64_t s0, int64_t s1,
    const int64_t* pos, const int64_t* groups, const float* meta, int64_t n, float* out) {
  const int64_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n*3) out[i]=x[pos[i/3]*s0+(i%3)*s1]*meta[groups[i/3]*23+20];
}
__global__ void scatterCompactKernel(ReadState in, float* x, float* s, float* q,
    const int64_t* pos, const int64_t* groups, const float* meta,
    const float* product, int64_t n) {
  const int64_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  const auto p=pos[i]; const auto* m=meta+groups[i]*23;
  const auto flags=static_cast<unsigned>(m[22]);
  if(flags&1) for(int j=0;j<3;++j) x[p*3+j]=product[i*3+j]+m[12+j];
  if(flags&2) for(int j=0;j<3;++j) s[p*3+j]=in.s[p*in.ss0+j*in.ss1]+m[21];
  if(flags&4) {
    const float a=m[16], b=m[17], c=m[18], d=m[19];
    const float w=in.q[p*in.qs0], u=in.q[p*in.qs0+in.qs1];
    const float v=in.q[p*in.qs0+2*in.qs1], z=in.q[p*in.qs0+3*in.qs1];
    const float o0=a*w-((b*u+d*z)+c*v);
    const float o1=(a*u+w*b)+__fmaf_rn(c,z,-(d*v));
    const float o2=(a*v+w*c)+__fmaf_rn(d,u,-(b*z));
    const float o3=(a*z+w*d)+__fmaf_rn(b,v,-(c*u));
    const float norm=fmaxf(sqrtf((o0*o0+o1*o1)+(o2*o2+o3*o3)),1e-12f);
    q[p*4]=o0/norm;q[p*4+1]=o1/norm;q[p*4+2]=o2/norm;q[p*4+3]=o3/norm;
  }
}
}
torch::Tensor compactXYZ(const torch::Tensor& x,const torch::Tensor& pos,
                         const torch::Tensor& groups,const torch::Tensor& meta) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto out=torch::empty({pos.numel(),3},x.options().requires_grad(false));
  if(pos.numel()) compactXYZKernel<<<(pos.numel()*3+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(),x.stride(0),x.stride(1),pos.data_ptr<int64_t>(),groups.data_ptr<int64_t>(),meta.data_ptr<float>(),pos.numel(),out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
void scatterCompact(const GaussianAttrState& input,const GaussianAttrState& output,
                    const torch::Tensor& pos,const torch::Tensor& groups,
                    const torch::Tensor& meta,const torch::Tensor& product) {
  const c10::cuda::CUDAGuard guard(input.xyz.device());
  const ReadState in{input.xyz.data_ptr<float>(),input.log_scaling.data_ptr<float>(),input.rotation.data_ptr<float>(),
      input.xyz.stride(0),input.xyz.stride(1),input.log_scaling.stride(0),input.log_scaling.stride(1),input.rotation.stride(0),input.rotation.stride(1)};
  if(pos.numel()) scatterCompactKernel<<<(pos.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      in,output.xyz.data_ptr<float>(),output.log_scaling.data_ptr<float>(),output.rotation.data_ptr<float>(),pos.data_ptr<int64_t>(),groups.data_ptr<int64_t>(),meta.data_ptr<float>(),product.data_ptr<float>(),pos.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}

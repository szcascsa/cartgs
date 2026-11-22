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
__global__ void gatherXYZ(const float* xyz, int64_t s0, int64_t s1,
                          const int64_t* positions, int64_t count, float scale,
                          float* output) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count * 3) output[i] = xyz[positions[i / 3] * s0 + (i % 3) * s1] * scale;
}
__global__ void postprocess(ReadState in, float* x, float* s, float* q,
                             const int64_t* positions, int64_t count,
                             const float* product, const float* matrix,
                             int64_t m0, int64_t m1, const float* rotation,
                             int64_t rs, float log_scale, unsigned flags) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int64_t p = positions[i];
  if (flags & 1)
    for (int j=0;j<3;++j) x[p*3+j] = product[i*3+j] + matrix[3*m0+j*m1];
  if (flags & 2)
    for (int j=0;j<3;++j) s[p*3+j] = in.s[p*in.ss0+j*in.ss1] + log_scale;
  if (flags & 4) {
    const float a=rotation[0], b=rotation[rs], c=rotation[2*rs], d=rotation[3*rs];
    const float w=in.q[p*in.qs0], u=in.q[p*in.qs0+in.qs1];
    const float v=in.q[p*in.qs0+2*in.qs1], z=in.q[p*in.qs0+3*in.qs1];
    const float o0=a*w-((b*u+d*z)+c*v);
    const float o1=(a*u+w*b)+(c*z-d*v);
    const float o2=(a*v+w*c)+(d*u-b*z);
    const float o3=(a*z+w*d)+(b*v-c*u);
    const float norm=fmaxf(sqrtf((o0*o0+o1*o1)+(o2*o2+o3*o3)),1e-12f);
    q[p*4]=o0/norm; q[p*4+1]=o1/norm; q[p*4+2]=o2/norm; q[p*4+3]=o3/norm;
  }
}
}

torch::Tensor gatherScaledXYZ(const torch::Tensor& xyz,
                             const torch::Tensor& positions, float scale) {
  const c10::cuda::CUDAGuard guard(xyz.device());
  auto out=torch::empty({positions.numel(),3},xyz.options().requires_grad(false));
  if (!positions.numel()) return out;
  gatherXYZ<<<(positions.numel()*3+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      xyz.data_ptr<float>(),xyz.stride(0),xyz.stride(1),positions.data_ptr<int64_t>(),
      positions.numel(),scale,out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
void scatter(const GaussianAttrState& input, const GaussianAttrState& output,
             const torch::Tensor& positions, const torch::Tensor& product,
             const GaussianTransformResult& correction) {
  if (!positions.numel()) return;
  const c10::cuda::CUDAGuard guard(input.xyz.device());
  const auto flags=correction.changed_attributes;
  const auto& matrix=correction.transform_4x4;
  const auto& rotation=correction.rotation_wxyz;
  const ReadState in{input.xyz.data_ptr<float>(),input.log_scaling.data_ptr<float>(),input.rotation.data_ptr<float>(),
      input.xyz.stride(0),input.xyz.stride(1),input.log_scaling.stride(0),input.log_scaling.stride(1),
      input.rotation.stride(0),input.rotation.stride(1)};
  postprocess<<<(positions.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      in,output.xyz.data_ptr<float>(),output.log_scaling.data_ptr<float>(),output.rotation.data_ptr<float>(),
      positions.data_ptr<int64_t>(),positions.numel(),(flags&1)?product.data_ptr<float>():nullptr,
      (flags&1)?matrix.data_ptr<float>():nullptr,(flags&1)?matrix.stride(0):0,(flags&1)?matrix.stride(1):0,
      (flags&4)?rotation.data_ptr<float>():nullptr,(flags&4)?rotation.stride(rotation.dim()-1):0,
      std::log(correction.scale),flags);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}

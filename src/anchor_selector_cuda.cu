#include "include/anchor_selector_cuda.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace mapping_cuda {
namespace {
constexpr float kSoftplusScaleShift=0.5413248546129181f;
constexpr float kScaleFloor=1.1754943508222875e-38f;
__device__ float scaleSoftplus(float x) {return x>20.f?x:log1pf(expf(x));}
__device__ float scaleSoftplusDerivative(float x) {
  if(x>20.f)return 1.f;
  const float e=expf(x);return e/(1.f+e);
}
__device__ void multiply(const float* a,const float* b,float* out) {
  // ATen sum over three contiguous columns uses two lanes: (0+2)+1.
  out[0]=a[0]*b[0]-((a[1]*b[1]+a[3]*b[3])+a[2]*b[2]);
  out[1]=(a[0]*b[1]+b[0]*a[1])+(a[2]*b[3]-a[3]*b[2]);
  out[2]=(a[0]*b[2]+b[0]*a[2])+(a[3]*b[1]-a[1]*b[3]);
  out[3]=(a[0]*b[3]+b[0]*a[3])+(a[1]*b[2]-a[2]*b[1]);
}
// Match ATen's four-lane reduction tree, rather than left-associated summation.
__device__ float norm4(const float* a) {return sqrtf((a[0]*a[0]+a[1]*a[1])+(a[2]*a[2]+a[3]*a[3]));}
__device__ void normalize4(const float* a,float* y) {
  const float d=fmaxf(norm4(a),1e-12f);for(int j=0;j<4;++j)y[j]=a[j]/d;
}
__device__ void normVjp(const float* x,const float* g,float* out) {
  const float n=norm4(x),d=fmaxf(n,1e-12f);
  // Legacy sqrt(sum(x*x)) has NaN VJP at an exactly zero squared norm.
  if(n==0){for(int j=0;j<4;++j)out[j]=nanf("");return;}
  float dot=0;for(int j=0;j<4;++j)dot+=g[j]*x[j];
  const float norm_grad=n>=1e-12f?-dot/(d*d):0;
  const float square_grad=norm_grad/(2*n);
  for(int j=0;j<4;++j)out[j]=g[j]/d+(square_grad*x[j]+square_grad*x[j]);
}
template<int ScaleMode>
__global__ void decodeForward(int64_t n,const float* x,const float* scale,const float* q,
    const float* logits,const float* dc,const float* delta,const float* mask,
    float* ox,float* os,float* oq,float* oo,float* odc) {
  const int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  const float* d=delta+14*i;
  for(int j=0;j<3;++j){ox[3*i+j]=x[3*i+j]+d[j];
    if(ScaleMode==3) {
      const float value=expf(scale[3*i+j])*scaleSoftplus(d[3+j]+kSoftplusScaleShift);
      // Keep NaN visible; fmaxf alone would silently replace it with the floor.
      os[3*i+j]=value<kScaleFloor?kScaleFloor:value;
    }else os[3*i+j]=ScaleMode==2?expf(scale[3*i+j])+d[3+j]:(ScaleMode==1?expf(scale[3*i+j]+d[3+j]):scale[3*i+j]*expf(d[3+j]));
    odc[3*i+j]=dc[3*i+j]+d[11+j];}
  float raw[4],a[4],product[4];for(int j=0;j<4;++j)raw[j]=d[6+j]+(j==0?1.f:0.f);
  normalize4(raw,a);multiply(a,q+4*i,product);normalize4(product,oq+4*i);
  const float s=1.f/(1.f+expf(-(logits[i]+d[10])));oo[i]=mask[i]*s;
}
template<int ScaleMode>
__global__ void decodeBackward(int64_t n,const float* scale,const float* q,const float* logits,
    const float* delta,const float* mask,const float* gx,const float* gs,const float* gq,const float* go,const float* gdc,
    float* dx,float* ds,float* dq,float* dl,float* ddc,float* dd,float* dm) {
  const int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  const float* d=delta+14*i;float* out=dd+14*i;for(int j=0;j<14;++j)out[j]=0;
  for(int j=0;j<3;++j){
    if(gx){dx[3*i+j]=gx[3*i+j];out[j]=gx[3*i+j];}
    if(gs){
      if(ScaleMode==3) {
        const float base=expf(scale[3*i+j]),t=d[3+j]+kSoftplusScaleShift;
        const float value=base*scaleSoftplus(t);
        if(value<kScaleFloor){ds[3*i+j]=0.f;out[3+j]=0.f;}
        else {ds[3*i+j]=gs[3*i+j]*value;out[3+j]=(gs[3*i+j]*base)*scaleSoftplusDerivative(t);}
      }
      else if(ScaleMode==2){ds[3*i+j]=gs[3*i+j]*expf(scale[3*i+j]);out[3+j]=gs[3*i+j];}
      else if(ScaleMode==1){const float v=gs[3*i+j]*expf(scale[3*i+j]+d[3+j]);ds[3*i+j]=v;out[3+j]=v;}
      else {const float e=expf(d[3+j]);ds[3*i+j]=gs[3*i+j]*e;out[3+j]=(gs[3*i+j]*scale[3*i+j])*e;}
    }
    if(gdc){ddc[3*i+j]=gdc[3*i+j];out[11+j]=gdc[3*i+j];}
  }
  if(gq){
    float raw[4],a[4],prod[4],gp[4],ga[4];for(int j=0;j<4;++j)raw[j]=d[6+j]+(j==0?1.f:0.f);
    normalize4(raw,a);const float* b=q+4*i;multiply(a,b,prod);normVjp(prod,gq+4*i,gp);
    ga[0]=gp[0]*b[0]+gp[1]*b[1]+gp[2]*b[2]+gp[3]*b[3];
    ga[1]=-gp[0]*b[1]+b[0]*gp[1]+(b[2]*gp[3]-b[3]*gp[2]);
    ga[2]=-gp[0]*b[2]+b[0]*gp[2]+(b[3]*gp[1]-b[1]*gp[3]);
    ga[3]=-gp[0]*b[3]+b[0]*gp[3]+(b[1]*gp[2]-b[2]*gp[1]);
    dq[4*i]=gp[0]*a[0]+gp[1]*a[1]+gp[2]*a[2]+gp[3]*a[3];
    dq[4*i+1]=-gp[0]*a[1]+a[0]*gp[1]+(gp[2]*a[3]-gp[3]*a[2]);
    dq[4*i+2]=-gp[0]*a[2]+a[0]*gp[2]+(gp[3]*a[1]-gp[1]*a[3]);
    dq[4*i+3]=-gp[0]*a[3]+a[0]*gp[3]+(gp[1]*a[2]-gp[2]*a[1]);
    normVjp(raw,ga,out+6);
  }
  if(go){const float s=1.f/(1.f+expf(-(logits[i]+d[10])));
    dl[i]=((go[i]*mask[i])*(1-s))*s;out[10]=dl[i];dm[i]=go[i]*s;}
}
void validate(const std::vector<torch::Tensor>& a) {
  TORCH_CHECK(a.size()==7,"Decode14 input count");const auto n=a[0].size(0);
  for(const auto& t:a)TORCH_CHECK(t.is_cuda()&&t.scalar_type()==torch::kFloat32&&t.device()==a[0].device()&&t.is_contiguous()&&t.size(0)==n,"Decode14 requires row-aligned CUDA FP32 inputs");
  TORCH_CHECK(a[0].sizes()==torch::IntArrayRef({n,3})&&a[1].sizes()==a[0].sizes()&&a[2].sizes()==torch::IntArrayRef({n,4})&&
      a[3].sizes()==torch::IntArrayRef({n,1})&&a[4].sizes()==torch::IntArrayRef({n,1,3})&&
      a[5].sizes()==torch::IntArrayRef({n,14})&&a[6].sizes()==torch::IntArrayRef({n}),"Decode14 attribute shapes");
}
const float* ptr(const torch::Tensor& t){return t.defined()?t.data_ptr<float>():nullptr;}
float* outptr(torch::Tensor& t){return t.defined()?t.data_ptr<float>():nullptr;}
}
template<int ScaleMode>
std::vector<torch::Tensor> decodeForwardImpl(const std::vector<torch::Tensor>& a) {
  validate(a);c10::cuda::CUDAGuard guard(a[0].device());const auto n=a[0].size(0);
  std::vector<torch::Tensor> out;for(int i=0;i<5;++i)out.push_back(torch::empty_like(a[i]));
  if(n){decodeForward<ScaleMode><<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(n,ptr(a[0]),ptr(a[1]),ptr(a[2]),ptr(a[3]),ptr(a[4]),ptr(a[5]),ptr(a[6]),
      outptr(out[0]),outptr(out[1]),outptr(out[2]),outptr(out[3]),outptr(out[4]));C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return out;
}
template<int ScaleMode>
std::vector<torch::Tensor> decodeBackwardImpl(const std::vector<torch::Tensor>& a,const std::vector<torch::Tensor>& g) {
  validate(a);TORCH_CHECK(g.size()==5,"Decode14 gradient count");c10::cuda::CUDAGuard guard(a[0].device());const auto n=a[0].size(0);
  std::vector<torch::Tensor> out(7);
  for(int i=0;i<5;++i)if(g[i].defined())out[i]=torch::empty_like(a[i]);
  out[5]=torch::empty_like(a[5]);if(g[3].defined())out[6]=torch::empty_like(a[6]);
  if(n){decodeBackward<ScaleMode><<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(n,ptr(a[1]),ptr(a[2]),ptr(a[3]),ptr(a[5]),ptr(a[6]),ptr(g[0]),ptr(g[1]),ptr(g[2]),ptr(g[3]),ptr(g[4]),
      outptr(out[0]),outptr(out[1]),outptr(out[2]),outptr(out[3]),outptr(out[4]),outptr(out[5]),outptr(out[6]));C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return out;
}
std::vector<torch::Tensor> decode14Forward(const std::vector<torch::Tensor>& a){return decodeForwardImpl<0>(a);}
std::vector<torch::Tensor> decode14Backward(const std::vector<torch::Tensor>& a,const std::vector<torch::Tensor>& g){return decodeBackwardImpl<0>(a,g);}
std::vector<torch::Tensor> decode14LogScaleForward(const std::vector<torch::Tensor>& a){return decodeForwardImpl<1>(a);}
std::vector<torch::Tensor> decode14LogScaleBackward(const std::vector<torch::Tensor>& a,const std::vector<torch::Tensor>& g){return decodeBackwardImpl<1>(a,g);}
}

namespace mapping_cuda {
std::vector<torch::Tensor> decode14AdditiveScaleForward(const std::vector<torch::Tensor>& a){return decodeForwardImpl<2>(a);}
std::vector<torch::Tensor> decode14AdditiveScaleBackward(const std::vector<torch::Tensor>& a,const std::vector<torch::Tensor>& g){return decodeBackwardImpl<2>(a,g);}
std::vector<torch::Tensor> decode14SoftplusScaleForward(const std::vector<torch::Tensor>& a){return decodeForwardImpl<3>(a);}
std::vector<torch::Tensor> decode14SoftplusScaleBackward(const std::vector<torch::Tensor>& a,const std::vector<torch::Tensor>& g){return decodeBackwardImpl<3>(a,g);}
}

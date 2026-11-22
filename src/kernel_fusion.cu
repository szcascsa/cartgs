#include "include/kernel_fusion.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
namespace mapping_cuda {
namespace {
const float* p(const torch::Tensor& t){return t.data_ptr<float>();}
void check(const std::vector<torch::Tensor>& a){
  for(const auto& t:a)TORCH_CHECK(t.is_cuda()&&t.scalar_type()==torch::kFloat32&&t.is_contiguous()&&t.device()==a[0].device(),"Fusion requires contiguous same-device FP32");
}
__global__ void anchorPack(int64_t n,const float* x,const float* c,const float* r,const float* f,
    const float* gate,const float* mean,const float* std,float* out,float* saved) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n*131)return;
  int64_t row=i/131,col=i%131;
  if(col<3) {
    float q=__fdiv_rn(__fsub_rn(x[row*3+col],c[row*3+col]),r[row]);
    float v=__fdiv_rn(__fsub_rn(asinhf(q),mean[3+col]),std[3+col]);
    out[i]=v<-15.f?-15.f:(v>15.f?15.f:v);
    saved[row*6+col]=q;saved[row*6+3+col]=v;
  }else out[i]=__fmul_rn(f[row*128+col-3],__fadd_rn(1.f,gate[col-3]));
}
__global__ void anchorPackVjp(int64_t n,const float* r,const float* f,const float* gate,
    const float* std,const float* saved,const float* g,float* gx,float* gf,float* gg) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n*131)return;
  int64_t row=i/131,col=i%131;
  if(col<3) {
    float q=saved[row*6+col],v=saved[row*6+3+col];
    float d=__fdiv_rn(v>=-15.f&&v<=15.f?g[i]:0.f,std[3+col]);
    d=__fmul_rn(d,rsqrtf(__fadd_rn(__fmul_rn(q,q),1.f)));
    gx[row*3+col]=__fdiv_rn(d,r[row]);
  }else {
    int64_t j=row*128+col-3;
    gf[j]=__fmul_rn(g[i],__fadd_rn(1.f,gate[col-3]));
    gg[j]=__fmul_rn(g[i],f[j]);
  }
}
__global__ void maturityPrep(int64_t n,int64_t stride,const float* w,const float* m,const bool* e,
    float* weights,float* weighted,float* mat,float* allowed) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  float a=e[i]?1.f:0.f,v=__fmul_rn(w[i*stride],a);
  allowed[i]=a;weights[i]=v;weighted[i]=__fmul_rn(v,m[i]);mat[i]=__fmul_rn(m[i],a);
}
__global__ void maturityEnd(const float* pool_num,const float* pool_den,const float* num,
    const float* den,float delta,float* out,float* saved) {
  float pd=*pool_den<1.f?1.f:*pool_den;
  float d=*den<1e-8f?1e-8f:*den;
  float pool=__fdiv_rn(*pool_num,pd),target=__fadd_rn(pool,delta);if(target>1.f)target=1.f;
  float relu=__fsub_rn(target,__fdiv_rn(*num,d));if(relu<0.f)relu=0.f;
  *out=__fmul_rn(relu,relu);
  saved[0]=*num;saved[1]=d;saved[2]=*den;saved[3]=relu;
}
__global__ void maturityVjp(int64_t n,const float* m,const bool* e,const float* saved,const float* g,float* out) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  float num=saved[0],den=saved[1],rawden=saved[2],relu=saved[3];
  float grad=__fmul_rn(*g,__fmul_rn(2.f,relu));if(relu<=0.f)grad=0.f;grad=-grad;
  float gn=__fdiv_rn(grad,den);
  float gd=__fmul_rn(-grad,__fdiv_rn(__fdiv_rn(num,den),den));
  if(!(rawden>=1e-8f))gd=0.f;
  out[i]=__fmul_rn(__fadd_rn(__fmul_rn(gn,m[i]),gd),e[i]?1.f:0.f);
}
__global__ void l1Pack(int64_t n,const float* x,const float* y,float* absolute,float* difference) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  float d=__fsub_rn(x[i],y[i]);difference[i]=d;absolute[i]=fabsf(d);
}
__global__ void l1Backward(int64_t n,const float* d,const float* g,float reciprocal,float* gx,float* gy) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  float sign=float(d[i]>0)-float(d[i]<0);
  gx[i]=__fmul_rn(__fmul_rn(*g,reciprocal),sign);if(gy)gy[i]=-gx[i];
}
__global__ void statsAdd(double* sums,bool first,int64_t selected,int64_t candidates,int64_t gi_count,int64_t selector_candidates,int64_t population) {
  int i=threadIdx.x;if(i>=9)return;
  double value=i==0?double(selected):(i==2?double(candidates):(i==4?double(gi_count):(i==6?double(selector_candidates):(i==7?double(population):0.))));
  sums[i]=__dadd_rn(first?0.:sums[i],value);
}
}
std::vector<torch::Tensor> anchor128PackForward(const std::vector<torch::Tensor>& a) {
  check(a);const auto n=a[0].size(0);
  TORCH_CHECK(a[0].sizes()==torch::IntArrayRef({n,3})&&a[1].sizes()==a[0].sizes()&&a[2].numel()==n&&
      a[3].sizes()==torch::IntArrayRef({n,128})&&a[4].sizes()==torch::IntArrayRef({1,128})&&a[5].numel()==19&&a[6].numel()==19,"Anchor128 pack shape");
  c10::cuda::CUDAGuard guard(a[0].device());auto out=torch::empty({n,131},a[0].options()),saved=torch::empty({n,6},a[0].options());
  if(n){anchorPack<<<(n*131+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,p(a[0]),p(a[1]),p(a[2]),p(a[3]),p(a[4]),p(a[5]),p(a[6]),out.data_ptr<float>(),saved.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return {out,saved};
}
std::vector<torch::Tensor> anchor128PackBackward(const std::vector<torch::Tensor>& a,const torch::Tensor& saved,const torch::Tensor& g) {
  const auto n=a[0].size(0);c10::cuda::CUDAGuard guard(a[0].device());
  auto gx=torch::empty_like(a[0]),gf=torch::empty_like(a[3]),gg=torch::empty_like(a[3]);
  if(n){anchorPackVjp<<<(n*131+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,p(a[2]),p(a[3]),p(a[4]),p(a[6]),p(saved),p(g),gx.data_ptr<float>(),gf.data_ptr<float>(),gg.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return {gx,gf,gg};
}
std::vector<torch::Tensor> maturityPrepare(const torch::Tensor& w,const torch::Tensor& m,const torch::Tensor& e) {
  TORCH_CHECK(w.is_cuda()&&w.scalar_type()==torch::kFloat32&&w.dim()==1&&m.sizes()==w.sizes()&&
      m.is_cuda()&&m.scalar_type()==torch::kFloat32&&m.is_contiguous()&&e.scalar_type()==torch::kBool&&e.is_contiguous()&&e.sizes()==w.sizes()&&e.device()==w.device()&&m.device()==w.device(),"Maturity fusion inputs");
  c10::cuda::CUDAGuard guard(w.device());std::vector<torch::Tensor> v;for(int i=0;i<4;++i)v.push_back(torch::empty({w.numel()},w.options()));
  if(w.numel()){maturityPrep<<<(w.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(w.numel(),w.stride(0),p(w),p(m),e.data_ptr<bool>(),v[0].data_ptr<float>(),v[1].data_ptr<float>(),v[2].data_ptr<float>(),v[3].data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return v;
}
std::vector<torch::Tensor> maturityFinish(const std::vector<torch::Tensor>& v,float delta) {
  c10::cuda::CUDAGuard guard(v[0].device());auto loss=torch::empty({},v[0].options()),saved=torch::empty({4},v[0].options());
  maturityEnd<<<1,1,0,c10::cuda::getCurrentCUDAStream()>>>(p(v[0]),p(v[1]),p(v[2]),p(v[3]),delta,loss.data_ptr<float>(),saved.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();return {loss,saved};
}
torch::Tensor maturityLossVjp(const torch::Tensor& m,const torch::Tensor& e,const torch::Tensor& saved,const torch::Tensor& g) {
  c10::cuda::CUDAGuard guard(m.device());auto out=torch::empty_like(m);
  if(m.numel()){maturityVjp<<<(m.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(m.numel(),p(m),e.data_ptr<bool>(),p(saved),p(g),out.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}return out;
}
std::vector<torch::Tensor> l1Prepare(const torch::Tensor& x,const torch::Tensor& y) {
  TORCH_CHECK(x.is_cuda()&&x.scalar_type()==torch::kFloat32&&y.scalar_type()==torch::kFloat32&&x.sizes()==y.sizes()&&x.device()==y.device(),"L1 fusion inputs");
  c10::cuda::CUDAGuard guard(x.device());auto a=torch::empty_like(x),d=torch::empty_like(x);
  if(x.numel()){l1Pack<<<(x.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(x.numel(),p(x),p(y),a.data_ptr<float>(),d.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}return {a,d};
}
std::vector<torch::Tensor> l1Vjp(const torch::Tensor& d,const torch::Tensor& g,bool target_gradient) {
  c10::cuda::CUDAGuard guard(d.device());auto gx=torch::empty_like(d),gy=target_gradient?torch::empty_like(d):torch::Tensor();
  if(d.numel()){l1Backward<<<(d.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(d.numel(),p(d),p(g),float(1./double(d.numel())),gx.data_ptr<float>(),gy.defined()?gy.data_ptr<float>():nullptr);C10_CUDA_KERNEL_LAUNCH_CHECK();}return {gx,gy};
}
void addZeroTfStatistics(torch::Tensor& sums,const torch::TensorOptions& options,int64_t selected,int64_t candidates,
    int64_t gi_count,int64_t selector_candidates,int64_t population) {
  bool first=!sums.defined();if(first)sums=torch::empty({9},options.dtype(torch::kDouble));
  c10::cuda::CUDAGuard guard(sums.device());statsAdd<<<1,32,0,c10::cuda::getCurrentCUDAStream()>>>(sums.data_ptr<double>(),first,selected,candidates,gi_count,selector_candidates,population);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}

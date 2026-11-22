#include "include/transform_field_cuda.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>

namespace tf_cuda {
namespace {
constexpr int kMaxPlanes = 48;  // Up to eight levels; includes fast and default.
struct Planes {
  const float* p[kMaxPlanes];
  float* g[kMaxPlanes];
  int height[kMaxPlanes], width[kMaxPlanes];
  int count, channels;
};
struct Inputs {
  const float *xyz, *ratio, *lo, *hi;
  int64_t xs0, xs1, rs0;
  bool scalar_ratio;
};

void check(const torch::Tensor& t, const torch::Tensor& reference) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 &&
                  t.device() == reference.device(), "TF CUDA requires same-device CUDA FP32 tensors");
}
Planes describe(const std::vector<torch::Tensor>& in,
                const std::vector<torch::Tensor>* grads = nullptr) {
  TORCH_CHECK(in.size() > 4 && (in.size() - 4) % 6 == 0 &&
                  in.size() - 4 <= kMaxPlanes, "TF CUDA requires 6 planes per level, at most 8 levels");
  const auto& x=in[0];
  TORCH_CHECK(x.dim()==2 && x.size(1)==3 && x.size(0)<=INT_MAX, "TF XYZ must be [N,3]");
  for(const auto& t:in) check(t,x);
  TORCH_CHECK(in[2].numel()==3 && in[3].numel()==3 && in[2].is_contiguous() &&
                  in[3].is_contiguous(), "TF AABB must be contiguous [3]");
  TORCH_CHECK(in[1].numel()==1 || (in[1].numel()==x.size(0) && in[1].dim()>=1 &&
                  in[1].dim()<=2 && (in[1].dim()==1 || in[1].size(1)==1)),
              "TF ratio must be scalar, [N] or [N,1]");
  TORCH_CHECK(in[4].dim()==4 && in[4].size(1)>0,"Invalid first TF plane");
  Planes pack{};pack.count=in.size()-4;pack.channels=in[4].size(1);
  TORCH_CHECK(x.size(0)*pack.channels*(pack.count/6)<=INT_MAX,"TF query indexing overflow");
  for(int i=0;i<pack.count;++i) {
    const auto& p=in[i+4];
    TORCH_CHECK(p.dim()==4 && p.size(0)==1 && p.size(1)==pack.channels &&
                    p.size(2)>=2 && p.size(3)>=2 && p.numel()<=INT_MAX &&
                    p.is_contiguous(), "Invalid TF CUDA plane layout");
    pack.p[i]=p.data_ptr<float>();pack.height[i]=p.size(2);pack.width[i]=p.size(3);
    if(grads)pack.g[i]=grads->at(i+4).data_ptr<float>();
  }
  return pack;
}
Inputs describeInputs(const std::vector<torch::Tensor>& in) {
  return {in[0].data_ptr<float>(),in[1].data_ptr<float>(),in[2].data_ptr<float>(),
          in[3].data_ptr<float>(),in[0].stride(0),in[0].stride(1),
          in[1].dim()?in[1].stride(0):0,in[1].numel()==1};
}
__device__ void coordinates(Inputs a,int n,float* c) {
  for(int j=0;j<3;++j)
    c[j]=((2.0f*(a.xyz[n*a.xs0+j*a.xs1]-a.lo[j]))/(a.hi[j]-a.lo[j]))-1.0f;
  c[3]=a.ratio[a.scalar_ratio?0:n*a.rs0];
}
__device__ int axisX(int p) { return p<3?0:(p<5?1:2); }
__device__ int axisY(int p) { return p==0?1:(p==1||p==3?2:3); }
struct Sample {
  int ix[4];float weight[4],z[4],value,tx,ty;bool clip_x,clip_y;
};
__device__ Sample sample(const Planes& a,int p,int ch,float u,float v) {
  const int w=a.width[p],h=a.height[p];
  const float raw_x=((u+1.0f)/2.0f)*(w-1),raw_y=((v+1.0f)/2.0f)*(h-1);
  const float x=fminf(w-1,fmaxf(0.0f,raw_x)),y=fminf(h-1,fmaxf(0.0f,raw_y));
  const int x0=static_cast<int>(floorf(x)),y0=static_cast<int>(floorf(y));
  const int x1=x0+1,y1=y0+1;
  const float tx=x-x0,ty=y-y0;
  Sample s{};
  s.weight[0]=(x1-x)*(y1-y);s.weight[1]=(x-x0)*(y1-y);
  s.weight[2]=(x1-x)*(y-y0);s.weight[3]=(x-x0)*(y-y0);
  const int xx[4]={x0,x1,x0,x1},yy[4]={y0,y0,y1,y1};
  float z[4];
  for(int i=0;i<4;++i) {
    s.ix[i]=(xx[i]<w && yy[i]<h)?(ch*h+yy[i])*w+xx[i]:-1;
    z[i]=s.ix[i]>=0?a.p[p][s.ix[i]]:0.0f;
    s.z[i]=z[i];s.value=fmaf(z[i],s.weight[i],s.value);
  }
  // Exactly clipped endpoints have zero derivative in ATen GridSampler.h.
  s.tx=tx;s.ty=ty;s.clip_x=raw_x<=0 || raw_x>=w-1;s.clip_y=raw_y<=0 || raw_y>=h-1;
  return s;
}
__global__ void queryKernel(Inputs in,Planes p,float* out,int n) {
  int index=blockIdx.x*blockDim.x+threadIdx.x;
  int f=p.channels*(p.count/6);
  if(index>=n*f)return;
  int row=index/f,k=index%f,level=k/p.channels,ch=k%p.channels;
  float c[4];coordinates(in,row,c);
  float value=1.0f;
  for(int j=0;j<6;++j)value*=sample(p,level*6+j,ch,c[axisX(j)],c[axisY(j)]).value;
  out[index]=value;
}
__global__ void queryGradKernel(Inputs in,Planes p,const float* df,float* dx,
                                float* dr,int n) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=n)return;
  float c[4],g[4]={0,0,0,0};coordinates(in,row,c);
  const int f=p.channels*(p.count/6);
  for(int level=p.count/6-1;level>=0;--level) {
    float gx[6]={},gy[6]={};bool clip_x[6],clip_y[6];
    for(int ch=0;ch<p.channels;++ch) {
      Sample s[6];float prefix[7];prefix[0]=1.0f;
      for(int j=0;j<6;++j) {
        s[j]=sample(p,level*6+j,ch,c[axisX(j)],c[axisY(j)]);
        prefix[j+1]=prefix[j]*s[j].value;
      }
      float upstream=df[row*f+level*p.channels+ch];
      for(int plane=5;plane>=0;--plane) {
        float gs=upstream*prefix[plane];upstream*=s[plane].value;
        const auto& target=s[plane];
        for(int a=0;a<4;++a)if(target.ix[a]>=0)
          atomicAdd(p.g[level*6+plane]+target.ix[a],gs*target.weight[a]);
        // Corner-by-corner accumulation preserves cancellation on ratio planes.
        float tx=target.tx,ty=target.ty;
        gx[plane]=fmaf(-target.z[0]*(1-ty),gs,gx[plane]);gy[plane]=fmaf(-target.z[0]*(1-tx),gs,gy[plane]);
        gx[plane]=fmaf(target.z[1]*(1-ty),gs,gx[plane]);gy[plane]=fmaf(-target.z[1]*tx,gs,gy[plane]);
        gx[plane]=fmaf(-target.z[2]*ty,gs,gx[plane]);gy[plane]=fmaf(target.z[2]*(1-tx),gs,gy[plane]);
        gx[plane]=fmaf(target.z[3]*ty,gs,gx[plane]);gy[plane]=fmaf(target.z[3]*tx,gs,gy[plane]);
        clip_x[plane]=target.clip_x;clip_y[plane]=target.clip_y;
      }
    }
    for(int plane=5;plane>=0;--plane) {
      int pi=level*6+plane;
      g[axisX(plane)]+=clip_x[plane]?0.0f:gx[plane]*((p.width[pi]-1)/2.0f);
      g[axisY(plane)]+=clip_y[plane]?0.0f:gy[plane]*((p.height[pi]-1)/2.0f);
    }
  }
  for(int j=0;j<3;++j)dx[row*3+j]=(g[j]/(in.hi[j]-in.lo[j]))*2.0f;
  if(in.scalar_ratio)atomicAdd(dr,g[3]);else dr[row]=g[3];
}
}  // namespace

torch::Tensor queryForward(const std::vector<torch::Tensor>& inputs) {
  auto pack=describe(inputs);c10::cuda::CUDAGuard guard(inputs[0].device());
  const int n=inputs[0].size(0),f=pack.channels*(pack.count/6);
  auto out=torch::empty({n,f},inputs[0].options().requires_grad(false));
  if(n) {
    queryKernel<<<(n*f+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(describeInputs(inputs),pack,out.data_ptr<float>(),n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}
std::vector<torch::Tensor> queryBackward(const std::vector<torch::Tensor>& inputs,
                                        const torch::Tensor& gradient) {
  describe(inputs);check(gradient,inputs[0]);c10::cuda::CUDAGuard guard(inputs[0].device());
  std::vector<torch::Tensor> result(inputs.size());
  result[0]=torch::empty({inputs[0].size(0),3},inputs[0].options().requires_grad(false));
  result[1]=torch::zeros_like(inputs[1],inputs[1].options().requires_grad(false),torch::MemoryFormat::Contiguous);
  for(size_t i=4;i<inputs.size();++i)result[i]=torch::zeros_like(inputs[i]);
  auto pack=describe(inputs,&result);const int n=inputs[0].size(0);
  TORCH_CHECK(gradient.is_contiguous() && gradient.dim()==2 && gradient.size(0)==n &&
                  gradient.size(1)==pack.channels*(pack.count/6),"Invalid feature gradient");
  if(n) {
    queryGradKernel<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(describeInputs(inputs),pack,gradient.data_ptr<float>(),result[0].data_ptr<float>(),result[1].data_ptr<float>(),n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return result;
}
}  // namespace tf_cuda

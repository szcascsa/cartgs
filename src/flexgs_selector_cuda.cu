#include "include/flexgs_selector_cuda.h"
#include <cub/cub.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <climits>
#include <map>
#include <string>

namespace mapping_cuda {
namespace {
void fp32(const torch::Tensor& t,const torch::Tensor& ref) {
  TORCH_CHECK(t.is_cuda()&&t.device()==ref.device()&&t.scalar_type()==torch::kFloat32&&t.is_contiguous(),
              "FlexGS CUDA requires contiguous same-device FP32 tensors");
}
const float* ptr(const torch::Tensor& t){return t.defined()?t.data_ptr<float>():nullptr;}
__global__ void input11(int n,const float* x,const float* s,const float* dc,const float* op,
    const int64_t* owner,const float* c,const float* ref,float logratio,float* out) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  auto a=owner[i];for(int j=0;j<3;++j){out[i*11+j]=(x[i*3+j]-c[a*3+j])/ref[a];
    out[i*11+3+j]=s[i*3+j];out[i*11+6+j]=dc[i*3+j];}
  out[i*11+9]=op[i];out[i*11+10]=logratio;
}
__global__ void inputBackward(int n,const float* g,const int64_t* owner,const float* ref,
    float* x,float* s,float* dc,float* op) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  for(int j=0;j<3;++j){x[i*3+j]=g[i*11+j]/ref[owner[i]];
    s[i*3+j]=g[i*11+3+j];dc[i*3+j]=g[i*11+6+j];}op[i]=g[i*11+9];
}
struct Weights {const float *w0,*b0,*w1,*b1,*w2,*b2;};
template<int Width>
__global__ void network(int n,const float* x,Weights p,float* h0,float* h1,float* logits) {
  int lane=threadIdx.x%32,row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  if(row>=n)return;
  float a=0;for(int j=0;j<Width;++j)a=fmaf(x[row*Width+j],p.w0[lane*Width+j],a);
  a=fmaxf(a+p.b0[lane],0.f);h0[row*32+lane]=a;
  float b=0;for(int j=0;j<32;++j)b=fmaf(__shfl_sync(0xffffffff,a,j),p.w1[lane*32+j],b);
  b=fmaxf(b+p.b1[lane],0.f);h1[row*32+lane]=b;
  // A fixed warp reduction of both output logits; no per-row atomics.
  float u=b*p.w2[lane],v=b*p.w2[32+lane];
  for(int d=16;d;d/=2){u+=__shfl_down_sync(0xffffffff,u,d);v+=__shfl_down_sync(0xffffffff,v,d);}
  if(!lane){logits[row*2]=u+p.b2[0];logits[row*2+1]=v+p.b2[1];}
}
template<int Width>
__global__ void tiledNetwork(int n,const float* x,Weights p,float* h0,float* h1,float* z) {
  __shared__ float w0[Width][33],w1[32][33],w2[2][32],input[8][Width+1],hidden[8][33],next[8][33];
  int tid=threadIdx.x,row=blockIdx.x*8+tid/32,col=tid%32;
  for(int i=tid;i<32*Width;i+=256)w0[i%Width][i/Width]=p.w0[i];
  for(int i=tid;i<1024;i+=256)w1[i%32][i/32]=p.w1[i];
  if(tid<64)w2[tid/32][tid%32]=p.w2[tid];
  for(int i=tid;i<8*Width;i+=256){int r=blockIdx.x*8+i/Width;input[i/Width][i%Width]=r<n?x[r*Width+i%Width]:0.f;}
  __syncthreads();
  float a=0.f;
  #pragma unroll
  for(int j=0;j<Width;++j)a=fmaf(input[tid/32][j],w0[j][col],a);
  a=fmaxf(a+p.b0[col],0.f);hidden[tid/32][col]=a;
  if(row<n)h0[row*32+col]=a;
  __syncthreads();
  float b=0.f;
  #pragma unroll
  for(int j=0;j<32;++j)b=fmaf(hidden[tid/32][j],w1[j][col],b);
  b=fmaxf(b+p.b1[col],0.f);next[tid/32][col]=b;
  if(row<n)h1[row*32+col]=b;
  __syncthreads();
  if(col<2&&row<n){float v=0.f;
    #pragma unroll
    for(int j=0;j<32;++j)v=fmaf(next[tid/32][j],w2[col][j],v);
    z[row*2+col]=v+p.b2[col];}
}
// Only scratch that is not saved by Autograd is reused. Outputs and saved
// activations remain private to each forward, including outstanding graphs.
struct CompactWorkspace {
  torch::Tensor flags,count,temp,host;
  int last_n=-1;
  size_t bytes=0;
};
CompactWorkspace& compactWorkspace(const torch::Tensor& z,int n) {
  using Key=std::pair<int,uintptr_t>;
  thread_local std::map<Key,CompactWorkspace> cache;
  auto stream=c10::cuda::getCurrentCUDAStream();
  Key key{z.get_device(),reinterpret_cast<uintptr_t>(stream.stream())};
  if(cache.find(key)==cache.end()&&cache.size()>=8)cache.erase(cache.begin());
  auto& value=cache[key];
  if(!value.flags.defined()||value.flags.numel()<n)
    value.flags=torch::empty({n},z.options().dtype(torch::kBool));
  if(!value.count.defined()) {
    value.count=torch::empty({},z.options().dtype(torch::kInt32));
    value.host=torch::empty({},torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true));
  }
  return value;
}
__global__ void finalVjp(int count,const float* g,const float* w,const float* h,float* out) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){int row=i/32,j=i%32;
    out[i]=h[i]>0?g[row*2]*w[j]+g[row*2+1]*w[32+j]:0.f;}
}
__global__ void reluAdd(int n,const float* h,const float* extra,float* g) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)g[i]=h[i]>0?g[i]+(extra?extra[i]:0.f):0.f;
}
__global__ void sample(int n,const float* z,const float* e,const bool* eligible,float tau,
    float* probability,float* mask,float* soft,float* plain,bool* flags) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  float a=z[i*2]/tau,b=z[i*2+1]/tau,m=fmaxf(a,b);
  float u=expf(a-m),v=expf(b-m),den=u+v;plain[i*2]=u/den;plain[i*2+1]=v/den;
  a=(z[i*2]-logf(e[i*2]))/tau;b=(z[i*2+1]-logf(e[i*2+1]))/tau;m=fmaxf(a,b);
  u=expf(a-m);v=expf(b-m);den=u+v;u/=den;v/=den;
  soft[i*2]=u;soft[i*2+1]=v;
  bool allowed=eligible[i];probability[i]=plain[i*2+1]*float(allowed);
  mask[i]=float(v>u)*float(allowed);flags[i]=mask[i]>0.f;
}
__global__ void sampleVjp(int n,const float* s,const float* p,const bool* eligible,
    const float* gp,const float* gm,float tau,float* out) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  float a=(gm?gm[i]:0.f)*float(eligible[i]),b=(gp?gp[i]:0.f)*float(eligible[i]);
  float dot=a*s[2*i+1],plain_dot=b*p[2*i+1];
  out[2*i]=((0-dot)*s[2*i]+(0-plain_dot)*p[2*i])/tau;
  out[2*i+1]=((a-dot)*s[2*i+1]+(b-plain_dot)*p[2*i+1])/tau;
}
}
SelectorExecutionPlan SelectorExecutionDispatcher::plan(int64_t population,int device) {
  const auto* properties=at::cuda::getDeviceProperties(device);
  if(population<1024||population>1048576||std::string(properties->name)!="NVIDIA GeForce RTX 3090"||
      properties->major!=8||properties->minor!=6||at::globalContext().allowTF32CuBLAS())
    return {SelectorForwardBackend::Legacy,SelectorBackwardBackend::Legacy,false};
  // Tiled was the robust winner across the calibrated range. Lt and separate
  // bias reduction remain research candidates, not speculative N thresholds.
  return {SelectorForwardBackend::Tiled,SelectorBackwardBackend::FusedVJP,true};
}
std::vector<torch::Tensor> flexgsInputForward(const std::vector<torch::Tensor>& a,double ratio) {
  TORCH_CHECK(a.size()==7,"Input11 input count");const int n=a[0].size(0),m=a[5].size(0);
  for(int i:{0,1,2,3,5,6})fp32(a[i],a[0]);
  TORCH_CHECK(n<=INT_MAX/32&&a[0].sizes()==torch::IntArrayRef({n,3})&&a[1].sizes()==a[0].sizes()&&
      a[2].numel()==n*3&&a[3].sizes()==torch::IntArrayRef({n,1})&&a[4].is_cuda()&&a[4].device()==a[0].device()&&
      a[4].scalar_type()==torch::kInt64&&a[4].is_contiguous()&&a[4].numel()==n&&a[5].sizes()==torch::IntArrayRef({m,3})&&a[6].numel()==m,
      "Invalid Input11 shapes");
  TORCH_CHECK(!a[5].requires_grad()&&!a[6].requires_grad()&&std::isfinite(ratio)&&ratio>0&&ratio<=1,"Input11 geometry/ratio");
  c10::cuda::CUDAGuard guard(a[0].device());auto out=torch::empty({n,11},a[0].options());
  if(n)input11<<<(n+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,ptr(a[0]),ptr(a[1]),ptr(a[2]),ptr(a[3]),a[4].data_ptr<int64_t>(),ptr(a[5]),ptr(a[6]),logf(float(ratio)),out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {out};
}
std::vector<torch::Tensor> flexgsInputBackward(const std::vector<torch::Tensor>& a,const torch::Tensor& g) {
  c10::cuda::CUDAGuard guard(a[0].device());const int n=a[0].size(0);std::vector<torch::Tensor> out;
  for(int j=0;j<4;++j)out.push_back(torch::empty_like(a[j]));
  if(n)inputBackward<<<(n+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,ptr(g),a[4].data_ptr<int64_t>(),ptr(a[6]),out[0].data_ptr<float>(),out[1].data_ptr<float>(),out[2].data_ptr<float>(),out[3].data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<torch::Tensor> flexgsMLPForward(const std::vector<torch::Tensor>& a) {
  TORCH_CHECK(a.size()==7,"FlexGS MLP input count");for(auto& t:a)fp32(t,a[0]);
  const int n=a[0].size(0),width=a[1].size(1);
  TORCH_CHECK((width==11||width==19||width==21)&&n<=INT_MAX/32&&a[0].sizes()==torch::IntArrayRef({n,width})&&a[1].sizes()==torch::IntArrayRef({32,width})&&
      a[2].numel()==32&&a[3].sizes()==torch::IntArrayRef({32,32})&&a[4].numel()==32&&a[5].sizes()==torch::IntArrayRef({2,32})&&a[6].numel()==2,"FlexGS MLP11/19/21 layout");
  c10::cuda::CUDAGuard guard(a[0].device());
  if(n&&SelectorExecutionDispatcher::plan(n,a[0].get_device()).forward==SelectorForwardBackend::Tiled) {
    auto logits=torch::empty({n,2},a[0].options()),h0=torch::empty({n,32},a[0].options()),h1=torch::empty_like(h0);
    Weights p{ptr(a[1]),ptr(a[2]),ptr(a[3]),ptr(a[4]),ptr(a[5]),ptr(a[6])};
    if(width==21)tiledNetwork<21><<<(n+7)/8,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,ptr(a[0]),p,h0.data_ptr<float>(),h1.data_ptr<float>(),logits.data_ptr<float>());
    else if(width==19)tiledNetwork<19><<<(n+7)/8,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,ptr(a[0]),p,h0.data_ptr<float>(),h1.data_ptr<float>(),logits.data_ptr<float>());
    else tiledNetwork<11><<<(n+7)/8,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,ptr(a[0]),p,h0.data_ptr<float>(),h1.data_ptr<float>(),logits.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {logits,h0,h1};
  }
  // Warp-per-Gaussian wins on small online maps; GEMM scales better on large maps.
  if(n>16384) {
    auto h0=torch::relu(torch::nn::functional::linear(a[0],a[1],a[2]));
    auto h1=torch::relu(torch::nn::functional::linear(h0,a[3],a[4]));
    return {torch::nn::functional::linear(h1,a[5],a[6]),h0,h1};
  }
  auto logits=torch::empty({n,2},a[0].options()),h0=torch::empty({n,32},a[0].options()),h1=torch::empty_like(h0);
  Weights p{ptr(a[1]),ptr(a[2]),ptr(a[3]),ptr(a[4]),ptr(a[5]),ptr(a[6])};
  if(n) {
    if(width==21)network<21><<<(n+3)/4,128,0,c10::cuda::getCurrentCUDAStream()>>>(n,ptr(a[0]),p,h0.data_ptr<float>(),h1.data_ptr<float>(),logits.data_ptr<float>());
    else if(width==19)network<19><<<(n+3)/4,128,0,c10::cuda::getCurrentCUDAStream()>>>(n,ptr(a[0]),p,h0.data_ptr<float>(),h1.data_ptr<float>(),logits.data_ptr<float>());
    else network<11><<<(n+3)/4,128,0,c10::cuda::getCurrentCUDAStream()>>>(n,ptr(a[0]),p,h0.data_ptr<float>(),h1.data_ptr<float>(),logits.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {logits,h0,h1};
}
std::vector<torch::Tensor> flexgsMLPBackward(const std::vector<torch::Tensor>& a,const torch::Tensor& gl,const torch::Tensor& gh) {
  c10::cuda::CUDAGuard guard(a[0].device());const int n=a[0].size(0);
  auto g=gl.defined()?gl:torch::zeros({n,2},a[0].options());auto d1=torch::empty_like(a[8]);
  if(n)finalVjp<<<(n*32+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n*32,ptr(g),ptr(a[5]),ptr(a[8]),d1.data_ptr<float>());
  auto dw2=g.t().mm(a[8]),db2=g.sum(0),dw1=d1.t().mm(a[7]),db1=d1.sum(0),d0=d1.mm(a[3]);
  if(n)reluAdd<<<(n*32+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n*32,ptr(a[7]),ptr(gh),d0.data_ptr<float>());
  auto dw0=d0.t().mm(a[0]),db0=d0.sum(0),dx=d0.mm(a[1]);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {dx,dw0,db0,dw1,db1,dw2,db2};
}
std::vector<torch::Tensor> flexgsSampleForward(const torch::Tensor& z,const torch::Tensor& e,const torch::Tensor& eligible,double tau) {
  fp32(z,z);fp32(e,z);const int n=z.size(0);
  TORCH_CHECK(n<=INT_MAX&&z.sizes()==torch::IntArrayRef({n,2})&&e.sizes()==z.sizes()&&eligible.is_cuda()&&
      eligible.device()==z.device()&&eligible.scalar_type()==torch::kBool&&eligible.is_contiguous()&&eligible.numel()==n&&std::isfinite(tau)&&tau>0,"FlexGS sampling shape/tau");
  c10::cuda::CUDAGuard guard(z.device());auto p=torch::empty({n},z.options()),mask=torch::empty_like(p),
      soft=torch::empty_like(z),plain=torch::empty_like(z),ids=torch::empty({n},eligible.options().dtype(torch::kLong));
  if(!n)return {p,mask,ids,soft,plain};
  auto stream=c10::cuda::getCurrentCUDAStream();
  auto& workspace=compactWorkspace(z,n);auto flags=workspace.flags;
  sample<<<(n+255)/256,256,0,stream>>>(n,ptr(z),ptr(e),eligible.data_ptr<bool>(),float(tau),p.data_ptr<float>(),mask.data_ptr<float>(),soft.data_ptr<float>(),plain.data_ptr<float>(),flags.data_ptr<bool>());
  auto count=workspace.count;size_t bytes=workspace.bytes;cub::CountingInputIterator<int64_t> rows(0);
  if(workspace.last_n!=n) {
    C10_CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr,bytes,rows,flags.data_ptr<bool>(),ids.data_ptr<int64_t>(),count.data_ptr<int>(),n,stream));
    workspace.last_n=n;workspace.bytes=bytes;
  }
  if(!workspace.temp.defined()||workspace.temp.numel()<int64_t(bytes))
    workspace.temp=torch::empty({int64_t(bytes)},z.options().dtype(torch::kUInt8));
  auto temp=workspace.temp;
  C10_CUDA_CHECK(cub::DeviceSelect::Flagged(temp.data_ptr(),bytes,rows,flags.data_ptr<bool>(),ids.data_ptr<int64_t>(),count.data_ptr<int>(),n,stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaMemcpyAsync(workspace.host.data_ptr<int>(),count.data_ptr<int>(),sizeof(int),cudaMemcpyDeviceToHost,stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  const int k=*workspace.host.data_ptr<int>(); // The current Renderer still needs a Host K.
  return {p,mask,ids.slice(0,0,k),soft,plain};
}
torch::Tensor flexgsSampleBackward(const std::vector<torch::Tensor>& a,const torch::Tensor& gp,const torch::Tensor& gm,double tau) {
  c10::cuda::CUDAGuard guard(a[0].device());auto out=torch::empty_like(a[0]);int n=a[0].size(0);
  if(n)sampleVjp<<<(n+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,ptr(a[0]),ptr(a[1]),a[2].data_ptr<bool>(),ptr(gp),ptr(gm),float(tau),out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
}

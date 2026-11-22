#include "include/compensation_lattice_cuda.h"
#include <torch/csrc/autograd/custom_function.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <mutex>
#include <deque>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace {
struct QueryTiming {
  struct Sample {cudaEvent_t begin=nullptr,end=nullptr;int stage=0;};
  std::mutex mutex;
  uint64_t calls[3]={};
  std::deque<Sample> pending;
  std::vector<Sample> idle;
  std::array<std::vector<float>,3> values;
  ~QueryTiming(){for(auto s:pending){cudaEventDestroy(s.begin);cudaEventDestroy(s.end);}for(auto s:idle){cudaEventDestroy(s.begin);cudaEventDestroy(s.end);}}
  Sample begin(int stage) {
    std::lock_guard<std::mutex> lock(mutex);
    if(++calls[stage]%19 || pending.size()>16)return {};
    Sample s;
    if(!idle.empty()){s=idle.back();idle.pop_back();}
    else{cudaEventCreate(&s.begin);cudaEventCreate(&s.end);}
    s.stage=stage;cudaEventRecord(s.begin,at::cuda::getCurrentCUDAStream());return s;
  }
  void end(Sample s) {
    if(!s.begin)return;
    cudaEventRecord(s.end,at::cuda::getCurrentCUDAStream());
    std::lock_guard<std::mutex> lock(mutex);pending.push_back(s);
  }
  void report() {
    std::lock_guard<std::mutex> lock(mutex);
    while(!pending.empty() && cudaEventQuery(pending.front().end)==cudaSuccess) {
      auto s=pending.front();pending.pop_front();float ms=0;cudaEventElapsedTime(&ms,s.begin,s.end);
      values[s.stage].push_back(ms);idle.push_back(s);
    }
    for(int k=0;k<3;++k)if(!values[k].empty()) {
      auto v=values[k];std::sort(v.begin(),v.end());
      const char* names[]={"tf_forward_ms","tf_backward_ms","demand_update_ms"};
      std::cout<<"[CompensationLattice CUDA] component="<<names[k]
          <<" mean="<<std::accumulate(v.begin(),v.end(),0.0)/v.size()<<" p50="<<v[v.size()/2]
          <<" p95="<<v[v.size()*95/100]<<" max="<<v.back()<<" samples="<<v.size()<<'\n';
    }
  }
};
QueryTiming query_timing;
__global__ void aggregateDemand(int n,const float* xyz,const float* gx,const float* gs,
    const float* lo,const float* hi,int c,float lambda,float* sums,float* counts) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  int b[3];float nx=0,ns=0;
  for(int a=0;a<3;++a) {
    float u=(xyz[3*i+a]-lo[a])/(hi[a]-lo[a]);
    b[a]=min(int(floorf(fminf(fmaxf(u,0.f),1.f)*c)),c-1);
    nx+=gx[3*i+a]*gx[3*i+a];ns+=gs[3*i+a]*gs[3*i+a];
  }
  int cell=(b[0]*c+b[1])*c+b[2];
  atomicAdd(sums+cell,sqrtf(nx)+lambda*sqrtf(ns));atomicAdd(counts+cell,1.f);
}
__global__ void demandEMA(int cells,const float* sums,const float* counts,float beta,float* demand,int64_t* updates) {
  int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=cells || counts[c]==0)return;
  demand[c]=beta*demand[c]+(1-beta)*sums[c]/counts[c];++updates[c];
}
// One thread per point; only six residual values are materialized.
__device__ bool coordinates(const float* x, const float* lo, const float* hi,
    int r, bool identity, int* base, float* f, float* derivative) {
  bool inside = true;
  for (int a=0; a<3; ++a) {
    inside &= isfinite(x[a]) && x[a]>=lo[a] && x[a]<=hi[a];
    float u=(x[a]-lo[a])/(hi[a]-lo[a]);
    bool finite=isfinite(u);
    if (!finite) u=0;
    float v=fminf(fmaxf(u,0.f),1.f)*(r-1);
    base[a]=min(static_cast<int>(floorf(v)),r-2);
    f[a]=v-base[a];
    derivative[a]=finite && u>=0.f && u<=1.f ? (r-1)/(hi[a]-lo[a]) : 0.f;
  }
  return !identity || inside;
}
__global__ void queryForward(int n, const float* xyz, const float* values,
    const float* lo, const float* hi, const float* bound, int r, int k,
    int lower, int upper, float beta, float scale, bool identity, float* output,
    const float* fine,const float* tile_lo,const float* tile_hi,const int64_t* owners,int width,int rf,int tiles) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=n) return;
  int base[3]; float f[3],derivative[3], z[6]={};
  const bool valid=coordinates(xyz+3*i,lo,hi,r,identity,base,f,derivative);
  if (valid) for (int c=0; c<8; ++c) {
    int dx=c&1,dy=(c>>1)&1,dz=(c>>2)&1;
    int node=((base[0]+dx)*r+base[1]+dy)*r+base[2]+dz;
    float w=(dx?f[0]:1-f[0])*(dy?f[1]:1-f[1])*(dz?f[2]:1-f[2]);
    for(int d=0;d<6;++d)
      z[d]+=w*((1-beta)*values[(node*k+lower)*6+d]+beta*values[(node*k+upper)*6+d]);
  }
  if(valid && tiles>0) {
    int cell=(base[0]*(r-1)+base[1])*(r-1)+base[2];
    for(int slot=0;slot<width;++slot) {
      int t=owners[cell*width+slot];if(t<0)continue;
      bool supported=true;
      for(int a=0;a<3;++a)supported &= xyz[3*i+a]>=tile_lo[3*t+a] && xyz[3*i+a]<tile_hi[3*t+a];
      if(!supported)continue;
      int b[3];float f[3],derivative[3];coordinates(xyz+3*i,tile_lo+3*t,tile_hi+3*t,rf,true,b,f,derivative);
      for(int c=0;c<8;++c) {
        int dx=c&1,dy=(c>>1)&1,dz=(c>>2)&1;
        int node=t*rf*rf*rf+((b[0]+dx)*rf+b[1]+dy)*rf+b[2]+dz;
        float w=(dx?f[0]:1-f[0])*(dy?f[1]:1-f[1])*(dz?f[2]:1-f[2]);
        for(int d=0;d<6;++d)z[d]+=w*((1-beta)*fine[(node*k+lower)*6+d]+beta*fine[(node*k+upper)*6+d]);
      }
    }
  }
  for(int d=0;d<6;++d) output[6*i+d]=valid ? tanhf(z[d])*(d<3?bound[0]:scale) : 0.f;
}
__global__ void queryBackward(int n, const float* xyz, const float* values,
    const float* lo,const float* hi,const float* bound,int r,int k,
    int lower,int upper,float beta,float scale,bool identity,
    const float* output,const float* grad,float* gx,float* gv,
    const float* fine,const float* tile_lo,const float* tile_hi,const int64_t* owners,int width,int rf,int tiles,float* gf) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  int base[3]; float f[3], derivative[3], grad_xyz[3]={},gz[6];
  if(!coordinates(xyz+3*i,lo,hi,r,identity,base,f,derivative)) return;
  for(int d=0;d<6;++d) {
    float b=d<3?bound[0]:scale, t=output[i*6+d]/b;
    gz[d]=grad[i*6+d]*b*(1-t*t);
  }
  for(int c=0;c<8;++c) {
    int bit[3]={c&1,(c>>1)&1,(c>>2)&1};
    int node=((base[0]+bit[0])*r+base[1]+bit[1])*r+base[2]+bit[2];
    float w[3]={bit[0]?f[0]:1-f[0],bit[1]?f[1]:1-f[1],bit[2]?f[2]:1-f[2]};
    float weight=w[0]*w[1]*w[2];
    for(int d=0;d<6;++d) {
      atomicAdd(gv+(node*k+lower)*6+d,gz[d]*weight*(1-beta));
      if(beta!=0.f) atomicAdd(gv+(node*k+upper)*6+d,gz[d]*weight*beta);
      float value=(1-beta)*values[(node*k+lower)*6+d]+beta*values[(node*k+upper)*6+d];
      for(int a=0;a<3;++a)
        grad_xyz[a]+=gz[d]*value*(bit[a]?1.f:-1.f)*w[(a+1)%3]*w[(a+2)%3]*derivative[a];
    }
  }
  if(tiles>0) {
    int cell=(base[0]*(r-1)+base[1])*(r-1)+base[2];
    for(int slot=0;slot<width;++slot) {
      int t=owners[cell*width+slot];if(t<0)continue;
      bool supported=true;
      for(int a=0;a<3;++a)supported &= xyz[3*i+a]>=tile_lo[3*t+a] && xyz[3*i+a]<tile_hi[3*t+a];
      if(!supported)continue;
      int b[3];float f[3],derivative[3];coordinates(xyz+3*i,tile_lo+3*t,tile_hi+3*t,rf,true,b,f,derivative);
      for(int c=0;c<8;++c) {
        int bit[3]={c&1,(c>>1)&1,(c>>2)&1};
        int node=t*rf*rf*rf+((b[0]+bit[0])*rf+b[1]+bit[1])*rf+b[2]+bit[2];
        float w[3]={bit[0]?f[0]:1-f[0],bit[1]?f[1]:1-f[1],bit[2]?f[2]:1-f[2]};
        float weight=w[0]*w[1]*w[2];
        for(int d=0;d<6;++d) {
          atomicAdd(gf+(node*k+lower)*6+d,gz[d]*weight*(1-beta));
          if(beta!=0.f)atomicAdd(gf+(node*k+upper)*6+d,gz[d]*weight*beta);
          float value=(1-beta)*fine[(node*k+lower)*6+d]+beta*fine[(node*k+upper)*6+d];
          for(int a=0;a<3;++a)grad_xyz[a]+=gz[d]*value*(bit[a]?1.f:-1.f)*w[(a+1)%3]*w[(a+2)%3]*derivative[a];
        }
      }
    }
  }
  for(int a=0;a<3;++a) gx[3*i+a]=grad_xyz[a];
}
class LatticeQuery : public torch::autograd::Function<LatticeQuery> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
      torch::Tensor xyz,torch::Tensor coarse,torch::Tensor lo,torch::Tensor hi,torch::Tensor bound,
      int64_t r,int64_t lower,int64_t upper,double beta,double scale,bool identity,
      torch::Tensor fine,torch::Tensor tile_min,torch::Tensor tile_max,torch::Tensor cell_tiles,int64_t rf) {
    c10::cuda::CUDAGuard guard(xyz.device());
    TORCH_CHECK(xyz.scalar_type()==torch::kFloat32 && coarse.scalar_type()==torch::kFloat32 &&
        xyz.is_cuda() && coarse.device()==xyz.device(),"CUDA lattice requires colocated float32 inputs");
    xyz=xyz.contiguous();
    auto timing=query_timing.begin(0);
    auto result=torch::empty({xyz.size(0),6},xyz.options());
    if(xyz.size(0)>0) queryForward<<<(xyz.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
        xyz.size(0),xyz.data_ptr<float>(),coarse.data_ptr<float>(),lo.data_ptr<float>(),hi.data_ptr<float>(),
        bound.data_ptr<float>(),r,coarse.size(1),lower,upper,beta,scale,identity,result.data_ptr<float>(),
        fine.data_ptr<float>(),tile_min.data_ptr<float>(),tile_max.data_ptr<float>(),cell_tiles.data_ptr<int64_t>(),
        cell_tiles.size(1),rf,tile_min.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    query_timing.end(timing);
    ctx->save_for_backward({xyz,coarse,lo,hi,bound,result,fine,tile_min,tile_max,cell_tiles});
    ctx->saved_data["r"]=r;ctx->saved_data["lower"]=lower;ctx->saved_data["upper"]=upper;
    ctx->saved_data["beta"]=beta;ctx->saved_data["scale"]=scale;ctx->saved_data["identity"]=identity;
    ctx->saved_data["rf"]=rf;
    return result;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                torch::autograd::variable_list grads) {
    auto t=ctx->get_saved_variables();
    c10::cuda::CUDAGuard guard(t[0].device());
    auto timing=query_timing.begin(1);
    auto gx=torch::zeros_like(t[0]),gv=torch::zeros_like(t[1]),grad=grads[0].contiguous();
    auto gf=torch::zeros_like(t[6]);
    if(t[0].size(0)>0) queryBackward<<<(t[0].size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
        t[0].size(0),t[0].data_ptr<float>(),t[1].data_ptr<float>(),t[2].data_ptr<float>(),t[3].data_ptr<float>(),
        t[4].data_ptr<float>(),ctx->saved_data["r"].toInt(),t[1].size(1),ctx->saved_data["lower"].toInt(),
        ctx->saved_data["upper"].toInt(),ctx->saved_data["beta"].toDouble(),ctx->saved_data["scale"].toDouble(),
        ctx->saved_data["identity"].toBool(),t[5].data_ptr<float>(),grad.data_ptr<float>(),gx.data_ptr<float>(),gv.data_ptr<float>(),
        t[6].data_ptr<float>(),t[7].data_ptr<float>(),t[8].data_ptr<float>(),t[9].data_ptr<int64_t>(),t[9].size(1),
        ctx->saved_data["rf"].toInt(),t[7].size(0),gf.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    query_timing.end(timing);
    return {gx,gv,{}, {}, {}, {}, {}, {}, {}, {}, {},t[6].numel()?gf:torch::Tensor(),{}, {}, {}, {}};
  }
};
} // namespace
torch::Tensor compensationLatticeCUDA(const torch::Tensor& xyz,const torch::Tensor& coarse,
    const torch::Tensor& lo,const torch::Tensor& hi,const torch::Tensor& bound,
    int r,int lower,int upper,float beta,float scale,bool identity,
    const torch::Tensor& fine,const torch::Tensor& tile_min,const torch::Tensor& tile_max,const torch::Tensor& cell_tiles,int rf) {
  return LatticeQuery::apply(xyz,coarse,lo,hi,bound,int64_t(r),int64_t(lower),int64_t(upper),
                            double(beta),double(scale),identity,fine,tile_min,tile_max,cell_tiles,int64_t(rf));
}
void reportCompensationCudaTiming(){query_timing.report();}
void compensationDemandCUDA(const torch::Tensor& xyz,const torch::Tensor& gx,const torch::Tensor& gs,
    const torch::Tensor& lo,const torch::Tensor& hi,int c,
    const torch::Tensor& demand,const torch::Tensor& updates,float beta,float lambda) {
  c10::cuda::CUDAGuard guard(xyz.device());
  auto timing=query_timing.begin(2);
  const int cells=c*c*c;
  auto sums=torch::zeros({2,cells},xyz.options()),x=xyz.contiguous(),px=gx.contiguous(),ps=gs.contiguous();
  if(xyz.size(0)>0)aggregateDemand<<<(xyz.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
      xyz.size(0),x.data_ptr<float>(),px.data_ptr<float>(),ps.data_ptr<float>(),lo.data_ptr<float>(),hi.data_ptr<float>(),
      c,lambda,sums.data_ptr<float>(),sums.data_ptr<float>()+cells);
  demandEMA<<<(cells+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(cells,sums.data_ptr<float>(),
      sums.data_ptr<float>()+cells,beta,demand.data_ptr<float>(),updates.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();query_timing.end(timing);
}

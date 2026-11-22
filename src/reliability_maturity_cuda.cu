#include "include/reliability_maturity_cuda.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>

namespace {
struct Rows {
  float *m,*ema,*px,*ps,*pq,*O,*V;
  bool *snapshot,*initialized,*touched;
  int64_t* seen;
  const float *x,*s,*q;
  const int* radii;
  const float* grad[6]; int width[6];
  int words,word; uint64_t bit;
  float tauO,tauV,tauD,fast,slow,floor;
  float *cema,*selector_m;const float* contribution;
  float inverse_samples,decay,tauR,weight;
};
__host__ __device__ void updateRow(int i,Rows p) {
  bool finite=true,nonzero=false;
  for(int j=0;j<3;++j)finite=finite&&isfinite(p.x[3*i+j])&&isfinite(p.s[3*i+j]);
  for(int j=0;j<4;++j)finite=finite&&isfinite(p.q[4*i+j]);
  bool observed=p.radii&&p.radii[i]>0;
  if(observed)for(int g=0;g<6;++g)if(p.grad[g])for(int j=0;j<p.width[g];++j) {
    float v=p.grad[g][i*p.width[g]+j];finite=finite&&isfinite(v);nonzero=nonzero||(v!=0);
  }
  bool effective=observed&&finite&&nonzero;
  if(effective) {
    p.O[i]+=1;
    auto& bits=p.seen[i*p.words+p.word];
    if(!(uint64_t(bits)&p.bit)){p.V[i]+=1;bits=int64_t(uint64_t(bits)|p.bit);}
  }
  bool valid=effective&&p.snapshot[i];
  float drift=0;
  if(valid) {
    float largest=fmaxf(p.ps[3*i],fmaxf(p.ps[3*i+1],p.ps[3*i+2]));
    float squares=0,dx=0,ds=0,dot=0,qn=0,pn=0;
    for(int j=0;j<3;++j) {
      squares+=expf(2*(p.ps[3*i+j]-largest));
      float d=p.x[3*i+j]-p.px[3*i+j];dx+=d*d;
      d=p.s[3*i+j]-p.ps[3*i+j];ds+=d*d;
    }
    for(int j=0;j<4;++j){float q=p.q[4*i+j],v=p.pq[4*i+j];dot+=q*v;qn+=q*q;pn+=v*v;}
    float size=fmaxf(expf(largest)*sqrtf(squares/3),p.floor);
    dot=fminf(1,fabsf(dot)/fmaxf(sqrtf(qn*pn),1e-12f));
    drift=fmaxf(sqrtf(dx)/size,fmaxf(sqrtf(ds/3),sqrtf(fmaxf(0,1-dot*dot))));
    valid=isfinite(drift)&&isfinite(size);
  }
  if(valid) {
    float beta=drift>p.ema[i]?p.fast:p.slow;
    p.ema[i]=p.initialized[i]?p.ema[i]+beta*(drift-p.ema[i]):drift;
    p.initialized[i]=true;
  }
  float O=-expm1f(-p.O[i]/p.tauO),V=-expm1f(-p.V[i]/p.tauV);
  p.m[i]=finite&&p.initialized[i]?powf(O,.35f)*powf(V,.35f)*expf(-.30f*p.ema[i]/p.tauD):0;
  if(p.cema) {
    if(p.contribution) {
      float c=p.contribution[i]*p.inverse_samples;
      if(isfinite(c)&&c>=0)p.cema[i]=p.decay*p.cema[i]+(1-p.decay)*c;
    }
    float R=-expm1f(-p.cema[i]/p.tauR);
    p.selector_m[i]=p.m[i]*(1-p.weight+p.weight*R);
  }
  for(int j=0;j<3;++j){p.px[3*i+j]=p.x[3*i+j];p.ps[3*i+j]=p.s[3*i+j];}
  for(int j=0;j<4;++j)p.pq[4*i+j]=p.q[4*i+j];
  p.snapshot[i]=finite;p.touched[i]=false;
}
__global__ void update(int n,Rows p){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)updateRow(i,p);}
}

void reliabilityMaturityUpdate(const std::vector<torch::Tensor>& a,
    const torch::Tensor& x,const torch::Tensor& s,const torch::Tensor& q,
    const torch::Tensor& radii,const std::vector<torch::Tensor>& gradients,
    int64_t ordinal,const ProtectionConfig& c,
    const torch::Tensor& contribution,int64_t sample_pixels) {
  TORCH_CHECK(a.size()==13&&gradients.size()==6,"Reliability state layout");
  int n=x.size(0);if(!n)return;
  for(auto& t:a)if(t.numel())TORCH_CHECK(t.is_contiguous()&&t.size(0)==n&&t.device()==x.device(),"Reliability row alignment");
  TORCH_CHECK(radii.scalar_type()==torch::kInt32&&radii.numel()==n&&radii.is_contiguous(),"Full radii must be row-aligned int32");
  Rows p{};p.m=a[0].data_ptr<float>();p.ema=a[1].data_ptr<float>();p.px=a[2].data_ptr<float>();
  p.ps=a[3].data_ptr<float>();p.pq=a[4].data_ptr<float>();p.snapshot=a[5].data_ptr<bool>();
  p.initialized=a[6].data_ptr<bool>();p.touched=a[7].data_ptr<bool>();p.O=a[8].data_ptr<float>();p.V=a[9].data_ptr<float>();
  p.seen=a[10].data_ptr<int64_t>();p.words=a[10].size(1);p.word=ordinal/64;p.bit=uint64_t(1)<<(ordinal%64);
  TORCH_CHECK(p.word<p.words,"Reliability view bitset capacity");
  p.x=x.data_ptr<float>();p.s=s.data_ptr<float>();p.q=q.data_ptr<float>();p.radii=radii.data_ptr<int>();
  for(int g=0;g<6;++g)if(gradients[g].defined()) {
    TORCH_CHECK(gradients[g].is_contiguous()&&gradients[g].device()==x.device()&&gradients[g].scalar_type()==torch::kFloat32&&gradients[g].size(0)==n,"Full gradient layout");
    p.grad[g]=gradients[g].data_ptr<float>();p.width[g]=gradients[g].numel()/n;
  }
  p.tauO=c.reliability_tau_o;p.tauV=c.reliability_tau_v;p.tauD=c.drift_temperature;
  if(c.rendering_contribution) {
    TORCH_CHECK(a[11].numel()==n&&a[12].numel()==n,"Contribution state layout");
    p.cema=a[11].data_ptr<float>();p.selector_m=a[12].data_ptr<float>();
    p.decay=c.contribution_decay;p.tauR=c.contribution_tau;p.weight=c.contribution_weight;
    if(contribution.defined()) {
      TORCH_CHECK(contribution.is_contiguous()&&contribution.numel()==n&&
          contribution.device()==x.device()&&contribution.scalar_type()==torch::kFloat32&&sample_pixels>0,
          "Contribution capture layout");
      p.contribution=contribution.data_ptr<float>();p.inverse_samples=1.f/sample_pixels;
    }
  }
  p.fast=c.beta_fast;p.slow=c.beta_slow;p.floor=c.reliability_xyz_floor;
  if(x.is_cuda()) {
    c10::cuda::CUDAGuard guard(x.device());
    update<<<(n+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,p);C10_CUDA_KERNEL_LAUNCH_CHECK();
  }else for(int i=0;i<n;++i)updateRow(i,p);
}

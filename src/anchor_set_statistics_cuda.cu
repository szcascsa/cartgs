#include "include/anchor_set_statistics_cuda.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>

namespace mapping_cuda {
namespace {
__device__ float sum(float value) {
  for(int d=16;d;d/=2)value+=__shfl_down_sync(0xffffffff,value,d);
  return __shfl_sync(0xffffffff,value,0);
}
__global__ void groupRows(int n,int a,const int64_t* owners,int* counts,int64_t* members) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  const auto anchor=owners[i];CUDA_KERNEL_ASSERT(anchor>=0 && anchor<a);
  const int slot=atomicAdd(counts+anchor,1);CUDA_KERNEL_ASSERT(slot<10);
  members[anchor*10+slot]=i;
}
// Four independent warps per block. Sorting at most ten indices makes the
// reduction order independent of the grouping kernel's atomic arrival order.
__device__ int64_t member(int a,int count,const int64_t* members,int64_t* scratch) {
  const int lane=threadIdx.x%32;
  if(!lane) {
    for(int j=0;j<count;++j)scratch[j]=members[a*10+j];
    for(int j=1;j<count;++j) {
      const auto value=scratch[j];int k=j;
      while(k>0 && scratch[k-1]>value){scratch[k]=scratch[k-1];--k;}
      scratch[k]=value;
    }
  }
  __syncwarp();return lane<count?scratch[lane]:-1;
}
__global__ void statistics(int a,const float* x,const float* s,const float* dc,const float* op,
    const float* center,const float* ref,const bool* eligible,const float* ratio,
    const int* counts,const int64_t* members,float* means,float* out) {
  const int anchor=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(anchor>=a)return;
  __shared__ int64_t sorted[4][10];
  const int count=counts[anchor];if(!count)return;
  const auto i=member(anchor,count,members,sorted[threadIdx.x/32]);
  const bool valid=i>=0 && eligible[i];float q[3]={0,0,0};
  if(valid)for(int j=0;j<3;++j)q[j]=(x[3*i+j]-center[3*anchor+j])/ref[anchor];
  const float m=sum(float(valid)),den=fmaxf(m,1.f);
  float mu[3];for(int j=0;j<3;++j)mu[j]=sum(q[j])/den;
  if(!lane)for(int j=0;j<3;++j)means[3*anchor+j]=mu[j];
  float d[3];for(int j=0;j<3;++j)d[j]=valid?q[j]-mu[j]:0.f;
  const float cov[6]={sum(d[0]*d[0])/den,sum(d[0]*d[1])/den,sum(d[0]*d[2])/den,
      sum(d[1]*d[1])/den,sum(d[1]*d[2])/den,sum(d[2]*d[2])/den};
  const float logratio=logf(ratio[0]);
  if(i>=0) {
    float* row=out+21*i;
    for(int j=0;j<3;++j){row[j]=q[j];row[3+j]=valid?s[3*i+j]:0.f;
      row[6+j]=valid?dc[3*i+j]:0.f;row[11+j]=mu[j];}
    row[9]=valid?op[i]:0.f;row[10]=logratio;
    for(int j=0;j<6;++j)row[14+j]=cov[j];row[20]=m/10.f;
  }
}
__global__ void vjp(int a,const float* x,const float* center,const float* ref,const bool* eligible,
    const int* counts,const int64_t* members,const float* means,const float* g,
    float* gx,float* gs,float* gc,float* go) {
  const int anchor=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(anchor>=a)return;
  __shared__ int64_t sorted[4][10];
  const int count=counts[anchor];if(!count)return;
  const auto i=member(anchor,count,members,sorted[threadIdx.x/32]);
  const bool valid=i>=0 && eligible[i];
  const float den=fmaxf(sum(float(valid)),1.f);
  float gm[3],cov[6];
  for(int j=0;j<3;++j)gm[j]=sum(i>=0?g[21*i+11+j]:0.f);
  for(int j=0;j<6;++j)cov[j]=sum(i>=0?g[21*i+14+j]:0.f)/den;
  float d[3]={0,0,0};
  if(valid)for(int j=0;j<3;++j)d[j]=(x[3*i+j]-center[3*anchor+j])/ref[anchor]-means[3*anchor+j];
  float gd[3]={2.f*cov[0]*d[0]+cov[1]*d[1]+cov[2]*d[2],
      cov[1]*d[0]+2.f*cov[3]*d[1]+cov[4]*d[2],
      cov[2]*d[0]+cov[4]*d[1]+2.f*cov[5]*d[2]};
  float mean_vjp[3];for(int j=0;j<3;++j)mean_vjp[j]=(gm[j]-sum(gd[j]))/den;
  if(i>=0) {
    for(int j=0;j<3;++j){gx[3*i+j]=valid?(g[21*i+j]+gd[j]+mean_vjp[j])/ref[anchor]:0.f;
      gs[3*i+j]=valid?g[21*i+3+j]:0.f;gc[3*i+j]=valid?g[21*i+6+j]:0.f;}
    go[i]=valid?g[21*i+9]:0.f;
  }
}
void check(const std::vector<torch::Tensor>& a) {
  TORCH_CHECK(a.size()>=9,"Statistics21 input count");
  const int64_t n=a[0].size(0),m=a[5].size(0);
  TORCH_CHECK(n<=INT_MAX/21 && m<=INT_MAX/10,"Statistics21 population exceeds supported range");
  for(int j=0;j<9;++j)TORCH_CHECK(a[j].is_cuda()&&a[j].device()==a[0].device()&&a[j].is_contiguous()&&
      a[j].scalar_type()==(j==4?torch::kInt64:(j==7?torch::kBool:torch::kFloat32)),"Statistics21 requires same-device contiguous CUDA FP32/Int64/Bool");
  TORCH_CHECK(a[0].sizes()==torch::IntArrayRef({n,3})&&a[1].sizes()==a[0].sizes()&&a[2].numel()==n*3&&
      a[3].sizes()==torch::IntArrayRef({n,1})&&a[4].sizes()==torch::IntArrayRef({n})&&
      a[5].sizes()==torch::IntArrayRef({m,3})&&a[6].sizes()==torch::IntArrayRef({m,1})&&
      a[7].sizes()==torch::IntArrayRef({n})&&a[8].numel()==1,"Statistics21 attribute shapes");
  TORCH_CHECK(!a[5].requires_grad()&&!a[6].requires_grad()&&!a[8].requires_grad(),"Statistics21 geometry buffers/ratio must be non-trainable");
}
const float* ptr(const torch::Tensor& t){return t.data_ptr<float>();}
}
std::vector<torch::Tensor> anchorSetStatisticsForward(const std::vector<torch::Tensor>& a) {
  check(a);c10::cuda::CUDAGuard guard(a[0].device());
  const int n=a[0].size(0),m=a[5].size(0);
  auto out=torch::empty({n,21},a[0].options()),members=torch::empty({m,10},a[4].options());
  auto counts=torch::zeros({m},a[4].options().dtype(torch::kInt32)),means=torch::empty({m,3},a[0].options());
  if(n) {
    auto stream=c10::cuda::getCurrentCUDAStream();
    groupRows<<<(n+127)/128,128,0,stream>>>(n,m,a[4].data_ptr<int64_t>(),counts.data_ptr<int>(),members.data_ptr<int64_t>());
    statistics<<<(m+3)/4,128,0,stream>>>(m,ptr(a[0]),ptr(a[1]),ptr(a[2]),ptr(a[3]),ptr(a[5]),ptr(a[6]),
        a[7].data_ptr<bool>(),ptr(a[8]),counts.data_ptr<int>(),members.data_ptr<int64_t>(),means.data_ptr<float>(),out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {out,members,counts,means};
}
std::vector<torch::Tensor> anchorSetStatisticsBackward(const std::vector<torch::Tensor>& a,const torch::Tensor& g) {
  check(a);TORCH_CHECK(a.size()==12 && g.sizes()==torch::IntArrayRef({a[0].size(0),21}),"Statistics21 saved tensors/VJP shape");
  c10::cuda::CUDAGuard guard(a[0].device());std::vector<torch::Tensor> out(9);
  for(int j=0;j<4;++j)out[j]=torch::empty_like(a[j]);
  const int n=a[0].size(0),m=a[5].size(0);
  if(n){vjp<<<(m+3)/4,128,0,c10::cuda::getCurrentCUDAStream()>>>(m,ptr(a[0]),ptr(a[5]),ptr(a[6]),a[7].data_ptr<bool>(),
      a[10].data_ptr<int>(),a[9].data_ptr<int64_t>(),ptr(a[11]),ptr(g),out[0].data_ptr<float>(),out[1].data_ptr<float>(),
      out[2].data_ptr<float>(),out[3].data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return out;
}
}

#include "include/anchor_context_residual_cuda.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>

namespace mapping_cuda {
namespace {
const float* ptr(const torch::Tensor& t){return t.defined()?t.data_ptr<float>():nullptr;}
__global__ void first(int n,const float* x,const float* w,const float* bias,float* hidden) {
  __shared__ float weights[11][33],input[8][12];
  const int tid=threadIdx.x,col=tid%32,row=blockIdx.x*8+tid/32;
  for(int i=tid;i<352;i+=256)weights[i%11][i/11]=w[i];
  for(int i=tid;i<88;i+=256){const int r=blockIdx.x*8+i/11;input[i/11][i%11]=r<n?x[r*11+i%11]:0.f;}
  __syncthreads();float value=0;
  #pragma unroll
  for(int j=0;j<11;++j)value=fmaf(input[tid/32][j],weights[j][col],value);
  if(row<n)hidden[row*32+col]=fmaxf(value+bias[col],0.f);
}
__global__ void group(int n,int anchors,const int64_t* owners,int* counts,int64_t* members) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  const int64_t a=owners[i];CUDA_KERNEL_ASSERT(a>=0 && a<anchors);
  const int slot=atomicAdd(counts+a,1);CUDA_KERNEL_ASSERT(slot<10);members[a*10+slot]=i;
}
__device__ int64_t member(int a,int count,const int64_t* members,int64_t* sorted) {
  const int lane=threadIdx.x%32;
  if(!lane) {
    for(int j=0;j<count;++j)sorted[j]=members[a*10+j];
    for(int j=1;j<count;++j){const auto value=sorted[j];int k=j;
      while(k>0 && sorted[k-1]>value){sorted[k]=sorted[k-1];--k;}sorted[k]=value;}
  }
  __syncwarp();return lane<count?sorted[lane]:-1;
}
__global__ void pool(int anchors,const float* h,const bool* valid,const int* counts,const int64_t* members,float* context) {
  int a=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(a>=anchors)return;
  __shared__ int64_t sorted[4][10];member(a,counts[a],members,sorted[threadIdx.x/32]);
  float total=0,maximum=0;int m=0;
  for(int j=0;j<counts[a];++j){const auto i=sorted[threadIdx.x/32][j];if(valid[i]) {
    ++m;if(lane<8){float v=h[32*i+lane];total+=v;maximum=fmaxf(maximum,v);}
  }}
  if(lane<8){context[17*a+lane]=total/fmaxf(float(m),1.f);context[17*a+8+lane]=maximum;}
  if(lane==8)context[17*a+16]=m/10.f;
}
__global__ void tail(int n,const float* h,const float* w,const float* bias,const float* w2,const float* b2,
    const int64_t* owners,const float* residual,float* next,float* logits) {
  __shared__ float weights[32][33],out_weights[2][32],hidden[8][33],activated[8][33];
  const int tid=threadIdx.x,col=tid%32,row=blockIdx.x*8+tid/32;
  for(int i=tid;i<1024;i+=256)weights[i%32][i/32]=w[i];
  if(tid<64)out_weights[tid/32][tid%32]=w2[tid];
  hidden[tid/32][col]=row<n?h[row*32+col]:0.f;__syncthreads();
  float value=0;
  #pragma unroll
  for(int j=0;j<32;++j)value=fmaf(hidden[tid/32][j],weights[j][col],value);
  value+=bias[col];if(row<n)value+=residual[owners[row]*32+col];
  value=fmaxf(value,0.f);activated[tid/32][col]=value;if(row<n)next[row*32+col]=value;
  __syncthreads();
  if(col<2 && row<n){float z=0;
    #pragma unroll
    for(int j=0;j<32;++j)z=fmaf(activated[tid/32][j],out_weights[col][j],z);
    logits[row*2+col]=z+b2[col];}
}
__global__ void finalVjp(int n,const float* g,const float* w,const float* hidden,float* out) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*32)return;
  const int row=i/32,j=i%32;out[i]=hidden[i]>0?g[row*2]*w[j]+g[row*2+1]*w[32+j]:0.f;
}
__global__ void gatherVjp(int anchors,const int* counts,const int64_t* members,const float* d1,float* sums) {
  int a=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(a>=anchors)return;
  __shared__ int64_t sorted[4][10];member(a,counts[a],members,sorted[threadIdx.x/32]);
  float value=0;for(int j=0;j<counts[a];++j)value+=d1[sorted[threadIdx.x/32][j]*32+lane];sums[a*32+lane]=value;
}
__global__ void poolVjp(int anchors,const float* h,const bool* valid,const int* counts,const int64_t* members,
    const float* context,const float* cg,float* g) {
  int a=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(a>=anchors)return;
  __shared__ int64_t sorted[4][10];member(a,counts[a],members,sorted[threadIdx.x/32]);
  if(lane>=8)return;
  int m=0,ties=0;const float maximum=context[a*17+8+lane];
  for(int j=0;j<counts[a];++j){const auto i=sorted[threadIdx.x/32][j];if(valid[i]){++m;ties+=h[i*32+lane]==maximum;}}
  const float mean=cg[a*17+lane]/fmaxf(float(m),1.f),mx=cg[a*17+8+lane]/fmaxf(float(ties),1.f);
  for(int j=0;j<counts[a];++j){const auto i=sorted[threadIdx.x/32][j];if(valid[i])
    g[i*32+lane]+=mean+(h[i*32+lane]==maximum?mx:0.f);}
}
__global__ void firstVjp(int n,const float* hidden,const float* shared_gradient,float* g) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*32)return;
  g[i]=hidden[i]>0?g[i]+(shared_gradient?shared_gradient[i]:0.f):0.f;
}
// Sort in the forward only. The current graph owns members and the two integer
// caches; a retained second VJP reads them without modifying saved tensors.
template<bool Packed>
__global__ void poolCached(int anchors,const float* h,const bool* valid,const int* counts,
    int64_t* members,float* context,int* eligible_count,int* max_masks) {
  int a=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(a>=anchors)return;
  if(!counts[a]){if(lane<8){context[17*a+lane]=0.f;context[17*a+8+lane]=0.f;}if(lane==8)context[17*a+16]=0.f;return;}
  __shared__ int64_t sorted[4][10];member(a,counts[a],members,sorted[threadIdx.x/32]);
  if(!Packed && lane<counts[a])members[a*10+lane]=sorted[threadIdx.x/32][lane];
  float total=0,maximum=0;int m=0,mask=0;
  for(int j=0;j<counts[a];++j){const auto i=sorted[threadIdx.x/32][j];if(valid[i]) {
    ++m;if(lane<8){const float v=h[32*i+lane];total+=v;
      if(v>maximum)mask=1<<j;else if(v==maximum)mask|=1<<j;
      maximum=fmaxf(maximum,v);}
  }}
  if(Packed){
    // n<=INT_MAX/32, so every row fits in the low 32 bits. Keep m and the
    // eight ten-bit tie masks in the high halves of this Anchor's ten slots.
    const int channel_mask=__shfl_sync(0xffffffff,mask,min(max(lane-1,0),7));
    if(lane<10){const uint64_t metadata=lane==0?m:(lane<=8?channel_mask:0);
      const uint64_t row=lane<counts[a]?uint64_t(sorted[threadIdx.x/32][lane]):0;
      members[a*10+lane]=int64_t((metadata<<32)|row);}
  }
  if(lane<8){context[17*a+lane]=total/fmaxf(float(m),1.f);context[17*a+8+lane]=maximum;if(!Packed)max_masks[a*8+lane]=mask;}
  if(lane==8){context[17*a+16]=m/10.f;if(!Packed)eligible_count[a]=m;}
}
template<bool Packed>
__device__ int64_t cachedMember(const int64_t* members,int offset) {
  return Packed?int64_t(uint32_t(members[offset])):members[offset];
}
template<bool Packed>
__global__ void gatherCached(int anchors,const int* counts,const int64_t* members,const float* d1,float* sums) {
  int a=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(a>=anchors)return;
  float value=0;for(int j=0;j<counts[a];++j)value+=d1[cachedMember<Packed>(members,a*10+j)*32+lane];sums[a*32+lane]=value;
}
template<bool Packed>
__global__ void finalGatherVjp(int anchors,const int* counts,const int64_t* members,const float* gz,
    const float* w,const float* hidden,float* d1,float* sums) {
  int a=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(a>=anchors)return;
  float sum=0;
  for(int j=0;j<counts[a];++j){const auto row=cachedMember<Packed>(members,a*10+j);
    const float value=hidden[row*32+lane]>0?gz[row*2]*w[lane]+gz[row*2+1]*w[32+lane]:0.f;
    d1[row*32+lane]=value;sum=__fadd_rn(sum,value);
  }
  sums[a*32+lane]=sum;
}
template<bool Packed>
__global__ void poolVjpCached(int anchors,const bool* valid,const int* counts,const int64_t* members,
    const int* eligible_count,const int* max_masks,const float* cg,float* g) {
  int a=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(a>=anchors || lane>=8)return;
  if(!counts[a])return;
  const int mask=Packed?int(uint64_t(members[a*10+1+lane])>>32):max_masks[a*8+lane];
  const int m=Packed?int(uint64_t(members[a*10])>>32):eligible_count[a];
  const float mean=cg[a*17+lane]/fmaxf(float(m),1.f),
      mx=cg[a*17+8+lane]/fmaxf(float(__popc(mask)),1.f);
  for(int j=0;j<counts[a];++j){const auto i=cachedMember<Packed>(members,a*10+j);if(valid[i])
    g[i*32+lane]+=mean+((mask&(1<<j))?mx:0.f);}
}
void check(const std::vector<torch::Tensor>& a,int64_t anchors) {
  TORCH_CHECK(a.size()>=10 && a[0].dim()==2,"ContextMLP input count");
  const int64_t n=a[0].size(0);TORCH_CHECK(n<=INT_MAX/32 && anchors>=0 && anchors<=INT_MAX/32,"ContextMLP size");
  for(int i=0;i<10;++i)TORCH_CHECK(a[i].is_cuda()&&a[i].is_contiguous()&&a[i].device()==a[0].device()&&
      a[i].scalar_type()==(i==8?torch::kLong:(i==9?torch::kBool:torch::kFloat32)),"ContextMLP CUDA FP32/owners/eligibility");
  TORCH_CHECK(a[0].size(1)==11&&a[1].sizes()==torch::IntArrayRef({32,11})&&a[2].numel()==32&&
      a[3].sizes()==torch::IntArrayRef({32,32})&&a[4].numel()==32&&a[5].sizes()==torch::IntArrayRef({2,32})&&
      a[6].numel()==2&&a[7].sizes()==torch::IntArrayRef({32,17})&&a[8].sizes()==torch::IntArrayRef({n})&&a[9].sizes()==a[8].sizes(),"ContextMLP layout");
}
}
std::vector<torch::Tensor> anchorContextMLPForward(const std::vector<torch::Tensor>& a,int64_t anchors) {
  check(a,anchors);c10::cuda::CUDAGuard guard(a[0].device());const int n=a[0].size(0);
  auto h0=torch::empty({n,32},a[0].options()),h1=torch::empty_like(h0),z=torch::empty({n,2},a[0].options());
  auto counts=torch::zeros({anchors},a[8].options().dtype(torch::kInt32));
  auto members=torch::empty({anchors,10},a[8].options()),context=torch::empty({anchors,17},a[0].options());
  auto stream=c10::cuda::getCurrentCUDAStream();
  if(n){first<<<(n+7)/8,256,0,stream>>>(n,ptr(a[0]),ptr(a[1]),ptr(a[2]),h0.data_ptr<float>());
    group<<<(n+127)/128,128,0,stream>>>(n,anchors,a[8].data_ptr<int64_t>(),counts.data_ptr<int>(),members.data_ptr<int64_t>());}
  if(anchors)pool<<<(anchors+3)/4,128,0,stream>>>(anchors,ptr(h0),a[9].data_ptr<bool>(),counts.data_ptr<int>(),members.data_ptr<int64_t>(),context.data_ptr<float>());
  auto residual=context.mm(a[7].t());
  if(n)tail<<<(n+7)/8,256,0,stream>>>(n,ptr(h0),ptr(a[3]),ptr(a[4]),ptr(a[5]),ptr(a[6]),a[8].data_ptr<int64_t>(),ptr(residual),h1.data_ptr<float>(),z.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {z,h0,h1,context,members,counts};
}
std::vector<torch::Tensor> anchorContextMLPBackward(const std::vector<torch::Tensor>& a,
    const torch::Tensor& logits_gradient,const torch::Tensor& shared_gradient) {
  TORCH_CHECK(a.size()==15,"ContextMLP saved tensors");const int n=a[0].size(0),anchors=a[12].size(0);
  check(a,anchors);c10::cuda::CUDAGuard guard(a[0].device());auto stream=c10::cuda::getCurrentCUDAStream();
  auto gz=logits_gradient.defined()?logits_gradient:torch::zeros({n,2},a[0].options()),d1=torch::empty_like(a[11]);
  if(n)finalVjp<<<(n*32+255)/256,256,0,stream>>>(n,ptr(gz),ptr(a[5]),ptr(a[11]),d1.data_ptr<float>());
  auto dw2=gz.t().mm(a[11]),db2=gz.sum(0),dw1=d1.t().mm(a[10]),db1=d1.sum(0),d0=d1.mm(a[3]);
  auto sums=torch::empty({anchors,32},a[0].options());
  if(anchors)gatherVjp<<<(anchors+3)/4,128,0,stream>>>(anchors,a[14].data_ptr<int>(),a[13].data_ptr<int64_t>(),ptr(d1),sums.data_ptr<float>());
  auto dp=sums.t().mm(a[12]),cg=sums.mm(a[7]);
  if(anchors)poolVjp<<<(anchors+3)/4,128,0,stream>>>(anchors,ptr(a[10]),a[9].data_ptr<bool>(),a[14].data_ptr<int>(),
      a[13].data_ptr<int64_t>(),ptr(a[12]),ptr(cg),d0.data_ptr<float>());
  if(n)firstVjp<<<(n*32+255)/256,256,0,stream>>>(n,ptr(a[10]),ptr(shared_gradient),d0.data_ptr<float>());
  auto dw0=d0.t().mm(a[0]),db0=d0.sum(0),dx=d0.mm(a[1]);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {dx,dw0,db0,dw1,db1,dw2,db2,dp,{}, {}};
}
std::vector<torch::Tensor> anchorContextMLPForwardOptimized(const std::vector<torch::Tensor>& a,int64_t anchors,int64_t options) {
  TORCH_CHECK(options==1 || options==3 || options==5 || options==7,"ContextMLP optimization options");
  check(a,anchors);c10::cuda::CUDAGuard guard(a[0].device());const int n=a[0].size(0);
  auto h0=torch::empty({n,32},a[0].options()),h1=torch::empty_like(h0),z=torch::empty({n,2},a[0].options());
  auto counts=torch::zeros({anchors},a[8].options().dtype(torch::kInt32));
  auto members=torch::empty({anchors,10},a[8].options()),context=torch::empty({anchors,17},a[0].options());
  const int64_t cache_stride=(anchors+31)/32*32;
  torch::Tensor metadata;if(!(options&4))metadata=torch::empty({cache_stride+anchors*8},counts.options());
  auto stream=c10::cuda::getCurrentCUDAStream();
  if(n){first<<<(n+7)/8,256,0,stream>>>(n,ptr(a[0]),ptr(a[1]),ptr(a[2]),h0.data_ptr<float>());
    group<<<(n+127)/128,128,0,stream>>>(n,anchors,a[8].data_ptr<int64_t>(),counts.data_ptr<int>(),members.data_ptr<int64_t>());}
  if(anchors){
    if(options&4)poolCached<true><<<(anchors+3)/4,128,0,stream>>>(anchors,ptr(h0),a[9].data_ptr<bool>(),counts.data_ptr<int>(),
        members.data_ptr<int64_t>(),context.data_ptr<float>(),nullptr,nullptr);
    else poolCached<false><<<(anchors+3)/4,128,0,stream>>>(anchors,ptr(h0),a[9].data_ptr<bool>(),counts.data_ptr<int>(),
        members.data_ptr<int64_t>(),context.data_ptr<float>(),metadata.data_ptr<int>(),metadata.data_ptr<int>()+cache_stride);
  }
  auto residual=context.mm(a[7].t());
  if(n)tail<<<(n+7)/8,256,0,stream>>>(n,ptr(h0),ptr(a[3]),ptr(a[4]),ptr(a[5]),ptr(a[6]),a[8].data_ptr<int64_t>(),ptr(residual),h1.data_ptr<float>(),z.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if(options&4)return {z,h0,h1,context,members,counts};
  return {z,h0,h1,context,members,counts,metadata};
}
std::vector<torch::Tensor> anchorContextMLPBackwardOptimized(const std::vector<torch::Tensor>& a,
    const torch::Tensor& logits_gradient,const torch::Tensor& shared_gradient,int64_t options) {
  TORCH_CHECK(a.size()==((options&4)?15:16) && (options==1 || options==3 || options==5 || options==7),"Cached ContextMLP saved tensors");
  const int n=a[0].size(0),anchors=a[12].size(0);check(a,anchors);
  const int64_t cache_stride=(int64_t(anchors)+31)/32*32;
  c10::cuda::CUDAGuard guard(a[0].device());auto stream=c10::cuda::getCurrentCUDAStream();
  auto gz=logits_gradient.defined()?logits_gradient:torch::zeros({n,2},a[0].options()),d1=torch::empty_like(a[11]);
  auto sums=torch::empty({anchors,32},a[0].options());
  if(options&2){
    if(anchors){
      if(options&4)finalGatherVjp<true><<<(anchors+3)/4,128,0,stream>>>(anchors,a[14].data_ptr<int>(),a[13].data_ptr<int64_t>(),ptr(gz),ptr(a[5]),ptr(a[11]),d1.data_ptr<float>(),sums.data_ptr<float>());
      else finalGatherVjp<false><<<(anchors+3)/4,128,0,stream>>>(anchors,a[14].data_ptr<int>(),a[13].data_ptr<int64_t>(),ptr(gz),ptr(a[5]),ptr(a[11]),d1.data_ptr<float>(),sums.data_ptr<float>());
    }
  }else if(n)finalVjp<<<(n*32+255)/256,256,0,stream>>>(n,ptr(gz),ptr(a[5]),ptr(a[11]),d1.data_ptr<float>());
  auto dw2=gz.t().mm(a[11]),db2=gz.sum(0),dw1=d1.t().mm(a[10]),db1=d1.sum(0),d0=d1.mm(a[3]);
  if(!(options&2) && anchors){
    if(options&4)gatherCached<true><<<(anchors+3)/4,128,0,stream>>>(anchors,a[14].data_ptr<int>(),a[13].data_ptr<int64_t>(),ptr(d1),sums.data_ptr<float>());
    else gatherCached<false><<<(anchors+3)/4,128,0,stream>>>(anchors,a[14].data_ptr<int>(),a[13].data_ptr<int64_t>(),ptr(d1),sums.data_ptr<float>());
  }
  auto dp=sums.t().mm(a[12]),cg=sums.mm(a[7]);
  if(anchors){
    if(options&4)poolVjpCached<true><<<(anchors+3)/4,128,0,stream>>>(anchors,a[9].data_ptr<bool>(),a[14].data_ptr<int>(),a[13].data_ptr<int64_t>(),
        nullptr,nullptr,ptr(cg),d0.data_ptr<float>());
    else poolVjpCached<false><<<(anchors+3)/4,128,0,stream>>>(anchors,a[9].data_ptr<bool>(),a[14].data_ptr<int>(),a[13].data_ptr<int64_t>(),
        a[15].data_ptr<int>(),a[15].data_ptr<int>()+cache_stride,ptr(cg),d0.data_ptr<float>());
  }
  if(n)firstVjp<<<(n*32+255)/256,256,0,stream>>>(n,ptr(a[10]),ptr(shared_gradient),d0.data_ptr<float>());
  auto dw0=d0.t().mm(a[0]),db0=d0.sum(0),dx=d0.mm(a[1]);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {dx,dw0,db0,dw1,db1,dw2,db2,dp,{}, {}};
}
}

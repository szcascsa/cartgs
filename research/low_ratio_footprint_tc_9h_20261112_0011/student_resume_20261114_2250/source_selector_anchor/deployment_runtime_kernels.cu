#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <cub/block/block_reduce.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include "deployment_device_control.h"

void deployment_select_device(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
void deployment_radix_device(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const std::vector<torch::Tensor>&,bool,bool,bool);

namespace {
struct Weights { const float* p[21]; };
struct Tables { const float* p[6]; int count; };
struct Master { const float* p[6]; int rest; };
struct Controls { float weights[6], ratio[3]; };
__device__ float relu(float x) { return x <= 0.f ? 0.f : x; }
__device__ float clamp(float x, float lo, float hi) {
  return isnan(x) ? x : fminf(fmaxf(x, lo), hi);
}
void check(const torch::Tensor& x, const torch::Tensor& ref, at::ScalarType type=at::kFloat) {
  TORCH_CHECK(x.is_cuda() && x.device()==ref.device() && x.scalar_type()==type &&
      x.is_contiguous(), "Deployment requires contiguous same-device FP32 tensors");
}
__device__ float dot4(const float* p, const float* m, int col, int64_t s0, int64_t s1) {
  float v=0.f;
  for(int j=0;j<4;++j)v=fmaf(p[j],m[j*s0+col*s1],v);
  return v;
}
__device__ float safeW(float x,float near) { return fabsf(x)<near?(x<0.f?-near:near):x; }
__global__ void viewKernel(int64_t n,const float* centers,const float* centers4,
    const float* corners,const float* camera,const float* vm,int64_t v0,int64_t v1,
    const float* pm,int64_t p0,int64_t p1,const float* scale,float aspect,float near,float* out,
    const DeploymentFrame* frame,const DeploymentRequest* request=nullptr,const bool* visible=nullptr) {
  if(frame && !frame->k)return;
  if(request){aspect=request->aspect;near=request->near;}
  const auto a=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(a>=n)return;
  if(visible && !visible[a]) {
    // Padded, inactive MLP rows use Anchor zero without publishing any output.
    if(!a)for(int j=0;j<7;++j)out[j]=0.f;
    return;
  }
  float v[3];for(int j=0;j<3;++j)v[j]=camera[j]-centers[3*a+j];
  const float norm=fmaxf(sqrtf((v[0]*v[0]+v[2]*v[2])+v[1]*v[1]),1e-6f);
  for(int j=0;j<3;++j)out[7*a+j]=v[j]/norm;
  out[7*a+3]=asinhf(dot4(centers4+4*a,vm,2,v0,v1)/scale[0]);
  const float w=safeW(dot4(centers4+4*a,pm,3,p0,p1),near);
  for(int j=0;j<2;++j)out[7*a+4+j]=clamp(dot4(centers4+4*a,pm,j,p0,p1)/w,-4.f,4.f);
  float lo[2]={CUDART_INF_F,CUDART_INF_F},hi[2]={-CUDART_INF_F,-CUDART_INF_F};
  float min_depth=CUDART_INF_F,max_depth=-CUDART_INF_F;
  for(int k=0;k<8;++k) {
    const float* p=corners+a*32+k*4;
    const float depth=dot4(p,vm,2,v0,v1),cw=safeW(dot4(p,pm,3,p0,p1),near);
    min_depth=fminf(min_depth,depth);max_depth=fmaxf(max_depth,depth);
    for(int j=0;j<2;++j){const float x=dot4(p,pm,j,p0,p1)/cw;lo[j]=fminf(lo[j],x);hi[j]=fmaxf(hi[j],x);}
  }
  float longest=clamp(fmaxf((hi[0]-lo[0])*aspect*.5f,(hi[1]-lo[1])*.5f),0.f,1e4f);
  if(min_depth<=near)longest=1e4f;
  if(max_depth<=near)longest=0.f;
  out[7*a+6]=log1pf(longest);
}

#include "student_view_project.cuh"

__global__ void selectorPrefix(int64_t n,const float* physical,const float* weights,float* out) {
  const int64_t row=int64_t(blockIdx.x)*4+threadIdx.x/32;
  const int col=threadIdx.x%32;
  if(row>=n)return;
  float sum=0.f;
  #pragma unroll
  for(int j=0;j<10;++j)sum=fmaf(physical[row*10+j],weights[j*32+col],sum);
  out[row*32+col]=sum;
}

template<int Width,bool Prefix=false> __global__ void selectorKernel(int64_t n,const float* physical,
    const int64_t* rows,const int64_t* owners,const float* view,const float* noise,
    Weights w,float log_ratio,bool local,bool real_view,float* scores,float* logits,int* status,
    const DeploymentFrame* frame,const float* prefix=nullptr) {
  if(frame){if(!frame->k)return;n=frame->n;log_ratio=frame->log_ratio;}
  const int warp=threadIdx.x/32,lane=threadIdx.x%32;
  const int64_t row=int64_t(blockIdx.x)*4+warp;
  if(row>=n)return;
  const auto id=rows[row];
  __shared__ float h0[4][32],h1[4][32];
  float x=0.f;
  if(lane<10){if(!Prefix)x=physical[id*10+lane];}
  else if(lane==10)x=log_ratio;
  else if(Width==19 && lane<18)x=local&&real_view?view[owners[id]*7+lane-11]:0.f;
  else if(Width==19 && lane==18)x=local?1.f:0.f;
  float sum=Prefix?prefix[id*32+lane]:0.f;
  #pragma unroll
  for(int j=Prefix?10:0;j<Width;++j)sum=fmaf(__shfl_sync(0xffffffff,x,j),w.p[0][j*32+lane],sum);
  h0[warp][lane]=relu(sum+w.p[1][lane]);__syncwarp();
  sum=0.f;
  #pragma unroll
  for(int j=0;j<32;++j)sum=fmaf(h0[warp][j],w.p[2][j*32+lane],sum);
  h1[warp][lane]=relu(sum+w.p[3][lane]);__syncwarp();
  if(lane<2) {
    sum=0.f;
    #pragma unroll
    for(int j=0;j<32;++j)sum=fmaf(h1[warp][j],w.p[4][j*2+lane],sum);
    h0[warp][lane]=sum+w.p[5][lane];
    logits[row*2+lane]=h0[warp][lane];
    if(!isfinite(h0[warp][lane]))atomicExch(status,1);
  }
  __syncwarp();
  if(!lane) {
    const float a=h0[warp][0]+noise[id*2],b=h0[warp][1]+noise[id*2+1];
    const float maximum=fmaxf(a,b),ea=expf(a-maximum),eb=expf(b-maximum);
    scores[row]=eb/(ea+eb);
    if(!isfinite(scores[row]))atomicExch(status,1);
  }
}

template<int Width,bool Prefix=false> __global__ void selectorRegisterTile(int64_t n,const float* physical,
    const int64_t* rows,const int64_t* owners,const float* view,const float* noise,
    Weights w,float log_ratio,bool local,bool real_view,float* scores,float* logits,int* status,
    const DeploymentFrame* frame,const float* prefix=nullptr) {
  if(frame){if(!frame->k)return;n=frame->n;log_ratio=frame->log_ratio;}
  const int tid=threadIdx.x,point=(tid/32)*4,col=tid%32;
  const int64_t begin=int64_t(blockIdx.x)*16;
  if(begin>=n)return;
  constexpr int Start=Prefix?10:0,InputWidth=Width-Start;
  __shared__ float input[16][InputWidth],h0[16][32],h1[16][32],result[16][2];
  for(int i=tid;i<16*InputWidth;i+=128) {
    const int p=i/InputWidth,c=i%InputWidth+Start;float value=0.f;
    if(begin+p<n) {
      const auto id=rows[begin+p];
      if(c<10){if(!Prefix)value=physical[id*10+c];}
      else if(c==10)value=log_ratio;
      else if(c<18)value=local&&real_view?view[owners[id]*7+c-11]:0.f;
      else value=local?1.f:0.f;
    }
    input[p][c-Start]=value;
  }
  __syncthreads();
  float acc[4]={0.f,0.f,0.f,0.f};
  if(Prefix) {
    #pragma unroll
    for(int q=0;q<4;++q)if(begin+point+q<n)acc[q]=prefix[rows[begin+point+q]*32+col];
  }
  #pragma unroll
  for(int j=Prefix?10:0;j<Width;++j) {
    const float weight=w.p[0][j*32+col];
    #pragma unroll
    for(int q=0;q<4;++q)acc[q]=fmaf(input[point+q][j-Start],weight,acc[q]);
  }
  #pragma unroll
  for(int q=0;q<4;++q){h0[point+q][col]=relu(acc[q]+w.p[1][col]);acc[q]=0.f;}
  __syncthreads();
  #pragma unroll
  for(int j=0;j<32;++j) {
    const float weight=w.p[2][j*32+col];
    #pragma unroll
    for(int q=0;q<4;++q)acc[q]=fmaf(h0[point+q][j],weight,acc[q]);
  }
  #pragma unroll
  for(int q=0;q<4;++q)h1[point+q][col]=relu(acc[q]+w.p[3][col]);
  __syncthreads();
  if(tid<32) {
    const int p=tid/2,c=tid%2;
    if(begin+p<n) {
      float sum=0.f;
      #pragma unroll
      for(int j=0;j<32;++j)sum=fmaf(h1[p][j],w.p[4][j*2+c],sum);
      result[p][c]=sum+w.p[5][c];
      if(logits)logits[(begin+p)*2+c]=result[p][c];
      if(!isfinite(result[p][c]))atomicExch(status,1);
    }
  }
  __syncthreads();
  if(tid<16 && begin+tid<n) {
    const auto id=rows[begin+tid];
    const float a=result[tid][0]+noise[id*2],b=result[tid][1]+noise[id*2+1];
    const float maximum=fmaxf(a,b),ea=expf(a-maximum),eb=expf(b-maximum);
    scores[begin+tid]=eb/(ea+eb);
    if(!isfinite(scores[begin+tid]))atomicExch(status,1);
  }
}
__device__ void normalize(float* q) {
  float n=0.f;for(int j=0;j<4;++j)n=fmaf(q[j],q[j],n);
  n=fmaxf(sqrtf(n),1e-12f);for(int j=0;j<4;++j)q[j]/=n;
}
template<bool Direct=false,bool Rest=true> __device__ void publishPoint(int64_t row,int64_t n,int64_t id,const float* d,
    Master master,bool softplus,float* output,int* status,int lane,int stride,
    const DeploymentRequest* request=nullptr) {
  if(Direct) {
    output=request->attributes;
    if(!lane)request->selected[row]=id;
  }
  float q[4];
  for(int j=0;j<4;++j)q[j]=d[6+j]+(j==0?1.f:0.f);
  normalize(q);
  const float* b=master.p[3]+id*4;
  float rotation[4];
  rotation[0]=q[0]*b[0]-((q[1]*b[1]+q[3]*b[3])+q[2]*b[2]);
  for(int j=1;j<4;++j){int k=j%3+1,l=(j+1)%3+1;rotation[j]=(q[0]*b[j]+b[0]*q[j])+fmaf(q[k],b[l],-q[l]*b[k]);}
  normalize(rotation);normalize(rotation);
  const int width=Rest?17+master.rest:17;
  for(int col=lane;col<width;col+=stride) {
    float value;int64_t dest;
    if(col<3){value=master.p[0][id*3+col]+d[col];dest=row*3+col;}
    else if(col==3){value=1.f/(1.f+expf(-(master.p[1][id]+d[10])));dest=n*3+row;}
    else if(col<7){int j=col-4;float factor=d[3+j];if(softplus){float z=factor+.5413248546129181f;factor=z>20.f?z:log1pf(expf(z));value=fmaxf(master.p[2][id*3+j]*factor,1.1754943508222875e-38f);}else value=master.p[2][id*3+j]+factor;value=expf(logf(value));dest=n*4+row*3+j;}
    else if(col<11){value=rotation[col-7];dest=n*7+row*4+col-7;}
    else if(col<14){value=master.p[4][id*3+col-11]+d[11+col-11];dest=n*11+row*3+col-11;}
    else if(Rest && col<14+master.rest){value=master.p[5][id*master.rest+col-14];dest=n*14+row*master.rest+col-14;}
    else{value=0.f;dest=n*(14+master.rest)+row*3+col-14-(Rest?master.rest:0);}
    output[dest]=value;if(!isfinite(value))atomicExch(status,2);
  }
}

// One block per Gaussian: trunk is shared; the four warps run four independent Heads.
template<bool Direct=false> __global__ void compensationBlock(int64_t n,const float* offset,const int64_t* rows,
    const int64_t* owners,const float* reference,Tables tables,Controls controls,
    Weights w,const float* view,const float* view_weight,bool local,bool real_view,
    Master master,bool softplus,float* output,float* residual,int* status,const DeploymentFrame* frame,const DeploymentRequest* request=nullptr) {
  // Selector status is complete on this stream. Value 1 is immutable during compensation;
  // do not exit on value 2, which could change mid-CTA and diverge around barriers.
  if(status[0]==1)return;
  if(frame) {
    n=frame->k;
    if(blockIdx.x>=n)return;
    for(int j=0;j<6;++j)controls.weights[j]=frame->weights[j];
    for(int j=0;j<3;++j)controls.ratio[j]=frame->ratio_code[j];
  }
  const auto row=int64_t(blockIdx.x),id=rows[row],anchor=owners[id];
  const int tid=threadIdx.x,lane=tid%32,head=tid/32;
  __shared__ float input[51],h0[64],h1[64],h2[4][64],delta[14];
  if(tid<51) {
    float value;
    if(tid<3)value=offset[id*3+tid];
    else{const auto a=anchor*48+tid-3;value=tables.p[0][a];for(int j=1;j<tables.count;++j){if(controls.weights[j]<0.f)break;value=__fadd_rn(value,__fmul_rn(controls.weights[j],tables.p[j][a]));}}
    input[tid]=value;
  }
  __syncthreads();
  if(tid<64) {
    float sum=0.f;for(int j=0;j<51;++j)sum=fmaf(input[j],w.p[0][j*64+tid],sum);
    sum+=w.p[1][tid];float rp=0.f;for(int j=0;j<3;++j)rp=fmaf(controls.ratio[j],w.p[4][j*64+tid],rp);sum+=rp;
    if(local){float vp=0.f;for(int j=0;j<7;++j)vp=fmaf(real_view?view[anchor*7+j]:0.f,view_weight[j*64+tid],vp);vp+=view_weight[7*64+tid];sum+=vp;}
    h0[tid]=relu(sum);
  }
  __syncthreads();
  if(tid<64){float sum=0.f;for(int j=0;j<64;++j)sum=fmaf(h0[j],w.p[2][j*64+tid],sum);h1[tid]=relu(sum+w.p[3][tid]);}
  __syncthreads();
  const int p=5+4*head,head_width=head<2?3:4,head_offset=head==0?0:head==1?3:head==2?6:10;
  for(int k=0;k<2;++k){const int col=lane+32*k;float sum=0.f;for(int j=0;j<64;++j)sum=fmaf(h1[j],w.p[p][j*64+col],sum);h2[head][col]=relu(sum+w.p[p+1][col]);}
  __syncwarp();
  if(lane<head_width){float sum=0.f;for(int j=0;j<64;++j)sum=fmaf(h2[head][j],w.p[p+2][j*head_width+lane],sum);sum+=w.p[p+3][lane];if(head==0)sum*=reference[id];delta[head_offset+lane]=sum;if(residual)residual[row*14+head_offset+lane]=sum;if(!isfinite(sum))atomicExch(status,2);}
  __syncthreads();
  publishPoint<Direct>(row,n,id,delta,master,softplus,output,status,tid,128,request);
}

// A warp owns one Gaussian. A bounded tile shares the current layer's weights;
// all dot products retain the original scalar FMA sequence and four Head weights.
template<int Points,bool Direct=false> __global__ void compensationTile(int64_t n,const float* offset,const int64_t* rows,
    const int64_t* owners,const float* reference,Tables tables,Controls controls,
    Weights w,const float* view,const float* view_weight,bool local,bool real_view,
    Master master,bool softplus,float* output,float* residual,int* status,const DeploymentFrame* frame,
    const DeploymentRequest* request=nullptr) {
  if(status[0]==1)return;
  if(frame) {
    n=frame->k;
    if(int64_t(blockIdx.x)*Points>=n)return;
    for(int j=0;j<6;++j)controls.weights[j]=frame->weights[j];
    for(int j=0;j<3;++j)controls.ratio[j]=frame->ratio_code[j];
  }
  const int tid=threadIdx.x,lane=tid&31,point=tid/32;
  const int64_t row=int64_t(blockIdx.x)*Points+point;
  const bool active=row<n;
  const int64_t id=active?rows[row]:0,anchor=active?owners[id]:0;
  __shared__ float weights[4096],input[Points][51],h0[Points][64],h1[Points][64],h2[Points][64],delta[Points][14];
  if(active)for(int j=lane;j<51;j+=32) {
    float value;
    if(j<3)value=offset[id*3+j];
    else{const auto a=anchor*48+j-3;value=tables.p[0][a];for(int k=1;k<tables.count;++k){if(controls.weights[k]<0.f)break;value=__fadd_rn(value,__fmul_rn(controls.weights[k],tables.p[k][a]));}}
    input[point][j]=value;
  }
  for(int j=tid;j<51*64;j+=Points*32)weights[j]=w.p[0][j];
  __syncthreads();
  if(active)for(int c=lane;c<64;c+=32) {
    float sum=0.f;for(int j=0;j<51;++j)sum=fmaf(input[point][j],weights[j*64+c],sum);
    sum+=w.p[1][c];float rp=0.f;for(int j=0;j<3;++j)rp=fmaf(controls.ratio[j],w.p[4][j*64+c],rp);sum+=rp;
    if(local){float vp=0.f;for(int j=0;j<7;++j)vp=fmaf(real_view?view[anchor*7+j]:0.f,view_weight[j*64+c],vp);vp+=view_weight[7*64+c];sum+=vp;}
    h0[point][c]=relu(sum);
  }
  __syncthreads();
  for(int j=tid;j<4096;j+=Points*32)weights[j]=w.p[2][j];
  __syncthreads();
  if(active)for(int c=lane;c<64;c+=32){float sum=0.f;for(int j=0;j<64;++j)sum=fmaf(h0[point][j],weights[j*64+c],sum);h1[point][c]=relu(sum+w.p[3][c]);}
  __syncthreads();
  for(int head=0;head<4;++head) {
    const int p=5+4*head,width=head<2?3:4,begin=head==0?0:head==1?3:head==2?6:10;
    for(int j=tid;j<4096;j+=Points*32)weights[j]=w.p[p][j];
    __syncthreads();
    if(active)for(int c=lane;c<64;c+=32){float sum=0.f;for(int j=0;j<64;++j)sum=fmaf(h1[point][j],weights[j*64+c],sum);h2[point][c]=relu(sum+w.p[p+1][c]);}
    __syncwarp();
    if(active && lane<width){float sum=0.f;for(int j=0;j<64;++j)sum=fmaf(h2[point][j],w.p[p+2][j*width+lane],sum);sum+=w.p[p+3][lane];if(!head)sum*=reference[id];delta[point][begin+lane]=sum;if(residual)residual[row*14+begin+lane]=sum;if(!isfinite(sum))atomicExch(status,2);}
    __syncthreads();
  }
  if(active)publishPoint<Direct>(row,n,id,delta[point],master,softplus,output,status,lane,32,request);
}

// Each thread reuses one FP32 weight across four independent Gaussian dots.
// Every individual dot retains v11's FMA order; no partial-sum reduction is used.
template<int Points,bool Direct=false> __global__ void compensationRegisterTile(int64_t n,const float* offset,const int64_t* rows,
    const int64_t* owners,const float* reference,Tables tables,Controls controls,
    Weights w,const float* view,const float* view_weight,bool local,bool real_view,
    Master master,bool softplus,float* output,float* residual,int* status,const DeploymentFrame* frame,
    const DeploymentRequest* request=nullptr) {
  if(status[0]==1)return;
  if(frame) {
    n=frame->k;
    if(int64_t(blockIdx.x)*Points>=n)return;
    for(int j=0;j<6;++j)controls.weights[j]=frame->weights[j];
    for(int j=0;j<3;++j)controls.ratio[j]=frame->ratio_code[j];
  }
  const int tid=threadIdx.x,point=(tid/64)*4,col=tid%64;
  const int64_t begin=int64_t(blockIdx.x)*Points;
  __shared__ float input[Points][51],h0[Points][64],h1[Points][64],h2[Points][64],delta[Points][14];
  __shared__ int64_t ids[Points],anchors[Points];
  if(tid<Points){ids[tid]=begin+tid<n?rows[begin+tid]:0;anchors[tid]=begin+tid<n?owners[ids[tid]]:0;}
  __syncthreads();
  for(int i=tid;i<Points*51;i+=Points*16) {
    const int p=i/51,c=i%51;float value=0.f;
    if(begin+p<n) {
      if(c<3)value=offset[ids[p]*3+c];
      else{const auto a=anchors[p]*48+c-3;value=tables.p[0][a];for(int j=1;j<tables.count;++j){if(controls.weights[j]<0.f)break;value=__fadd_rn(value,__fmul_rn(controls.weights[j],tables.p[j][a]));}}
    }
    input[p][c]=value;
  }
  __syncthreads();
  float accum[4]={0.f,0.f,0.f,0.f};
  for(int j=0;j<51;++j) {
    const float weight=w.p[0][j*64+col];
    #pragma unroll
    for(int q=0;q<4;++q)accum[q]=fmaf(input[point+q][j],weight,accum[q]);
  }
  float rp=0.f;for(int j=0;j<3;++j)rp=fmaf(controls.ratio[j],w.p[4][j*64+col],rp);
  #pragma unroll
  for(int q=0;q<4;++q) {
    float value=accum[q]+w.p[1][col];value+=rp;
    if(local){float vp=0.f;for(int j=0;j<7;++j)vp=fmaf(real_view?view[anchors[point+q]*7+j]:0.f,view_weight[j*64+col],vp);vp+=view_weight[7*64+col];value+=vp;}
    h0[point+q][col]=relu(value);accum[q]=0.f;
  }
  __syncthreads();
  for(int j=0;j<64;++j) {
    const float weight=w.p[2][j*64+col];
    #pragma unroll
    for(int q=0;q<4;++q)accum[q]=fmaf(h0[point+q][j],weight,accum[q]);
  }
  #pragma unroll
  for(int q=0;q<4;++q)h1[point+q][col]=relu(accum[q]+w.p[3][col]);
  __syncthreads();
  for(int head=0;head<4;++head) {
    const int p=5+4*head,width=head<2?3:4,start=head==0?0:head==1?3:head==2?6:10;
    #pragma unroll
    for(int q=0;q<4;++q)accum[q]=0.f;
    for(int j=0;j<64;++j) {
      const float weight=w.p[p][j*64+col];
      #pragma unroll
      for(int q=0;q<4;++q)accum[q]=fmaf(h1[point+q][j],weight,accum[q]);
    }
    #pragma unroll
    for(int q=0;q<4;++q)h2[point+q][col]=relu(accum[q]+w.p[p+1][col]);
    __syncthreads();
    if(tid<Points*width) {
      const int row=tid/width,c=tid%width;float sum=0.f;
      for(int j=0;j<64;++j)sum=fmaf(h2[row][j],w.p[p+2][j*width+c],sum);
      sum+=w.p[p+3][c];if(!head && begin+row<n)sum*=reference[ids[row]];
      delta[row][start+c]=sum;
      if(begin+row<n){if(residual)residual[(begin+row)*14+start+c]=sum;if(!isfinite(sum))atomicExch(status,2);}
    }
    __syncthreads();
  }
  const int row=tid/16;
  if(begin+row<n)publishPoint<Direct>(begin+row,n,ids[row],delta[row],master,softplus,output,status,tid%16,16,request);
}


template<int Points,bool Direct=false,int Accum=4> __global__ void compensationRegisterUnrolled(int64_t n,const float* offset,const int64_t* rows,
    const int64_t* owners,const float* reference,Tables tables,Controls controls,
    Weights w,const float* view,const float* view_weight,bool local,bool real_view,
    Master master,bool softplus,float* output,float* residual,int* status,const DeploymentFrame* frame,
    const DeploymentRequest* request=nullptr) {
  static_assert(Points*64/Accum==256,"Register tile requires 256 threads");
  if(status[0]==1)return;
  if(frame) {
    n=frame->k;
    if(int64_t(blockIdx.x)*Points>=n)return;
    for(int j=0;j<6;++j)controls.weights[j]=frame->weights[j];
    for(int j=0;j<3;++j)controls.ratio[j]=frame->ratio_code[j];
  }
  const int tid=threadIdx.x,point=(tid/64)*Accum,col=tid%64;
  const int64_t begin=int64_t(blockIdx.x)*Points;
  __shared__ float input[Points][51],h0[Points][64],h1[Points][64],h2[Points][64],delta[Points][14];
  __shared__ int64_t ids[Points],anchors[Points];
  if(tid<Points){ids[tid]=begin+tid<n?rows[begin+tid]:0;anchors[tid]=begin+tid<n?owners[ids[tid]]:0;}
  __syncthreads();
  for(int i=tid;i<Points*51;i+=Points*64/Accum) {
    const int p=i/51,c=i%51;float value=0.f;
    if(begin+p<n) {
      if(c<3)value=offset[ids[p]*3+c];
      else{const auto a=anchors[p]*48+c-3;value=tables.p[0][a];for(int j=1;j<tables.count;++j){if(controls.weights[j]<0.f)break;value=__fadd_rn(value,__fmul_rn(controls.weights[j],tables.p[j][a]));}}
    }
    input[p][c]=value;
  }
  __syncthreads();
  float accum[Accum]={};
  #pragma unroll 64
  for(int j=0;j<51;++j) {
    const float weight=w.p[0][j*64+col];
    #pragma unroll
    for(int q=0;q<Accum;++q)accum[q]=fmaf(input[point+q][j],weight,accum[q]);
  }
  float rp=0.f;for(int j=0;j<3;++j)rp=fmaf(controls.ratio[j],w.p[4][j*64+col],rp);
  #pragma unroll
  for(int q=0;q<Accum;++q) {
    float value=accum[q]+w.p[1][col];value+=rp;
    if(local){float vp=0.f;for(int j=0;j<7;++j)vp=fmaf(real_view?view[anchors[point+q]*7+j]:0.f,view_weight[j*64+col],vp);vp+=view_weight[7*64+col];value+=vp;}
    h0[point+q][col]=relu(value);accum[q]=0.f;
  }
  __syncthreads();
  #pragma unroll 64
  for(int j=0;j<64;++j) {
    const float weight=w.p[2][j*64+col];
    #pragma unroll
    for(int q=0;q<Accum;++q)accum[q]=fmaf(h0[point+q][j],weight,accum[q]);
  }
  #pragma unroll
  for(int q=0;q<Accum;++q)h1[point+q][col]=relu(accum[q]+w.p[3][col]);
  __syncthreads();
  for(int head=0;head<4;++head) {
    const int p=5+4*head,width=head<2?3:4,start=head==0?0:head==1?3:head==2?6:10;
    #pragma unroll
    for(int q=0;q<Accum;++q)accum[q]=0.f;
    #pragma unroll 64
    for(int j=0;j<64;++j) {
      const float weight=w.p[p][j*64+col];
      #pragma unroll
      for(int q=0;q<Accum;++q)accum[q]=fmaf(h1[point+q][j],weight,accum[q]);
    }
    #pragma unroll
    for(int q=0;q<Accum;++q)h2[point+q][col]=relu(accum[q]+w.p[p+1][col]);
    __syncthreads();
    if(tid<Points*width) {
      const int row=tid/width,c=tid%width;float sum=0.f;
      #pragma unroll 64
      for(int j=0;j<64;++j)sum=fmaf(h2[row][j],w.p[p+2][j*width+c],sum);
      sum+=w.p[p+3][c];if(!head && begin+row<n)sum*=reference[ids[row]];
      delta[row][start+c]=sum;
      if(begin+row<n){if(residual)residual[(begin+row)*14+start+c]=sum;if(!isfinite(sum))atomicExch(status,2);}
    }
    __syncthreads();
  }
  const int row=tid/(64/Accum);
  if(begin+row<n)publishPoint<Direct>(begin+row,n,ids[row],delta[row],master,softplus,output,status,tid%(64/Accum),64/Accum,request);
}

// Distilled 32/32 trunk with four independent linear output Heads.
// The frozen input packing and physical publication are shared with the teacher.
template<int Points,bool Direct=false> __global__ void compensationStudent(int64_t n,const float* offset,
    const int64_t* rows,const int64_t* owners,const float* reference,Tables tables,Controls controls,
    Weights w,const float* view,const float* view_weight,bool local,bool real_view,
    Master master,bool softplus,float* output,float* residual,int* status,const DeploymentFrame* frame,
    const DeploymentRequest* request=nullptr) {
  if(status[0]==1)return;
  if(frame) {
    n=frame->k;
    if(int64_t(blockIdx.x)*Points>=n)return;
    for(int j=0;j<6;++j)controls.weights[j]=frame->weights[j];
    for(int j=0;j<3;++j)controls.ratio[j]=frame->ratio_code[j];
  }
  constexpr int Accum=Points/4;
  const int tid=threadIdx.x,col=tid%32,point=(tid/32)*Accum;
  const int64_t begin=int64_t(blockIdx.x)*Points;
  __shared__ float input[Points][51],h0[Points][32],h1[Points][32],delta[Points][14];
  __shared__ int64_t ids[Points],anchors[Points];
  if(tid<Points){ids[tid]=begin+tid<n?rows[begin+tid]:0;anchors[tid]=begin+tid<n?owners[ids[tid]]:0;}
  __syncthreads();
  for(int i=tid;i<Points*51;i+=128) {
    const int p=i/51,c=i%51;float value=0.f;
    if(begin+p<n) {
      if(c<3)value=offset[ids[p]*3+c];
      else {
        const auto a=anchors[p]*48+c-3;value=tables.p[0][a];
        for(int j=1;j<tables.count;++j) {
          if(controls.weights[j]<0.f)break;
          value=__fadd_rn(value,__fmul_rn(controls.weights[j],tables.p[j][a]));
        }
      }
    }
    input[p][c]=value;
  }
  __syncthreads();
  float accum[Accum]={};
  #pragma unroll 64
  for(int j=0;j<51;++j) {
    const float weight=w.p[0][j*32+col];
    #pragma unroll
    for(int q=0;q<Accum;++q)accum[q]=fmaf(input[point+q][j],weight,accum[q]);
  }
  float rp=0.f;for(int j=0;j<3;++j)rp=fmaf(controls.ratio[j],w.p[4][j*32+col],rp);
  #pragma unroll
  for(int q=0;q<Accum;++q) {
    float value=accum[q]+w.p[1][col];value+=rp;
    if(local) {
      float vp=0.f;
      for(int j=0;j<7;++j)vp=fmaf(real_view?view[anchors[point+q]*7+j]:0.f,view_weight[j*32+col],vp);
      vp+=view_weight[7*32+col];value+=vp;
    }
    h0[point+q][col]=relu(value);accum[q]=0.f;
  }
  __syncthreads();
  #pragma unroll 32
  for(int j=0;j<32;++j) {
    const float weight=w.p[2][j*32+col];
    #pragma unroll
    for(int q=0;q<Accum;++q)accum[q]=fmaf(h0[point+q][j],weight,accum[q]);
  }
  #pragma unroll
  for(int q=0;q<Accum;++q)h1[point+q][col]=relu(accum[q]+w.p[3][col]);
  __syncthreads();
  for(int i=tid;i<Points*14;i+=128) {
    const int p=i/14,c=i%14;
    const int head=c<3?0:c<6?1:c<10?2:3;
    const int first=head==0?0:head==1?3:head==2?6:10;
    const int width=head<2?3:4;float value=0.f;
    #pragma unroll 32
    for(int j=0;j<32;++j)value=fmaf(h1[p][j],w.p[5+2*head][j*width+c-first],value);
    value+=w.p[6+2*head][c-first];
    if(!head && begin+p<n)value*=reference[ids[p]];
    delta[p][c]=value;
    if(begin+p<n){if(residual)residual[(begin+p)*14+c]=value;if(!isfinite(value))atomicExch(status,2);}
  }
  __syncthreads();
  const int row=tid/(128/Points);
  if(begin+row<n)publishPoint<Direct>(begin+row,n,ids[row],delta[row],master,softplus,output,status,
      tid%(128/Points),128/Points,request);
}

#include "tensor_networks.cuh"
#include "selector_anchor.cuh"
}

template<class Launch> void dispatchTensor(int64_t mode,int64_t warps,Launch launch){
  auto block=[&](auto m){
    if(warps==1)launch(m,std::integral_constant<int,1>{});
    else if(warps==2)launch(m,std::integral_constant<int,2>{});
    else launch(m,std::integral_constant<int,4>{});
  };
  if(mode==1)block(std::integral_constant<int,1>{});
  else block(std::integral_constant<int,2>{});
}

std::vector<torch::Tensor> deployment_tensor_pack(const std::vector<torch::Tensor>& selector,
    const std::vector<torch::Tensor>& student,const torch::Tensor& view_weight,int64_t selector_mode,int64_t compensation_mode){
  TORCH_CHECK(selector.size()==6,"Tensor Selector snapshot");
  TORCH_CHECK(!compensation_mode||student.size()==13,"Tensor Compensation requires a 32/32 student");
  c10::cuda::CUDAGuard guard(selector[0].device());
  cudaDeviceProp prop{};C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,selector[0].get_device()));
  TORCH_CHECK((!selector_mode&&!compensation_mode)||prop.major>=8,"Tensor inference requires Ampere or newer");
  Weights sw{},cw{};for(int j=0;j<6;++j)sw.p[j]=selector[j].data_ptr<float>();
  for(size_t j=0;j<student.size();++j)cw.p[j]=student[j].data_ptr<float>();
  std::vector<torch::Tensor> result(5);
  for(int which=0;which<5;++which){
    int mode=(which==0||which==4)?selector_mode:compensation_mode;if(!mode)continue;
    if(mode>=3)mode=2;
    int size=which==1?64*32:which==3||which==4?32*16:32*32;
    result[which]=torch::empty({size},selector[0].options().dtype(mode==1?at::kFloat:at::kHalf));
    if(mode==1)packTensorWeights<1><<<(size+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(sw,cw,view_weight.numel()?view_weight.data_ptr<float>():nullptr,selector[0].size(0),which,result[which].data_ptr<float>());
    else packTensorWeights<2><<<(size+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(sw,cw,view_weight.numel()?view_weight.data_ptr<float>():nullptr,selector[0].size(0),which,reinterpret_cast<__half*>(result[which].data_ptr<at::Half>()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}


#include "tensor_components.inc"

torch::Tensor deployment_view(const torch::Tensor& centers,const torch::Tensor& centers4,
    const torch::Tensor& corners,const torch::Tensor& camera,const torch::Tensor& vm,
    const torch::Tensor& pm,const torch::Tensor& scale,double aspect,double near) {
  check(centers,centers);check(centers4,centers);check(corners,centers);check(camera,centers);check(scale,centers);
  const auto anchors=centers.size(0);
  TORCH_CHECK(!torch::GradMode::is_enabled() && centers.sizes()==at::IntArrayRef({anchors,3}) &&
      centers4.sizes()==at::IntArrayRef({anchors,4}) && corners.sizes()==at::IntArrayRef({anchors,8,4}) &&
      camera.numel()==3 && scale.numel()==1 && vm.device()==centers.device() && pm.device()==centers.device() &&
      vm.scalar_type()==at::kFloat && pm.scalar_type()==at::kFloat && vm.sizes()==at::IntArrayRef({4,4}) &&
      pm.sizes()==vm.sizes() && aspect>0 && near>0,"Frozen camera layout");
  c10::cuda::CUDAGuard guard(centers.device());const auto n=centers.size(0);auto out=torch::empty({n,7},centers.options());
  if(n)viewKernel<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(n,centers.data_ptr<float>(),centers4.data_ptr<float>(),corners.data_ptr<float>(),camera.data_ptr<float>(),vm.data_ptr<float>(),vm.stride(0),vm.stride(1),pm.data_ptr<float>(),pm.stride(0),pm.stride(1),scale.data_ptr<float>(),aspect,near,out.data_ptr<float>(),nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<torch::Tensor> deployment_selector_impl(const torch::Tensor& physical,const torch::Tensor& rows,
    const torch::Tensor& owners,const torch::Tensor& view,const torch::Tensor& noise,
    const std::vector<torch::Tensor>& weights,double ratio,bool local,int64_t schema,bool tiled,
    const torch::Tensor& prefix=torch::Tensor()) {
  TORCH_CHECK(!torch::GradMode::is_enabled() && weights.size()==6 && ratio>0 && ratio<=1,"Frozen Selector inputs");
  check(physical,physical);check(rows,physical,at::kLong);check(owners,physical,at::kLong);check(view,physical);check(noise,physical);
  c10::cuda::CUDAGuard guard(physical.device());Weights w{};for(int j=0;j<6;++j){check(weights[j],physical);w.p[j]=weights[j].data_ptr<float>();}
  const auto n=rows.numel();auto scores=torch::empty({n},physical.options()),logits=torch::empty({n,2},physical.options()),status=torch::zeros({1},rows.options().dtype(at::kInt));
  const int width=weights[0].size(0);TORCH_CHECK(width==11 || width==19,"Selector width");
  TORCH_CHECK(physical.sizes()==at::IntArrayRef({owners.numel(),10}) && noise.sizes()==at::IntArrayRef({owners.numel(),2}) &&
      weights[0].size(1)==32 && weights[1].numel()==32 && weights[2].sizes()==at::IntArrayRef({32,32}) &&
      weights[3].numel()==32 && weights[4].sizes()==at::IntArrayRef({32,2}) && weights[5].numel()==2 &&
      (!local || !(schema==2||schema==3) || (view.dim()==2 && view.size(1)==7)),"Selector snapshot layout");
  const float lr=logf(float(ratio));const auto stream=c10::cuda::getCurrentCUDAStream();
  if(prefix.defined()) {
    check(prefix,physical);TORCH_CHECK(prefix.sizes()==at::IntArrayRef({owners.numel(),32}),"Selector prefix layout");
    if(n && tiled) {
      if(width==19)selectorRegisterTile<19,true><<<(n+15)/16,128,0,stream>>>(n,physical.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),view.data_ptr<float>(),noise.data_ptr<float>(),w,lr,local,schema==2||schema==3,scores.data_ptr<float>(),logits.data_ptr<float>(),status.data_ptr<int>(),nullptr,prefix.data_ptr<float>());
      else selectorRegisterTile<11,true><<<(n+15)/16,128,0,stream>>>(n,physical.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),view.data_ptr<float>(),noise.data_ptr<float>(),w,lr,local,false,scores.data_ptr<float>(),logits.data_ptr<float>(),status.data_ptr<int>(),nullptr,prefix.data_ptr<float>());
    } else if(n) {
      if(width==19)selectorKernel<19,true><<<(n+3)/4,128,0,stream>>>(n,physical.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),view.data_ptr<float>(),noise.data_ptr<float>(),w,lr,local,schema==2||schema==3,scores.data_ptr<float>(),logits.data_ptr<float>(),status.data_ptr<int>(),nullptr,prefix.data_ptr<float>());
      else selectorKernel<11,true><<<(n+3)/4,128,0,stream>>>(n,physical.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),view.data_ptr<float>(),noise.data_ptr<float>(),w,lr,local,false,scores.data_ptr<float>(),logits.data_ptr<float>(),status.data_ptr<int>(),nullptr,prefix.data_ptr<float>());
    }
  } else if(n && tiled) {
    if(width==19)selectorRegisterTile<19><<<(n+15)/16,128,0,stream>>>(n,physical.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),view.data_ptr<float>(),noise.data_ptr<float>(),w,lr,local,schema==2||schema==3,scores.data_ptr<float>(),logits.data_ptr<float>(),status.data_ptr<int>(),nullptr);
    else selectorRegisterTile<11><<<(n+15)/16,128,0,stream>>>(n,physical.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),view.data_ptr<float>(),noise.data_ptr<float>(),w,lr,local,false,scores.data_ptr<float>(),logits.data_ptr<float>(),status.data_ptr<int>(),nullptr);
  } else if(n){if(width==19)selectorKernel<19><<<(n+3)/4,128,0,stream>>>(n,physical.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),view.data_ptr<float>(),noise.data_ptr<float>(),w,lr,local,schema==2||schema==3,scores.data_ptr<float>(),logits.data_ptr<float>(),status.data_ptr<int>(),nullptr);
    else selectorKernel<11><<<(n+3)/4,128,0,stream>>>(n,physical.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),view.data_ptr<float>(),noise.data_ptr<float>(),w,lr,local,false,scores.data_ptr<float>(),logits.data_ptr<float>(),status.data_ptr<int>(),nullptr);}
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {scores,logits,status};
}

torch::Tensor deployment_selector_prefix(const torch::Tensor& physical,const torch::Tensor& weights) {
  check(physical,physical);check(weights,physical);
  TORCH_CHECK(!torch::GradMode::is_enabled() && physical.dim()==2 && physical.size(1)==10 &&
      weights.dim()==2 && weights.size(0)>=10 && weights.size(1)==32,"Frozen prefix layout");
  c10::cuda::CUDAGuard guard(physical.device());
  auto out=torch::empty({physical.size(0),32},physical.options());
  if(physical.size(0))selectorPrefix<<<(physical.size(0)+3)/4,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      physical.size(0),physical.data_ptr<float>(),weights.data_ptr<float>(),out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

torch::Tensor deployment_view_masked(const torch::Tensor& centers,const torch::Tensor& centers4,
    const torch::Tensor& corners,const torch::Tensor& camera,const torch::Tensor& vm,
    const torch::Tensor& pm,const torch::Tensor& scale,double aspect,double near,const torch::Tensor& visible) {
  check(visible,centers,at::kBool);
  TORCH_CHECK(visible.numel()==centers.size(0),"Anchor visibility layout");
  c10::cuda::CUDAGuard guard(centers.device());
  auto out=torch::empty({centers.size(0),7},centers.options());
  if(centers.size(0))viewKernel<<<(centers.size(0)+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(
      centers.size(0),centers.data_ptr<float>(),centers4.data_ptr<float>(),corners.data_ptr<float>(),camera.data_ptr<float>(),
      vm.data_ptr<float>(),vm.stride(0),vm.stride(1),pm.data_ptr<float>(),pm.stride(0),pm.stride(1),
      scale.data_ptr<float>(),aspect,near,out.data_ptr<float>(),nullptr,nullptr,visible.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

std::vector<torch::Tensor> deployment_selector(const torch::Tensor& physical,const torch::Tensor& rows,
    const torch::Tensor& owners,const torch::Tensor& view,const torch::Tensor& noise,
    const std::vector<torch::Tensor>& weights,double ratio,bool local,int64_t schema) {
  return deployment_selector_impl(physical,rows,owners,view,noise,weights,ratio,local,schema,false);
}

std::vector<torch::Tensor> deployment_selector_tiled(const torch::Tensor& physical,const torch::Tensor& rows,
    const torch::Tensor& owners,const torch::Tensor& view,const torch::Tensor& noise,
    const std::vector<torch::Tensor>& weights,double ratio,bool local,int64_t schema) {
  return deployment_selector_impl(physical,rows,owners,view,noise,weights,ratio,local,schema,true);
}
std::vector<torch::Tensor> deployment_compensate_impl(const torch::Tensor& offset,const torch::Tensor& rows,
    const torch::Tensor& owners,const torch::Tensor& reference,const std::vector<torch::Tensor>& tables,
    const std::vector<torch::Tensor>& weights,const std::vector<torch::Tensor>& attributes,
    const torch::Tensor& view,const torch::Tensor& view_weight,const std::vector<double>& residual_weights,
    const std::vector<double>& ratio,bool local,int64_t schema,bool softplus,const torch::Tensor& shared_status,int64_t tile) {
  const bool student=weights.size()==13;const int hidden=student?32:64;
  TORCH_CHECK(!torch::GradMode::is_enabled() && (student || weights.size()==21) && attributes.size()==6 &&
      tables.size()>=2 && tables.size()<=6 && residual_weights.size()==tables.size() && ratio.size()==3,"Frozen Compensator inputs");
  check(offset,offset);check(rows,offset,at::kLong);check(owners,offset,at::kLong);check(reference,offset);check(view,offset);
  c10::cuda::CUDAGuard guard(offset.device());Weights w{};Tables t{};Controls c{};Master m{};
  for(size_t j=0;j<weights.size();++j){check(weights[j],offset);w.p[j]=weights[j].data_ptr<float>();}
  t.count=tables.size();for(int j=0;j<t.count;++j){check(tables[j],offset);t.p[j]=tables[j].data_ptr<float>();c.weights[j]=residual_weights[j];}
  for(int j=0;j<3;++j)c.ratio[j]=ratio[j];for(int j=0;j<6;++j){check(attributes[j],offset);m.p[j]=attributes[j].data_ptr<float>();}
  m.rest=1;for(int d=1;d<attributes[5].dim();++d)m.rest*=attributes[5].size(d);
  TORCH_CHECK(offset.sizes()==at::IntArrayRef({owners.numel(),3}) && reference.numel()==owners.numel() &&
      weights[0].sizes()==at::IntArrayRef({51,hidden}) && weights[2].sizes()==at::IntArrayRef({hidden,hidden}) &&
      weights[1].numel()==hidden && weights[3].numel()==hidden &&
      weights[4].sizes()==at::IntArrayRef({3,hidden}) && attributes[0].sizes()==offset.sizes() &&
      attributes[1].numel()==owners.numel() && attributes[2].sizes()==offset.sizes() &&
      attributes[3].sizes()==at::IntArrayRef({owners.numel(),4}) && attributes[4].numel()==owners.numel()*3,
      "Compensation snapshot layout");
  for(int j=0;j<t.count;++j)TORCH_CHECK(tables[j].dim()==2 && tables[j].size(1)==48,"Residual48 table layout");
  for(int head=0;head<4;++head) {
    const int k=5+head*(student?2:4),width=head<2?3:4;
    if(student) {
      TORCH_CHECK(weights[k].sizes()==at::IntArrayRef({hidden,width}) && weights[k+1].numel()==width,"Student output Head layout");
    } else {
      TORCH_CHECK(weights[k].sizes()==at::IntArrayRef({64,64}) && weights[k+1].numel()==64 &&
          weights[k+2].sizes()==at::IntArrayRef({64,width}) && weights[k+3].numel()==width,"Independent Head layout");
    }
  }
  if(local){check(view_weight,offset);TORCH_CHECK(view_weight.sizes()==at::IntArrayRef({8,hidden}),"View projection layout");}
  const auto n=rows.numel();auto out=torch::empty({n*(17+m.rest)},offset.options()),delta=torch::empty({n,14},offset.options());
  auto status=shared_status.defined()?shared_status:torch::zeros({1},rows.options().dtype(at::kInt));
  check(status,offset,at::kInt);TORCH_CHECK(status.numel()==1,"Deployment finite status");
  TORCH_CHECK(tile==0 || tile==4 || tile==8 || tile==16 || tile==6416 || tile==8208,"Compensation tile width");
  if(n) {
    const auto stream=c10::cuda::getCurrentCUDAStream();
    if(student) {
      if(tile==16 || tile==6416)compensationStudent<16><<<(n+15)/16,128,0,stream>>>(n,offset.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),reference.data_ptr<float>(),t,c,w,view.data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local,schema==2||schema==4,m,softplus,out.data_ptr<float>(),delta.data_ptr<float>(),status.data_ptr<int>(),nullptr);
      else if(tile==8208 || tile==8)compensationStudent<8><<<(n+7)/8,128,0,stream>>>(n,offset.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),reference.data_ptr<float>(),t,c,w,view.data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local,schema==2||schema==4,m,softplus,out.data_ptr<float>(),delta.data_ptr<float>(),status.data_ptr<int>(),nullptr);
      else compensationStudent<4><<<(n+3)/4,128,0,stream>>>(n,offset.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),reference.data_ptr<float>(),t,c,w,view.data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local,schema==2||schema==4,m,softplus,out.data_ptr<float>(),delta.data_ptr<float>(),status.data_ptr<int>(),nullptr);
    } else if(tile==8208)compensationRegisterUnrolled<8,false,2><<<(n+7)/8,256,0,stream>>>(n,offset.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),reference.data_ptr<float>(),t,c,w,view.data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local,schema==2||schema==4,m,softplus,out.data_ptr<float>(),delta.data_ptr<float>(),status.data_ptr<int>(),nullptr);
    else if(tile==6416)compensationRegisterUnrolled<16><<<(n+15)/16,256,0,stream>>>(n,offset.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),reference.data_ptr<float>(),t,c,w,view.data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local,schema==2||schema==4,m,softplus,out.data_ptr<float>(),delta.data_ptr<float>(),status.data_ptr<int>(),nullptr);
    else if(tile==16)compensationRegisterTile<16><<<(n+15)/16,256,0,stream>>>(n,offset.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),reference.data_ptr<float>(),t,c,w,view.data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local,schema==2||schema==4,m,softplus,out.data_ptr<float>(),delta.data_ptr<float>(),status.data_ptr<int>(),nullptr);
    else if(tile==8)compensationTile<8><<<(n+7)/8,256,0,stream>>>(n,offset.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),reference.data_ptr<float>(),t,c,w,view.data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local,schema==2||schema==4,m,softplus,out.data_ptr<float>(),delta.data_ptr<float>(),status.data_ptr<int>(),nullptr);
    else if(tile==4)compensationTile<4><<<(n+3)/4,128,0,stream>>>(n,offset.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),reference.data_ptr<float>(),t,c,w,view.data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local,schema==2||schema==4,m,softplus,out.data_ptr<float>(),delta.data_ptr<float>(),status.data_ptr<int>(),nullptr);
    else compensationBlock<><<<n,128,0,stream>>>(n,offset.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),reference.data_ptr<float>(),t,c,w,view.data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local,schema==2||schema==4,m,softplus,out.data_ptr<float>(),delta.data_ptr<float>(),status.data_ptr<int>(),nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,delta,status};
}

std::vector<torch::Tensor> deployment_compensate_checked(const torch::Tensor& offset,const torch::Tensor& rows,
    const torch::Tensor& owners,const torch::Tensor& reference,const std::vector<torch::Tensor>& tables,
    const std::vector<torch::Tensor>& weights,const std::vector<torch::Tensor>& attributes,
    const torch::Tensor& view,const torch::Tensor& view_weight,const std::vector<double>& residual_weights,
    const std::vector<double>& ratio,bool local,int64_t schema,bool softplus,const torch::Tensor& status) {
  return deployment_compensate_impl(offset,rows,owners,reference,tables,weights,attributes,view,
      view_weight,residual_weights,ratio,local,schema,softplus,status,0);
}

std::vector<torch::Tensor> deployment_compensate_tiled(const torch::Tensor& offset,const torch::Tensor& rows,
    const torch::Tensor& owners,const torch::Tensor& reference,const std::vector<torch::Tensor>& tables,
    const std::vector<torch::Tensor>& weights,const std::vector<torch::Tensor>& attributes,
    const torch::Tensor& view,const torch::Tensor& view_weight,const std::vector<double>& residual_weights,
    const std::vector<double>& ratio,bool local,int64_t schema,bool softplus,int64_t tile) {
  return deployment_compensate_impl(offset,rows,owners,reference,tables,weights,attributes,view,
      view_weight,residual_weights,ratio,local,schema,softplus,torch::Tensor(),tile);
}

std::vector<torch::Tensor> deployment_compensate(const torch::Tensor& offset,const torch::Tensor& rows,
    const torch::Tensor& owners,const torch::Tensor& reference,const std::vector<torch::Tensor>& tables,
    const std::vector<torch::Tensor>& weights,const std::vector<torch::Tensor>& attributes,
    const torch::Tensor& view,const torch::Tensor& view_weight,const std::vector<double>& residual_weights,
    const std::vector<double>& ratio,bool local,int64_t schema,bool softplus) {
  return deployment_compensate_checked(offset,rows,owners,reference,tables,weights,attributes,view,
      view_weight,residual_weights,ratio,local,schema,softplus,torch::Tensor());
}

namespace {
struct CountPair {int64_t visible,eligible;};
struct AddPair {
  __device__ CountPair operator()(CountPair a,CountPair b) const {return {a.visible+b.visible,a.eligible+b.eligible};}
};
template<bool Reduce=false> __global__ void frameControls(int64_t* counts,DeploymentFrame* frame,unsigned* selection,
    DeploymentKnots knots,int64_t budget,double requested_ratio,bool use_ratio,
    const DeploymentRequest* request=nullptr,const int64_t* partial=nullptr,int blocks=0) {
  if(Reduce) {
    CountPair value{0,0};
    for(int j=threadIdx.x;j<blocks;j+=blockDim.x) {
      value.visible+=partial[j*2];value.eligible+=partial[j*2+1];
    }
    using Reduction=cub::BlockReduce<CountPair,128>;
    __shared__ typename Reduction::TempStorage temp;
    const auto sum=Reduction(temp).Reduce(value,AddPair());
    if(threadIdx.x)return;
    counts[1]=sum.visible;counts[2]=sum.eligible;
  }
  if(request){budget=request->budget;requested_ratio=request->ratio;use_ratio=request->use_ratio;}
  DeploymentFrame f{};
  f.n=counts[0];f.visible=counts[1];f.available=counts[2];
  f.budget=use_ratio?int64_t(floor(requested_ratio*double(f.n))):budget;
  f.k=f.budget<f.available?f.budget:f.available;
  f.has_ratio=f.n && (use_ratio?requested_ratio>0:budget>0);
  f.ratio=f.has_ratio?(use_ratio?requested_ratio:fmin(1.,double(budget)/double(f.n))):0.;
  for(int j=0;j<6;++j)f.weights[j]=-1.f;
  f.weights[0]=1.f;
  if(f.has_ratio) {
    f.log_ratio=logf(float(f.ratio));
    for(int j=1;j<knots.count;++j) {
      if(f.ratio<=knots.values[j-1]+1e-8)break;
      f.weights[j]=float(fmin(1.,(f.ratio-knots.values[j-1])/(knots.values[j]-knots.values[j-1])));
    }
    // Keep the original double control arithmetic and explicit intermediate rounding.
    const double numerator=__dsub_rn(log(f.ratio),knots.log_lo);
    const double t=__dsub_rn(__ddiv_rn(__dmul_rn(2.,numerator),knots.log_span),1.);
    const double angle=__ddiv_rn(__dmul_rn(CUDART_PI,t),2.);
    f.ratio_code[0]=float(t);f.ratio_code[1]=float(sin(angle));f.ratio_code[2]=float(cos(angle));
  }
  *frame=f;selection[0]=0;selection[1]=unsigned(f.k);
}

__global__ void uploadRequest(const float* vm,int64_t v0,int64_t v1,
    const float* pm,int64_t p0,int64_t p1,const float* center,int64_t center_stride,
    int64_t budget,double ratio,double aspect,double near,bool use_ratio,
    float* camera,DeploymentRequest* request,float* attributes,int64_t* selected) {
  const int i=threadIdx.x;
  if(i<16)camera[i]=vm[(i/4)*v0+(i%4)*v1];
  else if(i<32){const int j=i-16;camera[i]=pm[(j/4)*p0+(j%4)*p1];}
  else if(i<35)camera[i]=center[(i-32)*center_stride];
  if(!i) {
    request->budget=budget;
    request->ratio=ratio;
    request->aspect=aspect;
    request->near=near;
    request->use_ratio=use_ratio;
    request->attributes=attributes;
    request->selected=selected;
  }
}

__global__ void publishBaseDevice(const DeploymentFrame* frame,const int64_t* ids,Master base,float* out,
    const DeploymentRequest* request) {
  if(frame->status)return;
  const int64_t n=frame->k,width=17+base.rest,i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(request) {
    out=request->attributes;
    if(i<n)request->selected[i]=ids[i];
  }
  if(i>=n*width)return;
  int part;int64_t j,row,col;
  if(i<n*3){part=0;j=i;row=j/3;col=j%3;}
  else if(i<n*4){part=1;j=i-n*3;row=j;col=0;}
  else if(i<n*7){part=2;j=i-n*4;row=j/3;col=j%3;}
  else if(i<n*11){part=3;j=i-n*7;row=j/4;col=j%4;}
  else if(i<n*14){part=4;j=i-n*11;row=j/3;col=j%3;}
  else if(i<n*(14+base.rest)){part=5;j=i-n*14;row=j/base.rest;col=j%base.rest;}
  else{out[i]=0.f;return;}
  const int widths[]={3,1,3,4,3,base.rest};
  out[i]=base.p[part][ids[row]*widths[part]+col];
}
}

void deployment_frame_controls(const torch::Tensor& counts,const torch::Tensor& frame,
    const torch::Tensor& selection,const DeploymentKnots& knots,
    int64_t budget,double requested_ratio,bool use_ratio) {
  frameControls<><<<1,1,0,c10::cuda::getCurrentCUDAStream()>>>(counts.data_ptr<int64_t>(),
      reinterpret_cast<DeploymentFrame*>(frame.data_ptr<uint8_t>()),
      reinterpret_cast<unsigned*>(selection.data_ptr<int>()),knots,budget,requested_ratio,use_ratio);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void deployment_frame_controls_request(const torch::Tensor& counts,const torch::Tensor& frame,
    const torch::Tensor& selection,const DeploymentKnots& knots,const torch::Tensor& request) {
  frameControls<><<<1,1,0,c10::cuda::getCurrentCUDAStream()>>>(counts.data_ptr<int64_t>(),
      reinterpret_cast<DeploymentFrame*>(frame.data_ptr<uint8_t>()),
      reinterpret_cast<unsigned*>(selection.data_ptr<int>()),knots,0,0.,false,
      reinterpret_cast<const DeploymentRequest*>(request.data_ptr<uint8_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void deployment_frame_controls_fused(const torch::Tensor& counts,const torch::Tensor& frame,
    const torch::Tensor& selection,const DeploymentKnots& knots,const torch::Tensor& partial,
    const torch::Tensor& request,int64_t budget,double ratio,bool use_ratio) {
  frameControls<true><<<1,128,0,c10::cuda::getCurrentCUDAStream()>>>(counts.data_ptr<int64_t>(),
      reinterpret_cast<DeploymentFrame*>(frame.data_ptr<uint8_t>()),
      reinterpret_cast<unsigned*>(selection.data_ptr<int>()),knots,budget,ratio,use_ratio,
      request.defined()?reinterpret_cast<const DeploymentRequest*>(request.data_ptr<uint8_t>()):nullptr,
      partial.data_ptr<int64_t>(),partial.numel()/2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void deployment_upload_request(const torch::Tensor& vm,const torch::Tensor& pm,const torch::Tensor& center,
    int64_t budget,double ratio,double aspect,double near,bool use_ratio,
    const torch::Tensor& camera,const torch::Tensor& request,const torch::Tensor& attributes,const torch::Tensor& selected) {
  uploadRequest<<<1,64,0,c10::cuda::getCurrentCUDAStream()>>>(vm.data_ptr<float>(),vm.stride(0),vm.stride(1),
      pm.data_ptr<float>(),pm.stride(0),pm.stride(1),center.data_ptr<float>(),center.stride(0),
      budget,ratio,aspect,near,use_ratio,camera.data_ptr<float>(),
      reinterpret_cast<DeploymentRequest*>(request.data_ptr<uint8_t>()),attributes.data_ptr<float>(),selected.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void deployment_device_forward(const std::vector<torch::Tensor>& state,const std::vector<torch::Tensor>& selector,
    const std::vector<torch::Tensor>& compensation,const std::vector<torch::Tensor>& tables,
    const std::vector<torch::Tensor>& master,const std::vector<torch::Tensor>& base,
    const torch::Tensor& view_weight,const std::vector<torch::Tensor>& buffers,
    const torch::Tensor& vm,const torch::Tensor& pm,const torch::Tensor& camera,
    double aspect,double near,int64_t schema,bool softplus,bool local,bool compensate,int64_t tile,
    bool radix,const std::vector<torch::Tensor>& radix_workspace,bool selector_tiled,
    bool saturated_selection,
    const torch::Tensor& request,bool unroll_compensation,bool fused_selection,
    const torch::Tensor& prefix,const torch::Tensor& visible,
    const std::vector<torch::Tensor>& tensor_weights,int64_t selector_mode,int64_t compensation_mode,int64_t tensor_warps,int64_t tensor_comp_warps,bool small_finish) {
  // buffers: frame, candidate rows, view, scores, logits, selected IDs, attributes,
  // selection counts. Large maps retain the original C++ path.
  const auto n=state[3].numel(),capacity=buffers[5].numel(),anchors=state[0].size(0);
  if(!capacity)return;
  auto frame=reinterpret_cast<DeploymentFrame*>(buffers[0].data_ptr<uint8_t>());
  const auto destination=request.defined()?reinterpret_cast<const DeploymentRequest*>(request.data_ptr<uint8_t>()):nullptr;
  auto status=&frame->status;
  auto stream=c10::cuda::getCurrentCUDAStream();
  if(local && schema!=1 && anchors && compensation_mode!=9&&compensation_mode!=13)viewKernel<<<(anchors+127)/128,128,0,stream>>>(anchors,
      state[0].data_ptr<float>(),state[10].data_ptr<float>(),state[11].data_ptr<float>(),camera.data_ptr<float>(),
      vm.data_ptr<float>(),vm.stride(0),vm.stride(1),pm.data_ptr<float>(),pm.stride(0),pm.stride(1),
      state[12].data_ptr<float>(),aspect,near,buffers[2].data_ptr<float>(),frame,
      request.defined()?reinterpret_cast<const DeploymentRequest*>(request.data_ptr<uint8_t>()):nullptr,
      visible.defined()?visible.data_ptr<bool>():nullptr);
  Weights sw{},cw{};Tables t{};Controls c{};Master m{},b{};
  for(int j=0;j<6;++j){sw.p[j]=selector[j].data_ptr<float>();m.p[j]=master[j].data_ptr<float>();b.p[j]=base[j].data_ptr<float>();}
  m.rest=b.rest=base[5].numel()/std::max(int64_t(1),n);
  if(compensate && compensation_mode && tensor_weights.size()>=8 && compensation_mode!=9&&compensation_mode!=13){
    for(size_t j=0;j<compensation.size();++j)cw.p[j]=compensation[j].data_ptr<float>();
    preparationProjectAnchors<<<(anchors*32+127)/128,128,0,stream>>>(anchors,tensor_weights[6].data_ptr<float>(),
        tables.size()-1,c,cw,buffers[2].data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,
        local&&schema,schema==2||schema==4,tensor_weights[7].data_ptr<float>(),frame,
        visible.defined()?visible.data_ptr<bool>():nullptr);
  }
  if(compensation_mode==9||compensation_mode==13){
    TORCH_CHECK(local&&schema!=1&&tensor_weights.size()>=8,"Fused Student view requires local frozen projection");
    for(size_t j=0;j<compensation.size();++j)cw.p[j]=compensation[j].data_ptr<float>();
    studentViewProject<<<(anchors*32+127)/128,128,0,stream>>>(anchors,state[0].data_ptr<float>(),state[10].data_ptr<float>(),state[11].data_ptr<float>(),camera.data_ptr<float>(),vm.data_ptr<float>(),vm.stride(0),vm.stride(1),pm.data_ptr<float>(),pm.stride(0),pm.stride(1),state[12].data_ptr<float>(),aspect,near,buffers[2].data_ptr<float>(),frame,destination,visible.defined()?visible.data_ptr<bool>():nullptr,tensor_weights[6].data_ptr<float>(),tables.size()-1,cw,view_weight.data_ptr<float>(),schema==2||schema==4,tensor_weights[7].data_ptr<float>());
  }
  if(compensation_mode==13){
    TORCH_CHECK(selector_mode==2&&tensor_weights.size()==9,"Anchor projection requires existing FP16 Selector and workspace");
    if(selector[0].size(0)==19)selectorAnchorProjection<19><<<(anchors+63)/64,128,0,stream>>>(anchors,buffers[2].data_ptr<float>(),frame,local,schema==2||schema==3,tensor_weights[4].data_ptr(),tensor_weights[8].data_ptr<float>());
    else selectorAnchorProjection<11><<<(anchors+63)/64,128,0,stream>>>(anchors,buffers[2].data_ptr<float>(),frame,local,false,tensor_weights[4].data_ptr(),tensor_weights[8].data_ptr<float>());
  }
  const auto rows=buffers[1].data_ptr<int64_t>();
  if(selector_mode) {
    auto launch=[&](auto mode,auto warps){
      constexpr int Mode=decltype(mode)::value,Warps=decltype(warps)::value;
      if(compensation_mode==13){
        if(selector[0].size(0)==19)selectorProjected<Warps,19><<<(n+Warps*16-1)/(Warps*16),Warps*32,0,stream>>>(n,rows,state[3].data_ptr<int64_t>(),buffers[2].data_ptr<float>(),state[9].data_ptr<float>(),sw,0.f,local,schema==2||schema==3,buffers[3].data_ptr<float>(),buffers[4].data_ptr<float>(),status,frame,prefix.data_ptr<float>(),tensor_weights[0].data_ptr(),tensor_weights[8].data_ptr<float>());
        else if(selector[0].size(0)==11)selectorProjected<Warps,11><<<(n+Warps*16-1)/(Warps*16),Warps*32,0,stream>>>(n,rows,state[3].data_ptr<int64_t>(),buffers[2].data_ptr<float>(),state[9].data_ptr<float>(),sw,0.f,local,schema==2||schema==3,buffers[3].data_ptr<float>(),buffers[4].data_ptr<float>(),status,frame,prefix.data_ptr<float>(),tensor_weights[0].data_ptr(),tensor_weights[8].data_ptr<float>());
      }else{
      if(selector[0].size(0)==19)selectorTensor<Mode,Warps,19><<<(n+Warps*16-1)/(Warps*16),(Mode==2?Warps*32:128),0,stream>>>(n,state[6].data_ptr<float>(),rows,state[3].data_ptr<int64_t>(),buffers[2].data_ptr<float>(),state[9].data_ptr<float>(),sw,0.f,local,schema==2||schema==3,buffers[3].data_ptr<float>(),buffers[4].data_ptr<float>(),status,frame,prefix.defined()?prefix.data_ptr<float>():nullptr,tensor_weights[0].data_ptr(),tensor_weights[4].data_ptr());
      else selectorTensor<Mode,Warps,11><<<(n+Warps*16-1)/(Warps*16),(Mode==2?Warps*32:128),0,stream>>>(n,state[6].data_ptr<float>(),rows,state[3].data_ptr<int64_t>(),buffers[2].data_ptr<float>(),state[9].data_ptr<float>(),sw,0.f,local,false,buffers[3].data_ptr<float>(),buffers[4].data_ptr<float>(),status,frame,prefix.defined()?prefix.data_ptr<float>():nullptr,tensor_weights[0].data_ptr(),tensor_weights[4].data_ptr());
      }
    };
    dispatchTensor(selector_mode,tensor_warps,launch);
  } else if(prefix.defined()) {
  if(selector_tiled) {
    if(selector[0].size(0)==19)selectorRegisterTile<19,true><<<(n+15)/16,128,0,stream>>>(n,state[6].data_ptr<float>(),rows,
        state[3].data_ptr<int64_t>(),buffers[2].data_ptr<float>(),state[9].data_ptr<float>(),sw,0.f,local,
        schema==2||schema==3,buffers[3].data_ptr<float>(),buffers[4].data_ptr<float>(),status,frame,prefix.data_ptr<float>());
    else selectorRegisterTile<11,true><<<(n+15)/16,128,0,stream>>>(n,state[6].data_ptr<float>(),rows,
        state[3].data_ptr<int64_t>(),buffers[2].data_ptr<float>(),state[9].data_ptr<float>(),sw,0.f,local,
        false,buffers[3].data_ptr<float>(),buffers[4].data_ptr<float>(),status,frame,prefix.data_ptr<float>());
  } else if(selector[0].size(0)==19)selectorKernel<19,true><<<(n+3)/4,128,0,stream>>>(n,state[6].data_ptr<float>(),rows,
      state[3].data_ptr<int64_t>(),buffers[2].data_ptr<float>(),state[9].data_ptr<float>(),sw,0.f,local,
      schema==2||schema==3,buffers[3].data_ptr<float>(),buffers[4].data_ptr<float>(),status,frame,prefix.data_ptr<float>());
  else selectorKernel<11,true><<<(n+3)/4,128,0,stream>>>(n,state[6].data_ptr<float>(),rows,state[3].data_ptr<int64_t>(),
      buffers[2].data_ptr<float>(),state[9].data_ptr<float>(),sw,0.f,local,false,
      buffers[3].data_ptr<float>(),buffers[4].data_ptr<float>(),status,frame,prefix.data_ptr<float>());
  } else {
  if(selector_tiled) {
    if(selector[0].size(0)==19)selectorRegisterTile<19><<<(n+15)/16,128,0,stream>>>(n,state[6].data_ptr<float>(),rows,
        state[3].data_ptr<int64_t>(),buffers[2].data_ptr<float>(),state[9].data_ptr<float>(),sw,0.f,local,
        schema==2||schema==3,buffers[3].data_ptr<float>(),buffers[4].data_ptr<float>(),status,frame);
    else selectorRegisterTile<11><<<(n+15)/16,128,0,stream>>>(n,state[6].data_ptr<float>(),rows,
        state[3].data_ptr<int64_t>(),buffers[2].data_ptr<float>(),state[9].data_ptr<float>(),sw,0.f,local,
        false,buffers[3].data_ptr<float>(),buffers[4].data_ptr<float>(),status,frame);
  } else if(selector[0].size(0)==19)selectorKernel<19><<<(n+3)/4,128,0,stream>>>(n,state[6].data_ptr<float>(),rows,
      state[3].data_ptr<int64_t>(),buffers[2].data_ptr<float>(),state[9].data_ptr<float>(),sw,0.f,local,
      schema==2||schema==3,buffers[3].data_ptr<float>(),buffers[4].data_ptr<float>(),status,frame);
  else selectorKernel<11><<<(n+3)/4,128,0,stream>>>(n,state[6].data_ptr<float>(),rows,state[3].data_ptr<int64_t>(),
      buffers[2].data_ptr<float>(),state[9].data_ptr<float>(),sw,0.f,local,false,
      buffers[3].data_ptr<float>(),buffers[4].data_ptr<float>(),status,frame);
  }
  if(radix)deployment_radix_device(buffers[3],buffers[1],state[5],buffers[5],buffers[0],radix_workspace,saturated_selection,fused_selection && n>=16384,small_finish);
  else deployment_select_device(buffers[3],buffers[1],state[5],buffers[5],buffers[0],buffers[7]);
  if(compensate) {
    for(size_t j=0;j<compensation.size();++j)cw.p[j]=compensation[j].data_ptr<float>();
    t.count=tables.size();for(int j=0;j<t.count;++j)t.p[j]=tables[j].data_ptr<float>();
    auto launch_compensation=[&](auto direct) {
      constexpr bool Direct=decltype(direct)::value;
      if(compensation_mode) {
        TORCH_CHECK(compensation.size()==13,"Tensor Compensation requires the frozen 32/32 student");
        TensorWeights tw{};for(int i=1;i<4;++i)tw.p[i]=tensor_weights[i].data_ptr();
        if(tensor_weights.size()>=8){tw.p[5]=tensor_weights[5].data_ptr();tw.p[7]=tensor_weights[7].data_ptr();}
        auto launch=[&](auto mode,auto warps){constexpr int Mode=decltype(mode)::value,Warps=decltype(warps)::value;
          auto kernel=[&](auto projected){constexpr bool Projected=decltype(projected)::value;
            if(compensation_mode>=3&&compensation_mode<=6){
              const bool persistent=compensation_mode==4||compensation_mode==6;
              const int factor=compensation_mode==5?2:compensation_mode==6?4:1;
              const int64_t blocks=(capacity+Warps*16-1)/(Warps*16);
              const int64_t grid=persistent?std::min<int64_t>(blocks,4*at::cuda::getCurrentDeviceProperties()->multiProcessorCount):blocks;
              compensationStudentScheduled<Mode,Warps,Direct,Projected><<<grid,Warps*32*factor,0,stream>>>(capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),t,c,cw,buffers[2].data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,m,softplus,buffers[6].data_ptr<float>(),nullptr,status,frame,destination,tw);
            }else{
            compensationTensor<Mode,Warps,Direct,Projected><<<(capacity+Warps*16-1)/(Warps*16),Warps*128,0,stream>>>(capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),t,c,cw,buffers[2].data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,m,softplus,buffers[6].data_ptr<float>(),nullptr,status,frame,destination,tw);
            }};
          if(tensor_weights.size()>=8)kernel(std::true_type{});else kernel(std::false_type{});
        };dispatchTensor(compensation_mode>=3?2:compensation_mode,tensor_comp_warps,launch);
      } else if(compensation.size()==13) {
        if(tile==16)compensationStudent<16,Direct><<<(capacity+15)/16,128,0,stream>>>(capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),t,c,cw,buffers[2].data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,m,softplus,buffers[6].data_ptr<float>(),nullptr,status,frame,destination);
        else if(tile==8208 || tile==8)compensationStudent<8,Direct><<<(capacity+7)/8,128,0,stream>>>(capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),t,c,cw,buffers[2].data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,m,softplus,buffers[6].data_ptr<float>(),nullptr,status,frame,destination);
        else compensationStudent<4,Direct><<<(capacity+3)/4,128,0,stream>>>(capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),t,c,cw,buffers[2].data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,m,softplus,buffers[6].data_ptr<float>(),nullptr,status,frame,destination);
      } else if(tile==8208)compensationRegisterUnrolled<8,Direct,2><<<(capacity+7)/8,256,0,stream>>>(capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),
          state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),t,c,cw,buffers[2].data_ptr<float>(),
          local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,m,softplus,
          buffers[6].data_ptr<float>(),nullptr,status,frame,destination);
      else if(tile==16 && unroll_compensation)compensationRegisterUnrolled<16,Direct><<<(capacity+15)/16,256,0,stream>>>(capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),
          state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),t,c,cw,buffers[2].data_ptr<float>(),
          local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,m,softplus,
          buffers[6].data_ptr<float>(),nullptr,status,frame,destination);
      else if(tile==16)compensationRegisterTile<16,Direct><<<(capacity+15)/16,256,0,stream>>>(capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),
          state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),t,c,cw,buffers[2].data_ptr<float>(),
          local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,m,softplus,
          buffers[6].data_ptr<float>(),nullptr,status,frame,destination);
      else if(tile==4)compensationTile<4,Direct><<<(capacity+3)/4,128,0,stream>>>(capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),
          state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),t,c,cw,buffers[2].data_ptr<float>(),
          local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,m,softplus,
          buffers[6].data_ptr<float>(),nullptr,status,frame,destination);
      else if(tile==8)compensationTile<8,Direct><<<(capacity+7)/8,256,0,stream>>>(capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),
          state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),t,c,cw,buffers[2].data_ptr<float>(),
          local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,m,softplus,
          buffers[6].data_ptr<float>(),nullptr,status,frame,destination);
      else compensationBlock<Direct><<<capacity,128,0,stream>>>(capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),
          state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),t,c,cw,buffers[2].data_ptr<float>(),
          local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,m,softplus,
          buffers[6].data_ptr<float>(),nullptr,status,frame,destination);
    };
    if(destination)launch_compensation(std::true_type{});
    else launch_compensation(std::false_type{});
  } else publishBaseDevice<<<(capacity*(17+b.rest)+255)/256,256,0,stream>>>(frame,buffers[5].data_ptr<int64_t>(),b,buffers[6].data_ptr<float>(),destination);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

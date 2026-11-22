#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <math_constants.h>

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
    const float* pm,int64_t p0,int64_t p1,const float* scale,float aspect,float near,float* out) {
  const auto a=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(a>=n)return;
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

template<int Width> __global__ void selectorKernel(int64_t n,const float* physical,
    const int64_t* rows,const int64_t* owners,const float* view,const float* noise,
    Weights w,float log_ratio,bool local,bool real_view,float* scores,float* logits,int* status) {
  const int warp=threadIdx.x/32,lane=threadIdx.x%32;
  const int64_t row=int64_t(blockIdx.x)*4+warp;
  if(row>=n)return;
  const auto id=rows[row];
  __shared__ float h0[4][32],h1[4][32];
  float x=0.f;
  if(lane<10)x=physical[id*10+lane];
  else if(lane==10)x=log_ratio;
  else if(Width==19 && lane<18)x=local&&real_view?view[owners[id]*7+lane-11]:0.f;
  else if(Width==19 && lane==18)x=local?1.f:0.f;
  float sum=0.f;
  #pragma unroll
  for(int j=0;j<Width;++j)sum=fmaf(__shfl_sync(0xffffffff,x,j),w.p[0][j*32+lane],sum);
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
__device__ void normalize(float* q) {
  float n=0.f;for(int j=0;j<4;++j)n=fmaf(q[j],q[j],n);
  n=fmaxf(sqrtf(n),1e-12f);for(int j=0;j<4;++j)q[j]/=n;
}
__device__ void publishPoint(int64_t row,int64_t n,int64_t id,const float* d,
    Master master,bool softplus,float* output,int* status,int lane,int stride) {
  float q[4];
  for(int j=0;j<4;++j)q[j]=d[6+j]+(j==0?1.f:0.f);
  normalize(q);
  const float* b=master.p[3]+id*4;
  float rotation[4];
  rotation[0]=q[0]*b[0]-((q[1]*b[1]+q[3]*b[3])+q[2]*b[2]);
  for(int j=1;j<4;++j){int k=j%3+1,l=(j+1)%3+1;rotation[j]=(q[0]*b[j]+b[0]*q[j])+fmaf(q[k],b[l],-q[l]*b[k]);}
  normalize(rotation);normalize(rotation);
  const int width=17+master.rest;
  for(int col=lane;col<width;col+=stride) {
    float value;int64_t dest;
    if(col<3){value=master.p[0][id*3+col]+d[col];dest=row*3+col;}
    else if(col==3){value=1.f/(1.f+expf(-(master.p[1][id]+d[10])));dest=n*3+row;}
    else if(col<7){int j=col-4;float factor=d[3+j];if(softplus){float z=factor+.5413248546129181f;factor=z>20.f?z:log1pf(expf(z));value=fmaxf(master.p[2][id*3+j]*factor,1.1754943508222875e-38f);}else value=master.p[2][id*3+j]+factor;value=expf(logf(value));dest=n*4+row*3+j;}
    else if(col<11){value=rotation[col-7];dest=n*7+row*4+col-7;}
    else if(col<14){value=master.p[4][id*3+col-11]+d[11+col-11];dest=n*11+row*3+col-11;}
    else if(col<14+master.rest){value=master.p[5][id*master.rest+col-14];dest=n*14+row*master.rest+col-14;}
    else{value=0.f;dest=n*(14+master.rest)+row*3+col-14-master.rest;}
    output[dest]=value;if(!isfinite(value))atomicExch(status,2);
  }
}

// One block per Gaussian: trunk is shared; the four warps run four independent Heads.
__global__ void compensationBlock(int64_t n,const float* offset,const int64_t* rows,
    const int64_t* owners,const float* reference,Tables tables,Controls controls,
    Weights w,const float* view,const float* view_weight,bool local,bool real_view,
    Master master,bool softplus,float* output,float* residual,int* status) {
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
  if(lane<head_width){float sum=0.f;for(int j=0;j<64;++j)sum=fmaf(h2[head][j],w.p[p+2][j*head_width+lane],sum);sum+=w.p[p+3][lane];if(head==0)sum*=reference[id];delta[head_offset+lane]=sum;residual[row*14+head_offset+lane]=sum;if(!isfinite(sum))atomicExch(status,2);}
  __syncthreads();
  publishPoint(row,n,id,delta,master,softplus,output,status,tid,128);
}
}

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
  if(n)viewKernel<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(n,centers.data_ptr<float>(),centers4.data_ptr<float>(),corners.data_ptr<float>(),camera.data_ptr<float>(),vm.data_ptr<float>(),vm.stride(0),vm.stride(1),pm.data_ptr<float>(),pm.stride(0),pm.stride(1),scale.data_ptr<float>(),aspect,near,out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<torch::Tensor> deployment_selector(const torch::Tensor& physical,const torch::Tensor& rows,
    const torch::Tensor& owners,const torch::Tensor& view,const torch::Tensor& noise,
    const std::vector<torch::Tensor>& weights,double ratio,bool local,int64_t schema) {
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
  if(n){if(width==19)selectorKernel<19><<<(n+3)/4,128,0,stream>>>(n,physical.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),view.data_ptr<float>(),noise.data_ptr<float>(),w,lr,local,schema==2||schema==3,scores.data_ptr<float>(),logits.data_ptr<float>(),status.data_ptr<int>());
    else selectorKernel<11><<<(n+3)/4,128,0,stream>>>(n,physical.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),view.data_ptr<float>(),noise.data_ptr<float>(),w,lr,local,false,scores.data_ptr<float>(),logits.data_ptr<float>(),status.data_ptr<int>());}
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {scores,logits,status};
}
std::vector<torch::Tensor> deployment_compensate(const torch::Tensor& offset,const torch::Tensor& rows,
    const torch::Tensor& owners,const torch::Tensor& reference,const std::vector<torch::Tensor>& tables,
    const std::vector<torch::Tensor>& weights,const std::vector<torch::Tensor>& attributes,
    const torch::Tensor& view,const torch::Tensor& view_weight,const std::vector<double>& residual_weights,
    const std::vector<double>& ratio,bool local,int64_t schema,bool softplus) {
  TORCH_CHECK(!torch::GradMode::is_enabled() && weights.size()==21 && attributes.size()==6 &&
      tables.size()>=2 && tables.size()<=6 && residual_weights.size()==tables.size() && ratio.size()==3,"Frozen Compensator inputs");
  check(offset,offset);check(rows,offset,at::kLong);check(owners,offset,at::kLong);check(reference,offset);check(view,offset);
  c10::cuda::CUDAGuard guard(offset.device());Weights w{};Tables t{};Controls c{};Master m{};
  for(int j=0;j<21;++j){check(weights[j],offset);w.p[j]=weights[j].data_ptr<float>();}
  t.count=tables.size();for(int j=0;j<t.count;++j){check(tables[j],offset);t.p[j]=tables[j].data_ptr<float>();c.weights[j]=residual_weights[j];}
  for(int j=0;j<3;++j)c.ratio[j]=ratio[j];for(int j=0;j<6;++j){check(attributes[j],offset);m.p[j]=attributes[j].data_ptr<float>();}
  m.rest=1;for(int d=1;d<attributes[5].dim();++d)m.rest*=attributes[5].size(d);
  TORCH_CHECK(offset.sizes()==at::IntArrayRef({owners.numel(),3}) && reference.numel()==owners.numel() &&
      weights[0].sizes()==at::IntArrayRef({51,64}) && weights[2].sizes()==at::IntArrayRef({64,64}) &&
      weights[4].sizes()==at::IntArrayRef({3,64}) && attributes[0].sizes()==offset.sizes() &&
      attributes[1].numel()==owners.numel() && attributes[2].sizes()==offset.sizes() &&
      attributes[3].sizes()==at::IntArrayRef({owners.numel(),4}) && attributes[4].numel()==owners.numel()*3,
      "Compensation snapshot layout");
  for(int j=0;j<t.count;++j)TORCH_CHECK(tables[j].dim()==2 && tables[j].size(1)==48,"Residual48 table layout");
  for(int head=0;head<4;++head) {
    int k=5+head*4,width=head<2?3:4;
    TORCH_CHECK(weights[k].sizes()==at::IntArrayRef({64,64}) && weights[k+1].numel()==64 &&
        weights[k+2].sizes()==at::IntArrayRef({64,width}) && weights[k+3].numel()==width,"Independent Head layout");
  }
  if(local){check(view_weight,offset);TORCH_CHECK(view_weight.sizes()==at::IntArrayRef({8,64}),"View projection layout");}
  const auto n=rows.numel();auto out=torch::empty({n*(17+m.rest)},offset.options()),delta=torch::empty({n,14},offset.options()),status=torch::zeros({1},rows.options().dtype(at::kInt));
  if(n)compensationBlock<<<n,128,0,c10::cuda::getCurrentCUDAStream()>>>(n,offset.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),reference.data_ptr<float>(),t,c,w,view.data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local,schema==2||schema==4,m,softplus,out.data_ptr<float>(),delta.data_ptr<float>(),status.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,delta,status};
}

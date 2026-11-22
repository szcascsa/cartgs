#include "include/spatial_compensation_cuda.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <math_constants.h>
namespace mapping_cuda {
namespace {
__device__ void rotation(const float* q,float* R){
  float w=q[0],x=q[1],y=q[2],z=q[3];
  R[0]=1-2*(y*y+z*z);R[1]=2*(x*y-w*z);R[2]=2*(x*z+w*y);
  R[3]=2*(x*y+w*z);R[4]=1-2*(x*x+z*z);R[5]=2*(y*z-w*x);
  R[6]=2*(x*z-w*y);R[7]=2*(y*z+w*x);R[8]=1-2*(x*x+y*y);
}
__global__ void liveForward(int64_t n,const float* xyz,const float* scale,const float* q,const float* dc,
    const float* logits,const float* center,const float* reference,const float* wc,const float* ws,
    const float* mean,const float* stddev,float* out,float* saved){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  float R[9],A[9],C[9],raw[19];rotation(q+4*i,R);
  for(int row=0;row<3;++row)for(int col=0;col<3;++col)A[row*3+col]=R[row*3+col]*(scale[i*3+col]*scale[i*3+col]);
  for(int row=0;row<3;++row)for(int col=0;col<3;++col){float v=0;for(int j=0;j<3;++j)v+=A[row*3+j]*R[col*3+j];C[row*3+col]=v;}
  float trace=(C[0]+C[4])+C[8],denom=fmaxf(trace,1e-12f);
  for(int j=0;j<3;++j){raw[j]=(xyz[i*3+j]-wc[j])/ws[j];raw[3+j]=asinhf((xyz[i*3+j]-center[i*3+j])/reference[i]);raw[6+j]=logf(scale[i*3+j]);raw[9+j]=dc[i*3+j];}
  raw[12]=logits[i];int index[6]={0,4,8,1,2,5};for(int j=0;j<6;++j)raw[13+j]=C[index[j]]/denom;
  for(int j=0;j<19;++j){float value=(raw[j]-mean[j])/stddev[j];out[i*55+j]=fminf(15.f,fmaxf(-15.f,value));saved[i*38+j]=raw[j];}
  for(int j=0;j<18;++j){float frequency=ldexpf(1.f,j%6)*CUDART_PI_F,angle=raw[j/6]*frequency;out[i*55+19+j]=sinf(angle);out[i*55+37+j]=cosf(angle);}
  for(int j=0;j<9;++j){saved[i*38+19+j]=R[j];saved[i*38+28+j]=C[j];}saved[i*38+37]=trace;
}
__global__ void liveBackward(int64_t n,const float* xyz,const float* scale,const float* q,const float* center,
    const float* reference,const float* ws,const float* mean,const float* stddev,const float* saved,const float* g,
    float* gx,float* gs,float* gq,float* gd,float* gl,float* gc,float* ga){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  float d[19],R[9],G[9]={0},A[9],dA[9],dR[9];const float* raw=saved+i*38;
  for(int j=0;j<19;++j){float v=(raw[j]-mean[j])/stddev[j];d[j]=(v>=-15.f&&v<=15.f)?g[i*55+j]/stddev[j]:0.f;}
  float dr=0;
  for(int j=0;j<3;++j){float df=0;for(int k=0;k<6;++k){float frequency=ldexpf(1.f,k)*CUDART_PI_F,angle=raw[j]*frequency;df+=(g[i*55+19+j*6+k]*cosf(angle)+g[i*55+37+j*6+k]*(-sinf(angle)))*frequency;}
    float offset=(xyz[i*3+j]-center[i*3+j])/reference[i];float off=d[3+j]/sqrtf(1.f+offset*offset);
    gx[i*3+j]=d[j]/ws[j]+df/ws[j]+off/reference[i];gc[i*3+j]=-off/reference[i];dr-=off*offset/reference[i];
    gs[i*3+j]=d[6+j]/scale[i*3+j];gd[i*3+j]=d[9+j];}
  ga[i]=dr;gl[i]=d[12];
  for(int j=0;j<9;++j)R[j]=raw[19+j];
  float trace=raw[37],denom=fmaxf(trace,1e-12f),dt=0;int index[6]={0,4,8,1,2,5};
  for(int j=0;j<6;++j){G[index[j]]=d[13+j]/denom;dt-=d[13+j]*raw[13+j]/denom;}
  if(trace>=1e-12f){G[0]+=dt;G[4]+=dt;G[8]+=dt;}
  for(int row=0;row<3;++row)for(int col=0;col<3;++col)A[row*3+col]=R[row*3+col]*(scale[i*3+col]*scale[i*3+col]);
  for(int row=0;row<3;++row)for(int col=0;col<3;++col){float da=0,db=0;for(int k=0;k<3;++k){da+=G[row*3+k]*R[k*3+col];db+=G[k*3+row]*A[k*3+col];}dA[row*3+col]=da;dR[row*3+col]=da*(scale[i*3+col]*scale[i*3+col])+db;}
  for(int j=0;j<3;++j){float ss=0;for(int row=0;row<3;++row)ss+=dA[row*3+j]*R[row*3+j];gs[i*3+j]+=2*scale[i*3+j]*ss;}
  float w=q[i*4],x=q[i*4+1],y=q[i*4+2],z=q[i*4+3];
  gq[i*4]=2*(-z*dR[1]+y*dR[2]+z*dR[3]-x*dR[5]-y*dR[6]+x*dR[7]);
  gq[i*4+1]=2*(y*dR[1]+z*dR[2]+y*dR[3]-2*x*dR[4]-w*dR[5]+z*dR[6]+w*dR[7]-2*x*dR[8]);
  gq[i*4+2]=2*(-2*y*dR[0]+x*dR[1]+w*dR[2]+x*dR[3]+z*dR[5]-w*dR[6]+z*dR[7]-2*y*dR[8]);
  gq[i*4+3]=2*(-2*z*dR[0]-w*dR[1]+x*dR[2]+w*dR[3]-2*z*dR[4]+y*dR[5]+x*dR[6]+y*dR[7]);
}
const float* p(const torch::Tensor& t){return t.data_ptr<float>();}
void check(const std::vector<torch::Tensor>& a){
  TORCH_CHECK(a.size()>=11,"Spatial live input count");for(int j=0;j<11;++j)TORCH_CHECK(a[j].is_cuda()&&a[j].is_contiguous()&&a[j].scalar_type()==torch::kFloat32&&a[j].device()==a[0].device(),"Spatial live requires contiguous same-device FP32");
  int64_t n=a[0].size(0);const std::vector<int64_t> sizes{3,3,4,3,1,3,1};for(int j=0;j<7;++j)TORCH_CHECK(a[j].numel()==n*sizes[j],"Spatial live attribute shape mismatch");
}
}
std::vector<torch::Tensor> spatialLiveForward(const std::vector<torch::Tensor>& a){
  check(a);c10::cuda::CUDAGuard guard(a[0].device());auto n=a[0].size(0);auto out=torch::empty({n,55},a[0].options()),saved=torch::empty({n,38},a[0].options());
  if(n){liveForward<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(n,p(a[0]),p(a[1]),p(a[2]),p(a[3]),p(a[4]),p(a[5]),p(a[6]),p(a[7]),p(a[8]),p(a[9]),p(a[10]),out.data_ptr<float>(),saved.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}return {out,saved};
}
std::vector<torch::Tensor> spatialLiveBackward(const std::vector<torch::Tensor>& a,const torch::Tensor& saved,const torch::Tensor& gradient){
  check(a);c10::cuda::CUDAGuard guard(a[0].device());auto n=a[0].size(0);std::vector<torch::Tensor> g;for(int j=0;j<7;++j)g.push_back(torch::empty_like(a[j]));
  if(n){liveBackward<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(n,p(a[0]),p(a[1]),p(a[2]),p(a[5]),p(a[6]),p(a[8]),p(a[9]),p(a[10]),p(saved),p(gradient),g[0].data_ptr<float>(),g[1].data_ptr<float>(),g[2].data_ptr<float>(),g[3].data_ptr<float>(),g[4].data_ptr<float>(),g[5].data_ptr<float>(),g[6].data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();}return g;
}
}

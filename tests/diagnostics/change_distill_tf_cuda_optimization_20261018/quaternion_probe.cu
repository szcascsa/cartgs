#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include "include/quaternion_utils.h"
#include <iostream>
__global__ void probe(const float* q,const float* m,float* out,int n) {
  const int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=n*36)return;
  const int p=t%n,mode=t/n,dot=mode/12,norm=(mode/3)%4,cross=mode%3;
  float a=m[0],b=m[1],c=m[2],d=m[3],w=q[4*p],u=q[4*p+1],v=q[4*p+2],z=q[4*p+3];
  float s=dot==0 ? (b*u+c*v)+d*z : dot==1 ? (b*u+d*z)+c*v : b*u+(c*v+d*z);
  float r1=cross==0 ? c*z-d*v : cross==1 ? __fmaf_rn(c,z,-(d*v)) : __fmaf_rn(-d,v,c*z);
  float r2=cross==0 ? d*u-b*z : cross==1 ? __fmaf_rn(d,u,-(b*z)) : __fmaf_rn(-b,z,d*u);
  float r3=cross==0 ? b*v-c*u : cross==1 ? __fmaf_rn(b,v,-(c*u)) : __fmaf_rn(-c,u,b*v);
  float o0=a*w-s,o1=(a*u+w*b)+r1,o2=(a*v+w*c)+r2,o3=(a*z+w*d)+r3;
  float x=o0*o0,y=o1*o1,h=o2*o2,k=o3*o3;
  float sum=norm==0 ? ((x+y)+h)+k : norm==1 ? (x+y)+(h+k) : norm==2 ? (x+h)+(y+k) : (x+k)+(y+h);
  float len=fmaxf(sqrtf(sum),1e-12f);
  out[t*4]=o0/len;out[t*4+1]=o1/len;out[t*4+2]=o2/len;out[t*4+3]=o3/len;
}
int main() {
  torch::manual_seed(13);torch::NoGradGuard no_grad;torch::set_num_threads(4);
  auto options=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  const int n=100000;auto q=torch::randn({n,4},options),m=torch::randn({1,4},options);
  auto out=torch::empty({36,n,4},options);
  probe<<<(n*36+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(q.data_ptr<float>(),m.data_ptr<float>(),out.data_ptr<float>(),n);
  auto ref=quaternion_utils::normalize(quaternion_utils::multiply(m.expand({n,4}),q));
  for(int i=0;i<36;++i) std::cout<<i<<","<<torch::equal(ref,out[i])<<","<<(ref-out[i]).abs().max().item<float>()<<","<<(ref!=out[i]).sum().item<int64_t>()<<"\n";
}

#include "include/transform_field_cuda.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>

namespace tf_cuda {
namespace {
void checkTensor(const torch::Tensor& t,const torch::Tensor& ref) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type()==torch::kFloat32 && t.device()==ref.device(),
              "Neural TF CUDA requires same-device CUDA FP32");
}
void checkMatrix(const torch::Tensor& t,const torch::Tensor& ref,int rows,int cols) {
  checkTensor(t,ref);
  TORCH_CHECK(t.dim()==2 && t.size(0)==rows && t.size(1)==cols && t.is_contiguous(),
              "Invalid neural TF matrix shape/layout");
}
// LibTorch's handle belongs to the current device and stream. Restore modes so
// the dedicated FP32 backend does not alter surrounding renderer/LibTorch work.
struct Blas {
  cublasHandle_t h;
  cublasMath_t math;
  cublasPointerMode_t pointer;
  Blas():h(at::cuda::getCurrentCUDABlasHandle()) {
    TORCH_CUDABLAS_CHECK(cublasGetMathMode(h,&math));
    TORCH_CUDABLAS_CHECK(cublasGetPointerMode(h,&pointer));
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(h,CUBLAS_PEDANTIC_MATH));
    TORCH_CUDABLAS_CHECK(cublasSetPointerMode(h,CUBLAS_POINTER_MODE_HOST));
  }
  ~Blas() { cublasSetMathMode(h,math);cublasSetPointerMode(h,pointer); }
};
__global__ void biasKernel(float* y,const float* bias,int count,int width,bool relu) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) {float z=y[i]+bias[i%width];y[i]=relu?(z<=0?0:z):z;}
}
void linear(Blas& blas,const torch::Tensor& x,const torch::Tensor& w,
            const torch::Tensor& b,torch::Tensor& y,bool relu) {
  const int n=x.size(0),in=x.size(1),out=w.size(0);
  if(!n)return;
  const float one=1,zero=0;
  // Row-major Y=X W^T is column-major Y^T=W X^T.
  TORCH_CUDABLAS_CHECK(cublasSgemm(blas.h,CUBLAS_OP_T,CUBLAS_OP_N,out,n,in,
      &one,w.data_ptr<float>(),in,x.data_ptr<float>(),in,&zero,y.data_ptr<float>(),out));
  biasKernel<<<(n*out+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      y.data_ptr<float>(),b.data_ptr<float>(),n*out,out,relu);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
struct Neural {
  const float* p[14];
  float *shared,*head[3],*residual[3];
};
__global__ void warpMlpKernel(const float* feature,Neural a,int n) {
  const int lane=threadIdx.x%32,row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  if(row>=n)return;
  float h=0;
  for(int k=0;k<16;++k)h+=feature[row*16+k]*a.p[0][lane*16+k];
  h+=a.p[1][lane];h=h<=0?0:h;a.shared[row*32+lane]=h;
  for(int branch=0;branch<3;++branch) {
    const int layer=2+branch*4,dim=branch==2?4:3;
    float v=0;
    for(int k=0;k<32;++k)v+=__shfl_sync(0xffffffff,h,k)*a.p[layer][lane*32+k];
    v+=a.p[layer+1][lane];v=v<=0?0:v;a.head[branch][row*32+lane]=v;
    float z=0;
    // All lanes participate in shuffles, including inactive output neurons.
    for(int k=0;k<32;++k) {
      float hidden=__shfl_sync(0xffffffff,v,k);
      if(lane<dim)z+=hidden*a.p[layer+2][lane*32+k];
    }
    if(lane<dim)a.residual[branch][row*dim+lane]=z+a.p[layer+3][lane];
  }
}
struct Attr {
  const float *x,*s,*q,*dx,*ds,*dq;
  int64_t x0,x1,s0,s1,q0,q1;
};
__device__ float norm4(const float* q) {
  // ATen's block-x reduction shuffles by 1, then 2 for four elements.
  return sqrtf((q[0]*q[0]+q[1]*q[1])+(q[2]*q[2]+q[3]*q[3]));
}
__device__ void normalize4(float* q) {
  float norm=norm4(q),denom=norm<1e-12f?1e-12f:norm;
  for(int k=0;k<4;++k)q[k]/=denom;
}
__device__ void compose(const float* a,const float* b,float* z) {
  float dot=(a[1]*b[1]+a[3]*b[3])+a[2]*b[2];
  z[0]=a[0]*b[0]-dot;
  for(int k=1;k<4;++k) {
    int i=k%3+1,j=(k+1)%3+1;
    // ATen cross is a fused multiply-subtract; preserve its rounding.
    float cross=fmaf(a[i],b[j],-a[j]*b[i]);
    z[k]=(a[0]*b[k]+b[0]*a[k])+cross;
  }
}
__global__ void outputKernel(Attr a,float* x,float* s,float* q,int n,bool log_scale) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
  for(int k=0;k<3;++k) {
    x[row*3+k]=a.x[row*a.x0+k*a.x1]+a.dx[row*3+k];
    float v=a.s[row*a.s0+k*a.s1],d=a.ds[row*3+k];
    s[row*3+k]=log_scale?v+d:v*expf(d);
  }
  float residual[4],base[4],out[4];
  for(int k=0;k<4;++k) {residual[k]=a.dq[row*4+k]+(k==0?1.0f:0.0f);base[k]=a.q[row*a.q0+k*a.q1];}
  normalize4(residual);compose(residual,base,out);normalize4(out);
  for(int k=0;k<4;++k)q[row*4+k]=out[k];
}
__global__ void reluGradKernel(float* gradient,const float* activation,int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count && !(activation[i]>0))gradient[i]=0;
}
__global__ void sharedGradKernel(float* out,const float* a,const float* b,const float* c,
                                 const float* activation,int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) {
    float v=a?a[i]:0;if(b)v+=b[i];if(c)v+=c[i];
    out[i]=activation[i]>0?v:0;
  }
}
__global__ void biasGradKernel(const float* dy,float* db,int n,int dim) {
  __shared__ float partial[256];int col=blockIdx.x,tid=threadIdx.x;
  float sum=0;for(int row=tid;row<n;row+=256)sum+=dy[row*dim+col];
  partial[tid]=sum;__syncthreads();
  for(int stride=128;stride;stride/=2) {
    if(tid<stride)partial[tid]+=partial[tid+stride];__syncthreads();
  }
  if(!tid)db[col]=partial[0];
}
torch::Tensor linearBackward(Blas& blas,const torch::Tensor& x,const torch::Tensor& w,
    const torch::Tensor& dy,torch::Tensor& dw,torch::Tensor& db) {
  const int n=x.size(0),in=x.size(1),out=w.size(0);
  auto options=x.options().requires_grad(false);
  auto dx=torch::empty({n,in},options);dw=torch::empty_like(w);db=torch::empty({out},options);
  const float one=1,zero=0;
  if(n) {
    TORCH_CUDABLAS_CHECK(cublasSgemm(blas.h,CUBLAS_OP_N,CUBLAS_OP_N,in,n,out,
        &one,w.data_ptr<float>(),in,dy.data_ptr<float>(),out,&zero,dx.data_ptr<float>(),in));
    TORCH_CUDABLAS_CHECK(cublasSgemm(blas.h,CUBLAS_OP_N,CUBLAS_OP_T,in,out,n,
        &one,x.data_ptr<float>(),in,dy.data_ptr<float>(),out,&zero,dw.data_ptr<float>(),in));
  } else dw.zero_();
  biasGradKernel<<<out,256,0,c10::cuda::getCurrentCUDAStream()>>>(dy.data_ptr<float>(),db.data_ptr<float>(),n,out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return dx;
}
__device__ void normalizeVjp(const float* q,const float* g,float* dx) {
  float norm=norm4(q),denom=norm<1e-12f?1e-12f:norm,t[4];
  for(int k=0;k<4;++k)t[k]=-g[k]*((q[k]/denom)/denom);
  float gn=(t[0]+t[1])+(t[2]+t[3]);
  if(norm<1e-12f)gn=0;
  // Keep sqrt's zero singularity, exactly as the legacy autograd chain.
  float gs=gn/(2.0f*norm);
  // MulBackward's two q*q edges arrive separately after the direct div edge.
  for(int k=0;k<4;++k)dx[k]=(g[k]/denom+gs*q[k])+gs*q[k];
}
__device__ void composeVjp(const float* a,const float* b,const float* g,float* ga,float* gb) {
  float sa=(g[1]*b[1]+g[3]*b[3])+g[2]*b[2];
  float sb=(g[1]*a[1]+g[3]*a[3])+g[2]*a[2];
  ga[0]=sa+g[0]*b[0];gb[0]=sb+g[0]*a[0];
  for(int k=1;k<4;++k) {
    int i=k%3+1,j=(k+1)%3+1;
    // Match cross backward, vector-product accumulation, then scalar dot.
    float ca=fmaf(b[i],g[j],-b[j]*g[i]);
    float cb=fmaf(g[i],a[j],-g[j]*a[i]);
    ga[k]=(ca+g[k]*b[0])+(-g[0]*b[k]);
    gb[k]=(cb+g[k]*a[0])+(-g[0]*a[k]);
  }
}
struct Upstream {const float *x,*s,*q;};
struct OutputGrad {float *x,*s,*q,*dx,*ds,*dq;};
__global__ void outputGradKernel(Attr a,Upstream g,OutputGrad out,int n,bool log_scale) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
  for(int k=0;k<3;++k) {
    if(g.x){float gx=g.x[row*3+k];out.x[row*3+k]=gx;out.dx[row*3+k]=gx;}
    if(g.s) {
      float gs=g.s[row*3+k],e=expf(a.ds[row*3+k]);
      out.s[row*3+k]=log_scale?gs:gs*e;
      out.ds[row*3+k]=log_scale?gs:(gs*a.s[row*a.s0+k*a.s1])*e;
    }
  }
  if(!g.q)return;
  float raw[4],normalized[4],base[4],z[4],gz[4],ga[4],gb[4],dq[4];
  for(int k=0;k<4;++k) {
    raw[k]=a.dq[row*4+k]+(k==0?1.0f:0.0f);normalized[k]=raw[k];
    base[k]=a.q[row*a.q0+k*a.q1];
  }
  normalize4(normalized);compose(normalized,base,z);
  normalizeVjp(z,g.q+row*4,gz);composeVjp(normalized,base,gz,ga,gb);normalizeVjp(raw,ga,dq);
  for(int k=0;k<4;++k){out.q[row*4+k]=gb[k];out.dq[row*4+k]=dq[k];}
}
}  // namespace

std::vector<torch::Tensor> mlpForward(const torch::Tensor& feature,
    const std::vector<torch::Tensor>& parameters,MlpBackend backend) {
  TORCH_CHECK(feature.dim()==2 && feature.size(0)<=INT_MAX,"Invalid TF feature");
  const int n=feature.size(0),f=feature.size(1);
  checkMatrix(feature,feature,n,f);
  TORCH_CHECK(parameters.size()==14 && parameters[0].dim()==2,"TF requires seven W/b pairs");
  const int h=parameters[0].size(0);
  TORCH_CHECK(h>0 && static_cast<int64_t>(n)*h<=INT_MAX,"TF neural indexing overflow");
  for(int layer=0;layer<7;++layer) {
    int out=layer==2||layer==4?3:(layer==6?4:h),in=layer==0?f:h;
    checkMatrix(parameters[layer*2],feature,out,in);
    auto b=parameters[layer*2+1];checkTensor(b,feature);
    TORCH_CHECK(b.dim()==1 && b.numel()==out && b.is_contiguous(),"Invalid TF bias");
  }
  c10::cuda::CUDAGuard guard(feature.device());
  auto options=feature.options().requires_grad(false);
  std::vector<torch::Tensor> context{feature};
  for(int i=0;i<4;++i)context.push_back(torch::empty({n,h},options));
  for(int i=0;i<3;++i)context.push_back(torch::empty({n,i==2?4:3},options));
  if(backend==MlpBackend::Warp32) {
    TORCH_CHECK(h==32 && f==16,"Warp32 TF requires F16/H32");
    Neural a{};for(int i=0;i<14;++i)a.p[i]=parameters[i].data_ptr<float>();
    a.shared=context[1].data_ptr<float>();
    for(int i=0;i<3;++i){a.head[i]=context[2+i].data_ptr<float>();a.residual[i]=context[5+i].data_ptr<float>();}
    if(n){warpMlpKernel<<<(n+3)/4,128,0,c10::cuda::getCurrentCUDAStream()>>>(feature.data_ptr<float>(),a,n);C10_CUDA_KERNEL_LAUNCH_CHECK();}
  } else {
    Blas blas;linear(blas,feature,parameters[0],parameters[1],context[1],true);
    for(int i=0;i<3;++i) {
      int l=2+4*i;
      linear(blas,context[1],parameters[l],parameters[l+1],context[2+i],true);
      linear(blas,context[2+i],parameters[l+2],parameters[l+3],context[5+i],false);
    }
  }
  return context;
}
std::vector<torch::Tensor> outputForward(const torch::Tensor& xyz,const torch::Tensor& scale,
    const torch::Tensor& quaternion,const std::vector<torch::Tensor>& residual,bool log_scale) {
  TORCH_CHECK(xyz.dim()==2 && xyz.size(1)==3 && xyz.size(0)<=INT_MAX && residual.size()==3,"Invalid TF output inputs");
  const int n=xyz.size(0);
  for(auto t:{xyz,scale,quaternion})checkTensor(t,xyz);
  TORCH_CHECK(scale.dim()==2 && scale.size(0)==n && scale.size(1)==3 &&
              quaternion.dim()==2 && quaternion.size(0)==n && quaternion.size(1)==4,"Invalid TF attributes");
  for(int i=0;i<3;++i)checkMatrix(residual[i],xyz,n,i==2?4:3);
  c10::cuda::CUDAGuard guard(xyz.device());
  auto options=xyz.options().requires_grad(false);
  std::vector<torch::Tensor> out{torch::empty({n,3},options),torch::empty({n,3},options),torch::empty({n,4},options)};
  Attr a{xyz.data_ptr<float>(),scale.data_ptr<float>(),quaternion.data_ptr<float>(),
      residual[0].data_ptr<float>(),residual[1].data_ptr<float>(),residual[2].data_ptr<float>(),
      xyz.stride(0),xyz.stride(1),scale.stride(0),scale.stride(1),quaternion.stride(0),quaternion.stride(1)};
  if(n){outputKernel<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(a,out[0].data_ptr<float>(),out[1].data_ptr<float>(),out[2].data_ptr<float>(),n,log_scale);C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return out;
}
std::vector<torch::Tensor> mlpBackward(const std::vector<torch::Tensor>& context,
    const std::vector<torch::Tensor>& parameters,const std::vector<torch::Tensor>& gradients) {
  TORCH_CHECK(context.size()>=5 && parameters.size()==14 && gradients.size()==3,"Invalid MLP backward context");
  const auto& f=context[0];const int n=f.size(0),h=context[1].size(1);
  c10::cuda::CUDAGuard guard(f.device());Blas blas;
  for(int i=1;i<5;++i)checkMatrix(context[i],f,n,h);
  std::vector<torch::Tensor> result(15),branches(3);
  bool any=false;
  for(int i=0;i<3;++i) {
    if(!gradients[i].defined())continue;
    any=true;
    checkMatrix(gradients[i],f,n,i==2?4:3);
    int l=2+4*i;
    auto dh=linearBackward(blas,context[2+i],parameters[l+2],gradients[i],result[l+3],result[l+4]);
    if(n){reluGradKernel<<<(n*h+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(dh.data_ptr<float>(),context[2+i].data_ptr<float>(),n*h);C10_CUDA_KERNEL_LAUNCH_CHECK();}
    branches[i]=linearBackward(blas,context[1],parameters[l],dh,result[l+1],result[l+2]);
  }
  if(!any)return result;
  auto dh=torch::empty_like(context[1]);
  if(n){sharedGradKernel<<<(n*h+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(dh.data_ptr<float>(),branches[0].defined()?branches[0].data_ptr<float>():nullptr,branches[1].defined()?branches[1].data_ptr<float>():nullptr,branches[2].defined()?branches[2].data_ptr<float>():nullptr,context[1].data_ptr<float>(),n*h);C10_CUDA_KERNEL_LAUNCH_CHECK();}
  result[0]=linearBackward(blas,f,parameters[0],dh,result[1],result[2]);
  return result;
}
std::vector<torch::Tensor> outputBackward(const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& gradients,bool log_scale) {
  TORCH_CHECK(inputs.size()==6 && gradients.size()==3,"Invalid output backward context");
  const auto& x=inputs[0];const int n=x.size(0);
  c10::cuda::CUDAGuard guard(x.device());
  // Inputs were validated in forward; saved variable checks guard their versions.
  for(int i=0;i<3;++i)if(gradients[i].defined())checkMatrix(gradients[i],x,n,i==2?4:3);
  std::vector<torch::Tensor> result(6);
  for(int i=0;i<6;++i)if(gradients[i%3].defined())result[i]=torch::empty(inputs[i].sizes(),x.options().requires_grad(false));
  Attr a{inputs[0].data_ptr<float>(),inputs[1].data_ptr<float>(),inputs[2].data_ptr<float>(),
      inputs[3].data_ptr<float>(),inputs[4].data_ptr<float>(),inputs[5].data_ptr<float>(),
      inputs[0].stride(0),inputs[0].stride(1),inputs[1].stride(0),inputs[1].stride(1),inputs[2].stride(0),inputs[2].stride(1)};
  Upstream g{gradients[0].defined()?gradients[0].data_ptr<float>():nullptr,gradients[1].defined()?gradients[1].data_ptr<float>():nullptr,gradients[2].defined()?gradients[2].data_ptr<float>():nullptr};
  OutputGrad out{};float** ptr[6]={&out.x,&out.s,&out.q,&out.dx,&out.ds,&out.dq};
  for(int i=0;i<6;++i)if(result[i].defined())*ptr[i]=result[i].data_ptr<float>();
  if(n){outputGradKernel<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(a,g,out,n,log_scale);C10_CUDA_KERNEL_LAUNCH_CHECK();}
  return result;
}
}  // namespace tf_cuda

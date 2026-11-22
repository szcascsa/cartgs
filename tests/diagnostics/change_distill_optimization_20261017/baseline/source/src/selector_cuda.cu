#include "include/selector_cuda.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Context.h>
#include <cublasLt.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>
#include <map>
#include <memory>
#include <tuple>

namespace selector_cuda {
namespace {
cudaStream_t stream() { return c10::cuda::getCurrentCUDAStream(); }
void gemm(char ta,char tb,int m,int n,int k,float alpha,const float* a,int lda,
          const float* b,int ldb,float beta,float* c,int ldc) {
  auto handle=at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasSgemm(handle,ta=='t'?CUBLAS_OP_T:CUBLAS_OP_N,
      tb=='t'?CUBLAS_OP_T:CUBLAS_OP_N,m,n,k,&alpha,a,lda,b,ldb,&beta,c,ldc));
}
struct LtPlan {
  int rows;
  cublasLtMatmulDesc_t desc=nullptr;
  cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
  cublasLtMatmulPreference_t preference=nullptr;
  cublasLtMatmulHeuristicResult_t algorithm{};
  torch::Tensor workspace;
  LtPlan(int n,int dim,int out,bool activation,const torch::Tensor& reference,const torch::Tensor& bias):rows(n) {
    auto compute=at::globalContext().allowTF32CuBLAS()?CUBLAS_COMPUTE_32F_FAST_TF32:CUBLAS_COMPUTE_32F;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescCreate(&desc,compute,CUDA_R_32F));
    auto transpose=CUBLAS_OP_T;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));
    auto epilogue=activation?CUBLASLT_EPILOGUE_RELU_BIAS:CUBLASLT_EPILOGUE_BIAS;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_EPILOGUE,&epilogue,sizeof(epilogue)));
    const auto* bias_pointer=bias.data_ptr<float>();
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias_pointer,sizeof(bias_pointer)));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&a,CUDA_R_32F,dim,out,dim));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&b,CUDA_R_32F,dim,n,dim));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&c,CUDA_R_32F,out,n,out));
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    size_t bytes=1024*1024;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
    workspace=torch::empty({static_cast<int64_t>(bytes)},reference.options().dtype(torch::kUInt8).requires_grad(false));
    int returned=0;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(at::cuda::getCurrentCUDABlasLtHandle(),desc,a,b,c,c,
        preference,1,&algorithm,&returned));
    TORCH_CHECK(returned==1,"No FP32 cuBLASLt Selector algorithm");
  }
  ~LtPlan() {
    if(preference)cublasLtMatmulPreferenceDestroy(preference);
    if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);
    if(desc)cublasLtMatmulDescDestroy(desc);
  }
};
void gemmBiasRelu(const torch::Tensor& x,const torch::Tensor& w,const torch::Tensor& bias,torch::Tensor& y,bool activation=true) {
  // cuBLAS wrappers inside LibTorch are hidden symbols in the native 2.3.1
  // distribution. Use public cuBLASLt directly, with stream-local workspaces.
  using Key=std::tuple<int,int,int,bool,bool,uintptr_t>;
  thread_local std::map<Key,std::unique_ptr<LtPlan>> plans;
  // One current row-count per layer layout/stream: online M changes on append,
  // prune and elastic entry. Never retain a 1 MiB workspace for every past M.
  Key key{x.get_device(),x.size(1),y.size(1),at::globalContext().allowTF32CuBLAS(),activation,reinterpret_cast<uintptr_t>(stream())};
  auto& plan=plans[key];if(!plan || plan->rows!=x.size(0))
    plan=std::make_unique<LtPlan>(x.size(0),x.size(1),y.size(1),activation,x,bias);
  const auto* pointer=bias.data_ptr<float>();
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(plan->desc,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&pointer,sizeof(pointer)));
  const float alpha=1,beta=0;
  TORCH_CUDABLAS_CHECK(cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(),plan->desc,&alpha,w.data_ptr<float>(),plan->a,
      x.data_ptr<float>(),plan->b,&beta,y.data_ptr<float>(),plan->c,y.data_ptr<float>(),plan->c,&plan->algorithm.algo,
      plan->workspace.data_ptr(),plan->workspace.numel(),stream()));
}
void check(const torch::Tensor& t,const torch::Tensor& reference) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type()==torch::kFloat32 && t.device()==reference.device(),
              "Selector CUDA requires same-device FP32 CUDA tensors");
}
void validate(const std::vector<torch::Tensor>& x,const std::vector<torch::Tensor>& p) {
  TORCH_CHECK(x.size()==4 && p.size()==18,"Selector CUDA expects 4 inputs/18 parameters");
  TORCH_CHECK(x[0].dim()==2 && x[0].size(0)<=INT_MAX/128,"Selector row indexing overflow");
  const int widths[4]={3,4,3,1};
  for(int b=0;b<4;++b) {
    check(x[b],x[0]);
    TORCH_CHECK(x[b].dim()==2 && x[b].size(1)==widths[b] &&
        (x[b].size(0)==x[0].size(0)||(b==3&&x[b].size(0)==1)),"Selector input shape mismatch");
    for(int layer=0;layer<2;++layer) {
      int i=b*4+layer*2,dim=layer?32:widths[b];
      check(p[i],x[0]);check(p[i+1],x[0]);
      TORCH_CHECK(p[i].sizes()==torch::IntArrayRef({32,dim}) && p[i+1].sizes()==torch::IntArrayRef({32}) &&
          p[i].is_contiguous() && p[i+1].is_contiguous(),"Invalid Selector branch parameter");
    }
  }
  check(p[16],x[0]);check(p[17],x[0]);
  TORCH_CHECK(p[16].sizes()==torch::IntArrayRef({2,128}) && p[17].sizes()==torch::IntArrayRef({2}) &&
      p[16].is_contiguous() && p[17].is_contiguous(),"Invalid Selector final parameter");
}
__global__ void fillBias(float* y,const float* bias,int count,int width) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)y[i]=bias[i%width];
}
__global__ void relu(float* y,int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count && y[i]<0)y[i]=0;
}
void linear(const torch::Tensor& x,const torch::Tensor& w,const torch::Tensor& b,torch::Tensor& y) {
  const int n=x.size(0),dim=x.size(1);if(!n)return;
  if(dim>1) {
    gemmBiasRelu(x,w,b,y);
  } else {
    // Singleton weight.t() is contiguous in legacy addmm: preserve its
    // non-transposed [32,1] BLAS layout as well as the bias accumulation.
    fillBias<<<(n*32+255)/256,256,0,stream()>>>(y.data_ptr<float>(),b.data_ptr<float>(),n*32,32);
    gemm('n','n',32,n,1,1.0f,w.data_ptr<float>(),32,x.data_ptr<float>(),1,
        1.0f,y.data_ptr<float>(),32);
    relu<<<(n*32+255)/256,256,0,stream()>>>(y.data_ptr<float>(),n*32);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void repeatRows(const float* source,float* dest,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*32)dest[i]=source[i%32];
}
__global__ void addRatio(float* logits,const float* ratio,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*2)logits[i]+=ratio[i%2];
}
struct Features {const float* h[4];int ratio_rows;};
__global__ void packFeatures(Features f,float4* packed,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*32)return;
  int row=i/32,branch=(i%32)/8,col=i%8;
  packed[i]=reinterpret_cast<const float4*>(f.h[branch])[(branch==3&&f.ratio_rows==1?0:row)*8+col];
}
__global__ void splitGradient(const float* full,float* branch,int n,int offset) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*32)branch[i]=full[(i/32)*128+offset+i%32];
}
struct WarpArgs {
  const float *x[3],*p[18];float *h[6],*out;const float* ratio;
  int ratio_rows;
};
__global__ void warpNetwork(WarpArgs a,int n) {
  const int lane=threadIdx.x%32,row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  if(row>=n)return;
  float final[2]={0,0};
  for(int b=0;b<3;++b) {
    const int dim=b==1?4:3,base=b*4;
    float first=0;
    for(int k=0;k<dim;++k)first=fmaf(a.x[b][row*dim+k],__ldg(a.p[base]+lane*dim+k),first);
    first+=__ldg(a.p[base+1]+lane);first=first<0?0:first;a.h[b*2][row*32+lane]=first;
    float second=0;
    for(int k=0;k<32;++k)second=fmaf(__shfl_sync(0xffffffff,first,k),__ldg(a.p[base+2]+lane*32+k),second);
    second+=__ldg(a.p[base+3]+lane);second=second<0?0:second;a.h[b*2+1][row*32+lane]=second;
    for(int k=0;k<32;++k) {
      float value=__shfl_sync(0xffffffff,second,k);
      if(lane<2)final[0]=fmaf(value,__ldg(a.p[16]+lane*128+b*32+k),final[0]);
    }
  }
  if(lane<2) {
    for(int k=0;k<32;++k)final[0]=fmaf(a.ratio[(a.ratio_rows==1?0:row)*32+k],
        __ldg(a.p[16]+lane*128+96+k),final[0]);
    a.out[row*2+lane]=final[0]+a.p[17][lane];
  }
}
__global__ void reluVjp(float* dy,const float* h,int count,int stride=32) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count && h[(i/32)*stride+i%32]<=0)dy[i]=0;
}
__global__ void splitRelu(const float* full,float* branch,const float* h,int n,int offset,int hstride) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n*32)branch[i]=h[(i/32)*hstride+i%32]<=0?0:full[(i/32)*128+offset+i%32];
}
template<bool Mask>
__global__ void directFinal(const float* dy,const float* w,float* branch,const float* h,int n,int offset,int hstride) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n*32) {
    int row=i/32,col=offset+i%32;
    float value=fmaf(dy[row*2+1],w[128+col],dy[row*2]*w[col]);
    branch[i]=Mask && h[row*hstride+i%32]<=0?0:value;
  }
}
__global__ void maskBias(const float* input,float* output,const float* h,float* partial,
                         int rows,int input_stride,int offset,int hstride) {
  __shared__ float sums[8][32];
  int col=threadIdx.x%32,warp=threadIdx.x/32;
  float sum=0;
  for(int r=blockIdx.x*256+warp;r<min(rows,(int(blockIdx.x)+1)*256);r+=8) {
    float v=h[r*hstride+col]<=0?0:input[r*input_stride+offset+col];
    output[r*32+col]=v;sum+=v;
  }
  sums[warp][col]=sum;__syncthreads();
  if(warp==0){float v=0;for(int k=0;k<8;++k)v+=sums[k][col];partial[blockIdx.x*32+col]=v;}
}
torch::Tensor maskedBias(const torch::Tensor& input,torch::Tensor& output,const torch::Tensor& h,int offset) {
  int n=output.size(0),blocks=(n+255)/256;
  auto partial=torch::empty({blocks,32},output.options());
  if(n)maskBias<<<blocks,256,0,stream()>>>(input.data_ptr<float>(),output.data_ptr<float>(),h.data_ptr<float>(),
      partial.data_ptr<float>(),n,input.stride(0),offset,h.size(0)==1?0:h.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return partial.sum(0);
}
void batchedGemm(char ta,char tb,int m,int n,int k,const std::vector<torch::Tensor>& a,int lda,
                 const std::vector<torch::Tensor>& b,int ldb,std::vector<torch::Tensor>& c,int ldc) {
  int count=a.size();
  auto host=torch::empty({count*3},torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
  auto ptr=host.data_ptr<int64_t>();
  for(int i=0;i<count;++i){ptr[i]=reinterpret_cast<int64_t>(a[i].data_ptr<float>());
    ptr[count+i]=reinterpret_cast<int64_t>(b[i].data_ptr<float>());ptr[2*count+i]=reinterpret_cast<int64_t>(c[i].data_ptr<float>());}
  auto device=host.to(a[0].device(),true);auto base=device.data_ptr<int64_t>();
  const float alpha=1,beta=0;
  TORCH_CUDABLAS_CHECK(cublasSgemmBatched(at::cuda::getCurrentCUDABlasHandle(),ta=='t'?CUBLAS_OP_T:CUBLAS_OP_N,
      tb=='t'?CUBLAS_OP_T:CUBLAS_OP_N,m,n,k,&alpha,reinterpret_cast<const float**>(base),lda,
      reinterpret_cast<const float**>(base+count),ldb,&beta,reinterpret_cast<float**>(base+2*count),ldc,count));
}
__global__ void sumBias(const float* dy,float* db,int rows,int dim) {
  __shared__ float partial[256];const int col=blockIdx.x,tid=threadIdx.x;
  float sum=0;for(int row=tid;row<rows;row+=256)sum+=dy[row*dim+col];
  partial[tid]=sum;__syncthreads();
  for(int s=128;s;s/=2){if(tid<s)partial[tid]+=partial[tid+s];__syncthreads();}
  if(!tid)db[col]=partial[0];
}
torch::Tensor biasGradient(const torch::Tensor& dy,bool legacy_reduction=false) {
  // Strict candidate keeps the exact ATen reduction tree. The first custom
  // reduction is measured separately; no tolerance is relaxed to accommodate it.
  if(legacy_reduction)return dy.sum(0);
  auto db=torch::empty({dy.size(1)},dy.options());
  sumBias<<<dy.size(1),256,0,stream()>>>(dy.data_ptr<float>(),db.data_ptr<float>(),dy.size(0),dy.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return db;
}
// Row-major DX=DY*W and DW=DY^T*X. Weight gradients use GEMM, never per-point atomics.
torch::Tensor linearVjp(const torch::Tensor& x,const torch::Tensor& w,const torch::Tensor& dy,
                        torch::Tensor& dw,torch::Tensor& db,bool legacy_reduction,const torch::Tensor& precomputed_db={}) {
  int rows=x.size(0),in=x.size(1),out=dy.size(1);
  auto dx=torch::empty_like(x);dw=torch::empty_like(w);db=precomputed_db.defined()?precomputed_db:biasGradient(dy,legacy_reduction);
  if(rows) {
    gemm('n','n',in,rows,out,1.f,w.data_ptr<float>(),in,
        dy.data_ptr<float>(),out,0.f,dx.data_ptr<float>(),in);
    gemm('n','t',in,out,rows,1.f,x.data_ptr<float>(),in,
        dy.data_ptr<float>(),out,0.f,dw.data_ptr<float>(),in);
  } else dw.zero_();
  return dx;
}
}  // namespace

std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& p,Candidate candidate) {
  validate(inputs,p);c10::cuda::CUDAGuard guard(inputs[0].device());
  std::vector<torch::Tensor> x;for(auto t:inputs)x.push_back(t.contiguous());
  const int n=x[0].size(0);auto options=x[0].options().requires_grad(false);
  std::vector<torch::Tensor> h;
  for(int b=0;b<4;++b)for(int l=0;l<2;++l)h.push_back(torch::empty({x[b].size(0),32},options));
  h.push_back(torch::empty({n,2},options));
  linear(x[3],p[12],p[13],h[6]);linear(h[6],p[14],p[15],h[7]);
  if(candidate==Candidate::Warp) {
    WarpArgs a{};for(int b=0;b<3;++b)a.x[b]=x[b].data_ptr<float>();
    for(int i=0;i<18;++i)a.p[i]=p[i].data_ptr<float>();
    for(int i=0;i<6;++i)a.h[i]=h[i].data_ptr<float>();
    a.ratio=h[7].data_ptr<float>();a.ratio_rows=h[7].size(0);a.out=h[8].data_ptr<float>();
    if(n)warpNetwork<<<(n+3)/4,128,0,stream()>>>(a,n);
  } else {
    for(int b=0;b<3;++b){linear(x[b],p[b*4],p[b*4+1],h[b*2]);linear(h[b*2],p[b*4+2],p[b*4+3],h[b*2+1]);}
    if(candidate==Candidate::CublasStrict) {
      // Split final GEMMs failed the exact tie fixture. Preserve the legacy
      // 128-term GEMM and its reduction layout. This candidate intentionally
      // retains a packed [M,128] workspace; it does not claim to remove it.
      auto packed=torch::empty({n,128},options);Features f{};
      for(int b=0;b<4;++b)f.h[b]=h[b*2+1].data_ptr<float>();f.ratio_rows=h[7].size(0);
      if(n) {
        packFeatures<<<(n*32+255)/256,256,0,stream()>>>(f,reinterpret_cast<float4*>(packed.data_ptr<float>()),n);
        gemmBiasRelu(packed,p[16],p[17],h[8],false);
      }
      h.push_back(packed);
    } else if(n) {
      fillBias<<<(n*2+255)/256,256,0,stream()>>>(h[8].data_ptr<float>(),p[17].data_ptr<float>(),n*2,2);
      for(int b=0;b<4;++b) {
        auto feature=h[b*2+1];
        if(b==3 && feature.size(0)==1 && n!=1) {
          auto contribution=torch::empty({1,2},options);
          gemm('t','n',2,1,32,1.f,p[16].data_ptr<float>()+96,128,
              feature.data_ptr<float>(),32,0.f,contribution.data_ptr<float>(),2);
          addRatio<<<(n*2+255)/256,256,0,stream()>>>(h[8].data_ptr<float>(),contribution.data_ptr<float>(),n);
          continue;
        }
        gemm('t','n',2,n,32,1.f,p[16].data_ptr<float>()+b*32,128,
            feature.data_ptr<float>(),32,1.f,h[8].data_ptr<float>(),2);
      }
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();return h;
}
std::vector<torch::Tensor> backward(const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& p,const std::vector<torch::Tensor>& h,
    const torch::Tensor& upstream,BackwardStrategy strategy) {
  validate(inputs,p);c10::cuda::CUDAGuard guard(inputs[0].device());
  std::vector<torch::Tensor> x;for(auto t:inputs)x.push_back(t.contiguous());
  auto dy=upstream.contiguous();const int n=x[0].size(0);
  // cuBLAS selects a different k=2 accumulator for very small row counts.
  // Keep its original implementation where direct routing is not bitwise equal.
  if(n<=16 && (strategy==BackwardStrategy::DirectFinal || strategy==BackwardStrategy::DirectFinalRelu))
    strategy=BackwardStrategy::FusedActivation;
  const bool strict=h.size()==9;
  std::vector<torch::Tensor> result(22);
  auto dw=torch::empty_like(p[16]);result[20]=dw;result[21]=biasGradient(dy,strict);
  torch::Tensor full_dx;
  const bool routed=strategy==BackwardStrategy::DirectFinal || strategy==BackwardStrategy::DirectFinalRelu;
  if(strict) {
    if(!routed)full_dx=torch::empty({n,128},dy.options());
    if(n) {
      if(full_dx.defined())gemm('n','n',128,n,2,1.f,p[16].data_ptr<float>(),128,dy.data_ptr<float>(),2,0.f,full_dx.data_ptr<float>(),128);
      gemm('n','t',128,2,n,1.f,h[8].data_ptr<float>(),128,dy.data_ptr<float>(),2,0.f,dw.data_ptr<float>(),128);
    } else dw.zero_();
  }
  const bool grouped=strict && n && (strategy==BackwardStrategy::BatchedWeight || strategy==BackwardStrategy::GroupedBranch);
  if(grouped) {
    std::vector<torch::Tensor> d2,d1,weights,first,dw2,dw1;
    for(int b=0;b<3;++b){
      d2.push_back(torch::empty({n,32},dy.options()));d1.push_back(torch::empty({n,32},dy.options()));
      splitRelu<<<(n*32+255)/256,256,0,stream()>>>(full_dx.data_ptr<float>(),d2[b].data_ptr<float>(),h[b*2+1].data_ptr<float>(),n,b*32,h[b*2+1].stride(0));
      result[4+b*4+3]=biasGradient(d2[b],true);
      weights.push_back(p[b*4+2]);first.push_back(h[b*2]);
      dw2.push_back(torch::empty_like(p[b*4+2]));dw1.push_back(torch::empty_like(p[b*4]));
    }
    if(strategy==BackwardStrategy::GroupedBranch)batchedGemm('n','n',32,n,32,weights,32,d2,32,d1,32);
    else for(int b=0;b<3;++b)gemm('n','n',32,n,32,1.f,weights[b].data_ptr<float>(),32,d2[b].data_ptr<float>(),32,0.f,d1[b].data_ptr<float>(),32);
    for(int b=0;b<3;++b){
      reluVjp<<<(n*32+255)/256,256,0,stream()>>>(d1[b].data_ptr<float>(),first[b].data_ptr<float>(),n*32);
      result[4+b*4+1]=biasGradient(d1[b],true);result[b]=torch::empty_like(x[b]);
      gemm('n','n',x[b].size(1),n,32,1.f,p[b*4].data_ptr<float>(),x[b].size(1),d1[b].data_ptr<float>(),32,0.f,result[b].data_ptr<float>(),x[b].size(1));
      result[4+b*4]=dw1[b];result[4+b*4+2]=dw2[b];
    }
    batchedGemm('n','t',32,32,n,first,32,d2,32,dw2,32);
    std::vector<torch::Tensor> xx{x[0],x[2]},dd{d1[0],d1[2]},ww{dw1[0],dw1[2]};
    batchedGemm('n','t',3,32,n,xx,3,dd,32,ww,3);
    gemm('n','t',4,32,n,1.f,x[1].data_ptr<float>(),4,d1[1].data_ptr<float>(),32,0.f,dw1[1].data_ptr<float>(),4);
  }
  for(int b=0;b<4;++b) {
    if(grouped && b<3)continue;
    auto feature=h[b*2+1],dx=torch::empty({n,32},dy.options());
    if(!strict && feature.size(0)==1 && n!=1) {
      auto expanded=torch::empty({n,32},dy.options());
      if(n)repeatRows<<<(n*32+255)/256,256,0,stream()>>>(feature.data_ptr<float>(),expanded.data_ptr<float>(),n);
      feature=expanded;
    }
    torch::Tensor db2,db1;
    const bool fused=strict && strategy!=BackwardStrategy::Current && strategy!=BackwardStrategy::DirectFinal;
    if(strict) {
      if(routed) {
        if(n && strategy==BackwardStrategy::DirectFinalRelu)
          directFinal<true><<<(n*32+255)/256,256,0,stream()>>>(dy.data_ptr<float>(),p[16].data_ptr<float>(),dx.data_ptr<float>(),h[b*2+1].data_ptr<float>(),n,b*32,h[b*2+1].size(0)==1?0:h[b*2+1].stride(0));
        else if(n)directFinal<false><<<(n*32+255)/256,256,0,stream()>>>(dy.data_ptr<float>(),p[16].data_ptr<float>(),dx.data_ptr<float>(),nullptr,n,b*32,0);
      } else if(strategy==BackwardStrategy::FusedBias && !(b==3 && inputs[3].size(0)==1 && n!=1))
        db2=maskedBias(full_dx,dx,h[b*2+1],b*32);
      else if(n && fused)splitRelu<<<(n*32+255)/256,256,0,stream()>>>(full_dx.data_ptr<float>(),dx.data_ptr<float>(),h[b*2+1].data_ptr<float>(),n,b*32,h[b*2+1].size(0)==1?0:h[b*2+1].stride(0));
      else if(n)splitGradient<<<(n*32+255)/256,256,0,stream()>>>(full_dx.data_ptr<float>(),dx.data_ptr<float>(),n,b*32);
    } else if(n) {
      gemm('n','n',32,n,2,1.f,p[16].data_ptr<float>()+b*32,128,
          dy.data_ptr<float>(),2,0.f,dx.data_ptr<float>(),32);
      gemm('n','t',32,2,n,1.f,feature.data_ptr<float>(),32,
          dy.data_ptr<float>(),2,0.f,dw.data_ptr<float>()+b*32,128);
    } else {dw.slice(1,b*32,(b+1)*32).zero_();}
    if(b==3 && inputs[3].size(0)==1 && n!=1)dx=biasGradient(dx,strict).reshape({1,32});
    int rows=dx.size(0);
    if(rows && !fused)reluVjp<<<(rows*32+255)/256,256,0,stream()>>>(dx.data_ptr<float>(),h[b*2+1].data_ptr<float>(),rows*32,h[b*2+1].stride(0));
    auto dh=linearVjp(h[b*2],p[b*4+2],dx,result[4+b*4+2],result[4+b*4+3],strict,db2);
    if(strategy==BackwardStrategy::FusedBias)db1=maskedBias(dh,dh,h[b*2],0);
    else if(rows)reluVjp<<<(rows*32+255)/256,256,0,stream()>>>(dh.data_ptr<float>(),h[b*2].data_ptr<float>(),rows*32);
    result[b]=linearVjp(x[b],p[b*4],dh,result[4+b*4],result[4+b*4+1],strict,db1);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}
}  // namespace selector_cuda

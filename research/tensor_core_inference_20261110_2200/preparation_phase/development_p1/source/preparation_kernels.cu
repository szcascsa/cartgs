#include <torch/extension.h>
#include <mma.h>
#include <cub/cub.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

torch::Tensor frustum_planes(const torch::Tensor&);
torch::Tensor frustum_test(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
std::vector<torch::Tensor> deployment_fused_candidates(const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);

namespace {
struct PreparationCounts {int64_t visible,eligible;};
struct PreparationAdd {
  __device__ PreparationCounts operator()(PreparationCounts a,PreparationCounts b) const {
    return {a.visible+b.visible,a.eligible+b.eligible};
  }
};
struct PreparationVisible {
  const bool* visible;const int64_t* owners;
  __device__ bool operator()(int64_t row) const {return visible[owners[row]];}
};

__global__ void preparationFrustum(int64_t n,const float* centers,const float* matrix,
    int64_t s0,int64_t s1,const float* scales,const bool* occupied,const int64_t* eligible,
    bool* output,PreparationCounts* partial,float* distances) {
  __shared__ float planes[24];
  const int tid=threadIdx.x;
  if(tid<24){int p=tid/4,c=tid%4;float a=matrix[c*s0+3*s1],b=matrix[c*s0+(p/2)*s1];
    planes[tid]=p==4?matrix[c*s0+2*s1]:(p%2?__fsub_rn(a,b):__fadd_rn(a,b));}
  __syncthreads();
  const int64_t i=int64_t(blockIdx.x)*blockDim.x+tid;
  bool visible=i<n && occupied[i];
  if(i<n){
    float x=centers[3*i],y=centers[3*i+1],z=centers[3*i+2];
    for(int p=0;p<6;++p){
      const float* plane=planes+p*4;
      float d=__fmaf_rn(x,plane[0],0.f);d=__fmaf_rn(y,plane[1],d);d=__fmaf_rn(z,plane[2],d);
      if(distances)distances[i*6+p]=d;
      float norm=__fadd_rn(__fadd_rn(fabsf(plane[0]),fabsf(plane[2])),fabsf(plane[1]));
      float radius=__fmul_rn(__fmul_rn(scales[i],.5f),norm);
      visible=visible && __fadd_rn(__fadd_rn(d,plane[3]),radius)>=0.f;
    }
    output[i]=visible;
  }
  using Reduce=cub::BlockReduce<PreparationCounts,256>;
  __shared__ typename Reduce::TempStorage temp;
  auto sum=Reduce(temp).Reduce(PreparationCounts{int64_t(visible),visible?eligible[i]:0},PreparationAdd());
  if(!tid)partial[blockIdx.x]=sum;
}

// 真实TF32 WMMA候选，包含补零、转换和6列输出打包；不默认用于视锥判定。
__global__ void preparationTensorDistance(int64_t n,const float* centers,const float* planes,float* out) {
  namespace wm=nvcuda::wmma;
  __shared__ __align__(32) float a[16*8],b[8*16],c[16*16];
  int tid=threadIdx.x,begin=blockIdx.x*16;
  for(int j=tid;j<16*8;j+=32){int r=j/8,k=j%8;
    a[j]=wm::__float_to_tf32(begin+r<n && k<3?centers[(begin+r)*3+k]:0.f);}
  for(int j=tid;j<8*16;j+=32){int k=j/16,p=j%16;b[j]=wm::__float_to_tf32(k<3&&p<6?planes[p*4+k]:0.f);}
  __syncwarp();
  wm::fragment<wm::matrix_a,16,16,8,wm::precision::tf32,wm::row_major> af;
  wm::fragment<wm::matrix_b,16,16,8,wm::precision::tf32,wm::row_major> bf;
  wm::fragment<wm::accumulator,16,16,8,float> cf;
  wm::fill_fragment(cf,0.f);wm::load_matrix_sync(af,a,8);wm::load_matrix_sync(bf,b,16);
  wm::mma_sync(cf,af,bf,cf);wm::store_matrix_sync(c,cf,16,wm::mem_row_major);
  __syncwarp();
  for(int j=tid;j<16*6;j+=32){int r=j/6,p=j%6;if(begin+r<n)out[(begin+r)*6+p]=c[r*16+p];}
}
}

std::vector<torch::Tensor> preparation_frustum_debug(const torch::Tensor& centers,const torch::Tensor& projection,
    const torch::Tensor& scales,const torch::Tensor& occupied,const torch::Tensor& eligible,int64_t mode) {
  c10::cuda::CUDAGuard guard(centers.device());const int64_t n=centers.size(0);
  auto planes=frustum_planes(projection);torch::Tensor distance,visible;
  if(mode==1){distance=torch::empty({n,6},centers.options());visible=torch::empty({n},occupied.options());
    auto partial=torch::empty({2*((n+255)/256)},eligible.options());
    if(n)preparationFrustum<<<(n+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,centers.data_ptr<float>(),projection.data_ptr<float>(),
      projection.stride(0),projection.stride(1),scales.data_ptr<float>(),occupied.data_ptr<bool>(),eligible.data_ptr<int64_t>(),
      visible.data_ptr<bool>(),reinterpret_cast<PreparationCounts*>(partial.data_ptr<int64_t>()),distance.data_ptr<float>());
  }else{
    if(mode==2){distance=torch::empty({n,6},centers.options());
      if(n)preparationTensorDistance<<<(n+15)/16,32,0,c10::cuda::getCurrentCUDAStream()>>>(n,centers.data_ptr<float>(),planes.data_ptr<float>(),distance.data_ptr<float>());
    }else distance=at::matmul(centers,planes.slice(1,0,3).t());
    visible=frustum_test(distance,planes,scales,occupied);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {visible,distance};
}

std::vector<torch::Tensor> preparation_candidates(const torch::Tensor& centers,const torch::Tensor& projection,
    const torch::Tensor& scales,const torch::Tensor& occupied,const torch::Tensor& owners,
    const torch::Tensor& eligible,const torch::Tensor& rows,const torch::Tensor& counts,
    const torch::Tensor& workspace,const torch::Tensor& partial,int64_t mode) {
  c10::cuda::CUDAGuard guard(centers.device());const auto stream=c10::cuda::getCurrentCUDAStream();
  const int64_t anchors=centers.size(0),blocks=(anchors+255)/256;
  if(mode==2){auto planes=frustum_planes(projection);auto distance=torch::empty({anchors,6},centers.options());
    if(anchors)preparationTensorDistance<<<(anchors+15)/16,32,0,stream>>>(anchors,centers.data_ptr<float>(),planes.data_ptr<float>(),distance.data_ptr<float>());
    return deployment_fused_candidates(distance,planes,scales,occupied,owners,eligible,rows,counts,workspace,partial);}
  auto visible=torch::empty({anchors},occupied.options());
  if(anchors)preparationFrustum<<<blocks,256,0,stream>>>(anchors,centers.data_ptr<float>(),projection.data_ptr<float>(),
    projection.stride(0),projection.stride(1),scales.data_ptr<float>(),occupied.data_ptr<bool>(),eligible.data_ptr<int64_t>(),
    visible.data_ptr<bool>(),reinterpret_cast<PreparationCounts*>(partial.data_ptr<int64_t>()),nullptr);
  auto temporary=workspace;
  if(owners.numel()){
    cub::CountingInputIterator<int64_t> input(0);PreparationVisible predicate{visible.data_ptr<bool>(),owners.data_ptr<int64_t>()};size_t bytes=0;
    C10_CUDA_CHECK(cub::DeviceSelect::If(nullptr,bytes,input,rows.data_ptr<int64_t>(),counts.data_ptr<int64_t>(),int(owners.numel()),predicate,stream));
    if(temporary.numel()<int64_t(bytes))temporary=torch::empty({int64_t(bytes)},workspace.options());
    C10_CUDA_CHECK(cub::DeviceSelect::If(temporary.data_ptr(),bytes,input,rows.data_ptr<int64_t>(),counts.data_ptr<int64_t>(),int(owners.numel()),predicate,stream));
  }else C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr<int64_t>(),0,sizeof(int64_t),stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {visible,temporary};
}

#include <torch/extension.h>
#include <cub/cub.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {
struct AnchorCounts {int64_t visible,eligible;};
struct AddCounts {
  __device__ AnchorCounts operator()(AnchorCounts a,AnchorCounts b) const {
    return {a.visible+b.visible,a.eligible+b.eligible};
  }
};
struct VisibleRow {
  const bool* visible;const int64_t* owners;
  __device__ bool operator()(int64_t row) const {return visible[owners[row]];}
};

__global__ void testAndCount(int64_t n,const float* distance,const float* planes,const float* scales,
    const bool* occupied,const int64_t* eligible,bool* output,AnchorCounts* partial) {
  const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  bool visible=i<n && occupied[i];
  if(i<n) {
    for(int p=0;p<6;++p) {
      float norm=__fadd_rn(__fadd_rn(fabsf(planes[p*4]),fabsf(planes[p*4+2])),fabsf(planes[p*4+1]));
      float radius=__fmul_rn(__fmul_rn(scales[i],.5f),norm);
      float d=__fadd_rn(distance[i*6+p],planes[p*4+3]);
      visible=visible && __fadd_rn(d,radius)>=0.f;
    }
    output[i]=visible;
  }
  using Reduce=cub::BlockReduce<AnchorCounts,256>;
  __shared__ typename Reduce::TempStorage temp;
  const auto sum=Reduce(temp).Reduce(AnchorCounts{int64_t(visible),visible?eligible[i]:0},AddCounts());
  if(!threadIdx.x)partial[blockIdx.x]=sum;
}

__global__ void finishCounts(int n,const AnchorCounts* partial,int64_t* counts) {
  AnchorCounts value{0,0};
  for(int i=threadIdx.x;i<n;i+=blockDim.x)value=AddCounts()(value,partial[i]);
  using Reduce=cub::BlockReduce<AnchorCounts,128>;
  __shared__ typename Reduce::TempStorage temp;
  const auto sum=Reduce(temp).Reduce(value,AddCounts());
  if(!threadIdx.x){counts[1]=sum.visible;counts[2]=sum.eligible;}
}
}

std::vector<torch::Tensor> deployment_fused_candidates(const torch::Tensor& distance,const torch::Tensor& planes,
    const torch::Tensor& scales,const torch::Tensor& occupied,const torch::Tensor& owners,
    const torch::Tensor& eligible,const torch::Tensor& rows,const torch::Tensor& counts,
    const torch::Tensor& workspace,const torch::Tensor& partial) {
  c10::cuda::CUDAGuard guard(distance.device());
  const int64_t anchors=scales.numel(),blocks=(anchors+255)/256;
  TORCH_CHECK(partial.scalar_type()==at::kLong && partial.numel()>=blocks*2 && counts.numel()==3 &&
      eligible.numel()==anchors && distance.is_contiguous() && distance.sizes()==at::IntArrayRef({anchors,6}),
      "Fused deployment candidate workspace");
  auto visible=torch::empty({anchors},occupied.options());
  const auto stream=c10::cuda::getCurrentCUDAStream();
  if(anchors)testAndCount<<<blocks,256,0,stream>>>(anchors,distance.data_ptr<float>(),planes.data_ptr<float>(),
      scales.data_ptr<float>(),occupied.data_ptr<bool>(),eligible.data_ptr<int64_t>(),visible.data_ptr<bool>(),
      reinterpret_cast<AnchorCounts*>(partial.data_ptr<int64_t>()));
  auto temporary=workspace;
  if(owners.numel()) {
    cub::CountingInputIterator<int64_t> input(0);
    VisibleRow predicate{visible.data_ptr<bool>(),owners.data_ptr<int64_t>()};
    size_t bytes=0;
    C10_CUDA_CHECK(cub::DeviceSelect::If(nullptr,bytes,input,rows.data_ptr<int64_t>(),counts.data_ptr<int64_t>(),int(owners.numel()),predicate,stream));
    if(temporary.numel()<int64_t(bytes))temporary=torch::empty({int64_t(bytes)},workspace.options());
    C10_CUDA_CHECK(cub::DeviceSelect::If(temporary.data_ptr(),bytes,input,rows.data_ptr<int64_t>(),counts.data_ptr<int64_t>(),int(owners.numel()),predicate,stream));
  } else C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr<int64_t>(),0,sizeof(int64_t),stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {visible,temporary};
}

void deployment_finish_counts(const torch::Tensor& partial,const torch::Tensor& counts) {
  finishCounts<<<1,128,0,c10::cuda::getCurrentCUDAStream()>>>(partial.numel()/2,
      reinterpret_cast<const AnchorCounts*>(partial.data_ptr<int64_t>()),counts.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

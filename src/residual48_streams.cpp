#include "include/residual48_fusion.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

namespace mapping_cuda {
namespace {
struct HeadStreams {
  int device;cudaEvent_t ready,done[4];std::vector<c10::cuda::CUDAStream> streams;
  explicit HeadStreams(int d):device(d) {
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
    for(int j=0;j<4;++j){streams.push_back(c10::cuda::getStreamFromPool());C10_CUDA_CHECK(cudaEventCreateWithFlags(&done[j],cudaEventDisableTiming));}
  }
  ~HeadStreams(){c10::cuda::CUDAGuard guard(device);cudaEventDestroy(ready);for(auto event:done)cudaEventDestroy(event);}
};
}
std::vector<torch::Tensor> residual48HeadsInference(const torch::Tensor& hidden,torch::nn::ModuleDict& heads) {
  TORCH_CHECK(!torch::GradMode::is_enabled()&&hidden.is_cuda()&&hidden.scalar_type()==torch::kFloat32,"Parallel Residual48 heads are inference-only");
  c10::cuda::CUDAGuard device_guard(hidden.device());auto current=c10::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus capturing;C10_CUDA_CHECK(cudaStreamIsCapturing(current,&capturing));
  std::vector<torch::Tensor> output;
  if(capturing!=cudaStreamCaptureStatusNone) {
    for(const auto& name:{"xyz","scale","rotation","other"})output.push_back(heads->at<torch::nn::SequentialImpl>(name).forward(hidden));
    return output;
  }
  // One bounded device state per host thread. It caches no model or topology values.
  thread_local std::unique_ptr<HeadStreams> state;
  if(!state||state->device!=hidden.get_device())state=std::make_unique<HeadStreams>(hidden.get_device());
  C10_CUDA_CHECK(cudaEventRecord(state->ready,current));int j=0;
  for(const auto& name:{"xyz","scale","rotation","other"}) {
    auto& head=heads->at<torch::nn::SequentialImpl>(name);auto stream=state->streams[j];
    C10_CUDA_CHECK(cudaStreamWaitEvent(stream,state->ready,0));
    c10::cuda::CUDACachingAllocator::recordStream(hidden.storage().data_ptr(),stream);
    for(const auto& parameter:head.parameters())c10::cuda::CUDACachingAllocator::recordStream(parameter.storage().data_ptr(),stream);
    {c10::cuda::CUDAStreamGuard stream_guard(stream);output.push_back(head.forward(hidden));C10_CUDA_CHECK(cudaEventRecord(state->done[j],stream));}
    C10_CUDA_CHECK(cudaStreamWaitEvent(current,state->done[j],0));
    c10::cuda::CUDACachingAllocator::recordStream(output.back().storage().data_ptr(),current);++j;
  }
  return output;
}
}

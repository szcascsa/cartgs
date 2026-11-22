#include "include/protection_evidence_cuda.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>

namespace {
__device__ void recordTransition(bool condition, int64_t* counters, int index) {
  const auto mask = __activemask();
  const auto flags = __ballot_sync(mask, condition);
  if ((threadIdx.x & 31) == 0 && flags)
    atomicAdd(reinterpret_cast<unsigned long long*>(counters + index), static_cast<unsigned long long>(__popc(flags)));
}
__global__ void advanceClock(const int64_t* birth, const int32_t* optimized, const int32_t* unique,
    int64_t* counters, int64_t count, int64_t before, int64_t after, int a, int u, int k, int stale) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  bool warmup_under = false, warmup_mature = false, under_stale = false;
  if (i < count) {
    const bool ready = optimized[i] >= u && unique[i] >= k;
    const auto old_age = before - birth[i], age = after - birth[i];
    const bool leaving = old_age < a && age >= a;
    warmup_under = leaving && !ready && age < stale;
    warmup_mature = leaving && ready;
    under_stale = old_age >= a && old_age < stale && age >= stale && !ready;
  }
  recordTransition(warmup_under, counters, 0);
  recordTransition(warmup_mature, counters, 1);
  recordTransition(under_stale, counters, 3);
}
__global__ void commitEvidence(const int64_t* visible, const int64_t* birth, int32_t* optimized,
    int32_t* unique, int64_t* ids, int64_t* counters, int64_t count, int width,
    int64_t iteration, int64_t keyframe, int a, int u, int stale) {
  const int64_t t = blockIdx.x * blockDim.x + threadIdx.x;
  bool under_mature = false, stale_mature = false;
  if (t < count) {
    const auto i = visible[t];
    const bool ready_before = optimized[i] >= u && unique[i] >= width;
    if (optimized[i] < INT_MAX) ++optimized[i];
    if (unique[i] < width) {
      bool exists = false;
      for (int j = 0; j < unique[i]; ++j) exists |= ids[i * width + j] == keyframe;
      if (!exists) { ids[i * width + unique[i]] = keyframe; ++unique[i]; }
    }
    const bool became_ready = !ready_before && optimized[i] >= u && unique[i] >= width;
    const auto age = iteration - birth[i];
    under_mature = became_ready && age >= a && age < stale;
    stale_mature = became_ready && age >= stale;
  }
  recordTransition(under_mature, counters, 2);
  recordTransition(stale_mature, counters, 4);
}
}
void advanceProtectionClockCUDA(const torch::Tensor& birth, const torch::Tensor& optimized,
    const torch::Tensor& unique, const torch::Tensor& transitions, int64_t before, int64_t after,
    int a, int u, int k, int stale) {
  if (!birth.numel()) return;
  c10::cuda::CUDAGuard guard(birth.device());
  advanceClock<<<(birth.numel() + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      birth.data_ptr<int64_t>(), optimized.data_ptr<int32_t>(), unique.data_ptr<int32_t>(),
      transitions.data_ptr<int64_t>(), birth.numel(), before, after, a, u, k, stale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void commitProtectionEvidenceCUDA(const torch::Tensor& visible, const torch::Tensor& birth,
    const torch::Tensor& optimized, const torch::Tensor& unique, const torch::Tensor& ids,
    const torch::Tensor& transitions, int64_t iteration, int64_t keyframe, int a, int u, int stale) {
  if (!visible.numel()) return;
  c10::cuda::CUDAGuard guard(birth.device());
  commitEvidence<<<(visible.numel() + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      visible.data_ptr<int64_t>(), birth.data_ptr<int64_t>(), optimized.data_ptr<int32_t>(),
      unique.data_ptr<int32_t>(), ids.data_ptr<int64_t>(), transitions.data_ptr<int64_t>(),
      visible.numel(), ids.size(1), iteration, keyframe, a, u, stale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

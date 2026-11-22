#pragma once

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <initializer_list>
#include <stdexcept>
#include <string>

// Rasterizer kernels also use the default stream. Device barriers cover both
// streams; event intervals include host launch gaps, wall time includes host work.
class UltrafastTrainingTimer {
 public:
  explicit UltrafastTrainingTimer(bool cuda, bool enabled = true) : cuda_(cuda), enabled_(enabled) {
    if (!enabled_ || !cuda_) return;
    check(cudaEventCreate(&begin_));
    check(cudaEventCreate(&end_));
    check(cudaEventCreate(&render_begin_));
    check(cudaEventCreate(&render_end_));
  }
  ~UltrafastTrainingTimer() {
    for (auto event : {begin_, end_, render_begin_, render_end_})
      if (event) cudaEventDestroy(event);
  }
  UltrafastTrainingTimer(const UltrafastTrainingTimer&) = delete;
  UltrafastTrainingTimer& operator=(const UltrafastTrainingTimer&) = delete;
  void begin() {
    if (!enabled_) return;
    rendered_ = false;
    wall_ms = render_ms = cuda_elapsed_ms = completed_render_ms_ = 0.0;
    if (cuda_) {
      check(cudaDeviceSynchronize());
      check(cudaEventRecord(begin_, c10::cuda::getCurrentCUDAStream()));
      check(cudaEventSynchronize(begin_));
    }
    start_ = Clock::now();
  }
  void beginRender() {
    if (!enabled_) return;
    if (cuda_) {
      check(cudaDeviceSynchronize());
      if (rendered_) {
        float value = 0.0f;
        check(cudaEventElapsedTime(&value, render_begin_, render_end_));
        completed_render_ms_ += value;
      }
      check(cudaEventRecord(render_begin_, c10::cuda::getCurrentCUDAStream()));
      check(cudaEventSynchronize(render_begin_));
    } else if (rendered_) completed_render_ms_ += render_ms;
    rendered_ = true;
    render_start_ = Clock::now();
  }
  void endRender() {
    if (!enabled_) return;
    if (cuda_) {
      check(cudaDeviceSynchronize());
      check(cudaEventRecord(render_end_, c10::cuda::getCurrentCUDAStream()));
    }
    render_ms = std::chrono::duration<double, std::milli>(Clock::now() - render_start_).count();
  }
  void end() {
    if (!enabled_) return;
    if (cuda_) {
      check(cudaDeviceSynchronize());
      check(cudaEventRecord(end_, c10::cuda::getCurrentCUDAStream()));
      check(cudaEventSynchronize(end_));
      float value = 0.0f;
      check(cudaEventElapsedTime(&value, begin_, end_));
      cuda_elapsed_ms = value;
      if (rendered_) {
        check(cudaEventElapsedTime(&value, render_begin_, render_end_));
        render_ms = value;
      }
    }
    render_ms += completed_render_ms_;
    wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
  }
  double wall_ms = 0.0, render_ms = 0.0, cuda_elapsed_ms = 0.0;
 private:
  using Clock = std::chrono::steady_clock;
  static void check(cudaError_t status) {
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("Ultrafast timing: ") + cudaGetErrorString(status));
  }
  bool cuda_, enabled_, rendered_ = false;
  double completed_render_ms_ = 0.0;
  Clock::time_point start_, render_start_;
  cudaEvent_t begin_ = nullptr, end_ = nullptr, render_begin_ = nullptr, render_end_ = nullptr;
};

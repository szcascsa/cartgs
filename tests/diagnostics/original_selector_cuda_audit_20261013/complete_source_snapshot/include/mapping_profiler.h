#pragma once

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

// Host wall time covers the whole mapping cycle. Sampled CUDA event intervals
// are harvested only after completion: profiling never synchronizes CUDA.
// CaRtGS uses the default stream for its rasterizer; this is not a multi-stream
// kernel profiler. Event intervals include host launch gaps on that stream.
class MappingProfiler {
 public:
  enum Stage { Operations, FullRender, GI, FullBackward, LOD, Densify,
               Optimizer, AABB, Historical, MaturePreparation, GITeacher,
               GILoss, LodRender, LodBackward, Total, Count };
  class Scope {
   public:
    Scope(MappingProfiler& owner, Stage stage) : owner_(owner), stage_(stage) {
      owner_.startStage(stage_);
    }
    ~Scope() { owner_.stopStage(stage_); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
   private:
    MappingProfiler& owner_;
    Stage stage_;
  };
  ~MappingProfiler() {
    destroy(current_);
    for (auto& sample : pending_) destroy(sample);
    for (auto& sample : idle_) destroy(sample);
  }
  void begin(bool cuda) {
    if (active_) return;
    harvest();
    active_ = true;
    cpu_.fill(0.0);
    sample_ = cuda && !cuda_failed_ && (++cycles_ % 19 == 0) && pending_.size() < 8;
    if (sample_ && !idle_.empty()) {
      current_ = idle_.back();
      idle_.pop_back();
    }
    current_.used.fill(false);
    startStage(Total);
  }
  void cancel() { active_ = false; destroy(current_); sample_ = false; }
  Scope scope(Stage stage) { return Scope(*this, stage); }
  void end(uint64_t iteration) {
    if (!active_) return;
    stopStage(Total);
    active_ = false;
    wall_.push_back(cpu_[Total]);
    all_wall_.push_back(cpu_[Total]);
    for (int i = 0; i < Count; ++i) {
      cpu_sum_[i] += cpu_[i];
      all_cpu_sum_[i] += cpu_[i];
    }
    if (sample_) {
      pending_.push_back(current_);
      current_ = Sample{};
    }
    sample_ = false;
    harvest();
    if (wall_.size() < 1000) return;
    auto sorted = wall_;
    std::sort(sorted.begin(), sorted.end());
    std::ostringstream out;
    out << std::fixed << std::setprecision(4)
        << "[Mapping PERF] iter=" << iteration << " cycles=" << wall_.size()
        << " wall_ms(avg/p95/max)=" << cpu_sum_[Total] / wall_.size()
        << '/' << sorted[(sorted.size() * 95 + 99) / 100 - 1]
        << '/' << sorted.back() << " stage_host_ms=";
    for (int i = 0; i < Total; ++i)
      out << (i ? "," : "") << name(i) << ':' << cpu_sum_[i] / wall_.size();
    out << " sampled_cuda_elapsed_ms=";
    for (int i = 0; i < Count; ++i) {
      out << (i ? "," : "") << name(i) << ':';
      if (gpu_count_[i]) out << gpu_sum_[i] / gpu_count_[i];
      else out << "NA";
      out << "(n=" << gpu_count_[i] << ')';
    }
    std::cout << out.str() << std::endl;
    wall_.clear(); cpu_sum_.fill(0.0); gpu_sum_.fill(0.0); gpu_count_.fill(0);
  }
  void report(uint64_t iteration) {
    harvest();
    if (all_wall_.empty()) return;
    auto sorted = all_wall_;
    std::sort(sorted.begin(), sorted.end());
    std::ostringstream out;
    out << std::fixed << std::setprecision(4) << "[Mapping PERF FINAL] iteration=" << iteration
        << " cycles=" << all_wall_.size()
        << " complete_mapping_mean_ms=" << all_cpu_sum_[Total] / all_wall_.size()
        << " complete_mapping_p95_ms=" << sorted[(sorted.size() * 95 + 99) / 100 - 1];
    for (int i = 0; i < Count; ++i) {
      out << ' ' << name(i) << "_host_ms_per_step=" << all_cpu_sum_[i] / all_wall_.size()
          << ' ' << name(i) << "_cuda_samples=" << all_gpu_count_[i];
      if (all_gpu_count_[i])
        out << ' ' << name(i) << "_cuda_elapsed_mean_ms=" << all_gpu_sum_[i] / all_gpu_count_[i];
    }
    std::cout << out.str() << std::endl;
  }
 private:
  static const char* name(int stage) {
    const char* names[] = {"operations", "full_render", "gi", "full_backward",
        "lod", "densify", "optimizer", "aabb", "historical", "mature_preparation",
        "gi_teacher", "gi_loss", "lod_render", "lod_backward", "total"};
    return names[stage];
  }
  using Clock = std::chrono::steady_clock;
  struct Sample {
    std::array<cudaEvent_t, Count> begin{}, end{};
    std::array<bool, Count> used{};
  };
  static void destroy(Sample& sample) {
    for (auto& event : sample.begin) { if (event) cudaEventDestroy(event); event = nullptr; }
    for (auto& event : sample.end) { if (event) cudaEventDestroy(event); event = nullptr; }
  }
  bool checked(cudaError_t result) {
    if (result == cudaSuccess) return true;
    if (!cuda_failed_)
      std::cerr << "[Mapping PERF WARNING] CUDA sampling disabled: "
                << cudaGetErrorString(result) << std::endl;
    cuda_failed_ = true;
    return false;
  }
  void startStage(Stage stage) {
    if (!active_) return;
    starts_[stage] = Clock::now();
    if (!sample_ || cuda_failed_) return;
    if (!current_.begin[stage] && !checked(cudaEventCreate(&current_.begin[stage]))) return;
    if (!current_.end[stage] && !checked(cudaEventCreate(&current_.end[stage]))) return;
    current_.used[stage] = checked(cudaEventRecord(
        current_.begin[stage], c10::cuda::getCurrentCUDAStream()));
  }
  void stopStage(Stage stage) {
    if (!active_) return;
    cpu_[stage] += std::chrono::duration<double, std::milli>(Clock::now() - starts_[stage]).count();
    if (sample_ && !cuda_failed_ && current_.used[stage])
      checked(cudaEventRecord(current_.end[stage], c10::cuda::getCurrentCUDAStream()));
  }
  void harvest() {
    while (!pending_.empty()) {
      auto& sample = pending_.front();
      const auto status = cudaEventQuery(sample.end[Total]);
      if (status == cudaErrorNotReady) break;
      if (checked(status) && !cuda_failed_) {
        for (int i = 0; i < Count; ++i) if (sample.used[i]) {
          float ms = 0;
          if (checked(cudaEventElapsedTime(&ms, sample.begin[i], sample.end[i]))) {
            gpu_sum_[i] += ms; ++gpu_count_[i];
            all_gpu_sum_[i] += ms; ++all_gpu_count_[i];
          }
        }
      }
      if (cuda_failed_) destroy(sample);
      else idle_.push_back(sample);
      pending_.pop_front();
    }
  }
  bool active_ = false, sample_ = false, cuda_failed_ = false;
  uint64_t cycles_ = 0;
  std::array<Clock::time_point, Count> starts_{};
  std::array<double, Count> cpu_{}, cpu_sum_{}, gpu_sum_{};
  std::array<double, Count> all_cpu_sum_{}, all_gpu_sum_{};
  std::array<uint64_t, Count> gpu_count_{};
  std::array<uint64_t, Count> all_gpu_count_{};
  std::vector<double> wall_;
  std::vector<double> all_wall_;
  Sample current_;
  std::deque<Sample> pending_;
  std::vector<Sample> idle_;
};

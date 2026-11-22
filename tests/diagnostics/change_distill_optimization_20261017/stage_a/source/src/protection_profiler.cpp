#include "include/protection_profiler.h"
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <deque>
#include <iomanip>
#include <numeric>
#include <map>

struct ProtectionProfiler::Impl {
  struct Events { cudaEvent_t begin = nullptr, end = nullptr; Stage stage; uint64_t cycle = 0; };
  struct Cycle { double sum = 0; int pending = 0; bool closed = false; };
  std::array<std::vector<double>, Count> host, gpu;
  std::array<uint64_t, Count> calls{};
  std::deque<Events*> pending;
  std::vector<Events*> idle;
  std::map<uint64_t, Cycle> cycle_gpu;
  uint64_t cycle = 0;
  bool active = false;
  double cycle_host = 0;
  static bool contributes(Stage stage) { return stage < Total; }
  static void destroy(Events* e) {
    if (e->begin) cudaEventDestroy(e->begin);
    if (e->end) cudaEventDestroy(e->end);
    delete e;
  }
  void harvest() {
    while (!pending.empty()) {
      auto* e = pending.front();
      auto status = cudaEventQuery(e->end);
      if (status == cudaErrorNotReady) break;
      float ms = 0;
      if (status == cudaSuccess && cudaEventElapsedTime(&ms, e->begin, e->end) == cudaSuccess)
        gpu[e->stage].push_back(ms);
      auto& total = cycle_gpu[e->cycle];
      if (status == cudaSuccess && contributes(e->stage)) total.sum += ms;
      --total.pending;
      if (total.closed && total.pending == 0) { gpu[Total].push_back(total.sum); cycle_gpu.erase(e->cycle); }
      pending.pop_front(); idle.push_back(e);
    }
  }
  ~Impl() { for (auto* e : pending) destroy(e); for (auto* e : idle) destroy(e); }
};
ProtectionProfiler::ProtectionProfiler() : impl_(std::make_unique<Impl>()) {}
ProtectionProfiler::~ProtectionProfiler() = default;
void ProtectionProfiler::reportSelector(std::ostream& out, float ratio) {
  impl_->harvest();
  const auto& host = impl_->host[SelectorAllRows];
  const auto& gpu = impl_->gpu[SelectorAllRows];
  const auto host_total = std::accumulate(host.begin(), host.end(), 0.0);
  const auto cuda_mean = gpu.empty() ? 0.0 : std::accumulate(gpu.begin(), gpu.end(), 0.0) / gpu.size();
  out << "[Elastic SELECTOR PERF] ratio=" << ratio
      << " selector_forward_host_ms_total=" << host_total
      << " selector_forward_host_ms_mean=" << host_total / std::max<size_t>(1, host.size())
      << " selector_forward_cuda_ms_mean=" << cuda_mean
      << " selector_forward_cuda_estimated_total=" << cuda_mean * impl_->calls[SelectorAllRows]
      << " selector_forward_calls=" << impl_->calls[SelectorAllRows]
      << " selector_forward_cuda_samples=" << gpu.size() << '\n';
}
ProtectionProfiler::Scope::Scope(ProtectionProfiler* owner, Stage stage, bool cuda)
    : owner_(owner), stage_(stage), start_(std::chrono::steady_clock::now()) {
  if (!owner_) return;
  auto& p = *owner_->impl_;
  p.harvest();
  ++p.calls[stage];
  // Full is every fifth step; sample Selector on LoD and geometry on Full.
  const bool sample = stage > Total ? p.cycle % 19 == 0
      : (stage == SelectorAllRows ? p.cycle % 20 == 1 : p.cycle % 20 == 0);
  if (!p.active || !sample || !cuda || p.pending.size() >= 64) return;
  Impl::Events* e;
  if (p.idle.empty()) {
    e = new Impl::Events;
    if (cudaEventCreate(&e->begin) != cudaSuccess || cudaEventCreate(&e->end) != cudaSuccess) {
      Impl::destroy(e); return;
    }
  } else { e = p.idle.back(); p.idle.pop_back(); }
  e->stage = stage;
  e->cycle = p.cycle;
  if (cudaEventRecord(e->begin, c10::cuda::getCurrentCUDAStream()) != cudaSuccess) {
    Impl::destroy(e); return;
  }
  events_ = e;
  ++p.cycle_gpu[p.cycle].pending;
}
ProtectionProfiler::Scope::~Scope() {
  if (!owner_) return;
  auto& p = *owner_->impl_;
  if (events_) {
    auto* e = static_cast<Impl::Events*>(events_);
    if (cudaEventRecord(e->end, c10::cuda::getCurrentCUDAStream()) == cudaSuccess)
      p.pending.push_back(e);
    else { --p.cycle_gpu[e->cycle].pending; Impl::destroy(e); }
  }
  const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
  p.host[stage_].push_back(ms);
  if (p.active && Impl::contributes(stage_)) p.cycle_host += ms;
}
void ProtectionProfiler::beginIteration() {
  impl_->harvest(); impl_->active = true; ++impl_->cycle; impl_->cycle_host = 0;
}
void ProtectionProfiler::endIteration() {
  impl_->host[Total].push_back(impl_->cycle_host); impl_->active = false;
  auto found = impl_->cycle_gpu.find(impl_->cycle);
  if (found != impl_->cycle_gpu.end()) {
    found->second.closed = true;
    if (!found->second.pending) { impl_->gpu[Total].push_back(found->second.sum); impl_->cycle_gpu.erase(found); }
  }
  impl_->harvest();
}
void ProtectionProfiler::report(std::ostream& out, double metadata_mb) {
  impl_->harvest();
  const char* names[] = {"maturity_update_ms", "maturity_snapshot_ms", "selector_all_rows_ms", "maturity_total_ms",
      "tf_forward_ms", "tf_regularization_ms", "tf_backward_ms", "tf_optimizer_ms", "maturity_diagnostics_ms"};
  out << std::fixed << std::setprecision(4) << "[Protection PERF] metadata_memory_mb=" << metadata_mb;
  for (int s = 0; s < Count; ++s) for (int device = 0; device < 2; ++device) {
    auto values = device ? impl_->gpu[s] : impl_->host[s];
    if (values.empty()) continue;
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    std::sort(values.begin(), values.end());
    out << ' ' << names[s] << (device ? "_cuda" : "_host") << "(mean/p95/max)="
        << mean << '/' << values[(values.size() * 95 + 99) / 100 - 1] << '/' << values.back()
        << " n=" << values.size();
  }
  out << '\n';
  double total_host = 0, total_cuda_estimate = 0;
  out << "[Continuous TF PERF] backward_scope=cuda_tf_autograd_nodes";
  for (int s = TFForward; s <= TFOptimizer; ++s) {
    const auto& host = impl_->host[s];
    const auto& gpu = impl_->gpu[s];
    const auto host_sum = std::accumulate(host.begin(), host.end(), 0.0);
    const auto estimate = gpu.empty() ? 0.0
        : std::accumulate(gpu.begin(), gpu.end(), 0.0) / gpu.size() * impl_->calls[s];
    total_host += host_sum; total_cuda_estimate += estimate;
    out << ' ' << names[s] << "_host_total=" << host_sum << ' ' << names[s]
        << "_cuda_estimated_total=" << estimate << ' ' << names[s] << "_calls=" << impl_->calls[s]
        << ' ' << names[s] << "_cuda_samples=" << gpu.size();
  }
  out << " total_tf_ms_host=" << total_host << " total_tf_ms_cuda_estimated=" << total_cuda_estimate << '\n';
}

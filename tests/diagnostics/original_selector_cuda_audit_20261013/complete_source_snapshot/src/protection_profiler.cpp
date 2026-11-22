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
  static bool contributes(Stage stage) { return stage == Evidence || stage == Maturity || stage == Budget || stage == Admission; }
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
ProtectionProfiler::Scope::Scope(ProtectionProfiler* owner, Stage stage, bool cuda)
    : owner_(owner), stage_(stage), start_(std::chrono::steady_clock::now()) {
  if (!owner_) return;
  auto& p = *owner_->impl_;
  p.harvest();
  ++p.calls[stage];
  if (!p.active || p.cycle % 20 || !cuda || p.pending.size() >= 64) return;
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
  const char* names[] = {"protection_evidence_update_ms", "unique_kf_support_update_ms",
      "maturity_compute_ms", "protection_budget_ms", "protection_admission_ms", "protection_total_ms"};
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
}

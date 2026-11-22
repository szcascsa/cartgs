#include "include/global_gi.h"

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
size_t bytes(const torch::Tensor& tensor) {
  return tensor.defined() ? tensor.numel() * tensor.element_size() : 0;
}
// Synchronize only GI operations, never every normal Mapping iteration.
// Wall time includes execution/wait; CUDA events also report stream elapsed time.
class GITimer {
 public:
  explicit GITimer(bool cuda) : cuda_(cuda), start_(Clock::now()) {
    if (cuda_) {
      TORCH_CHECK(cudaEventCreate(&begin_) == cudaSuccess, "SGI CUDA event creation failed");
      TORCH_CHECK(cudaEventCreate(&end_) == cudaSuccess, "SGI CUDA event creation failed");
      TORCH_CHECK(cudaEventRecord(begin_, c10::cuda::getCurrentCUDAStream()) == cudaSuccess,
                  "SGI CUDA event record failed");
    }
  }
  ~GITimer() { if (begin_) cudaEventDestroy(begin_); if (end_) cudaEventDestroy(end_); }
  double finish() {
    if (cuda_) {
      TORCH_CHECK(cudaEventRecord(end_, c10::cuda::getCurrentCUDAStream()) == cudaSuccess &&
                  cudaEventSynchronize(end_) == cudaSuccess, "SGI CUDA timing failed");
      float elapsed = 0;
      TORCH_CHECK(cudaEventElapsedTime(&elapsed, begin_, end_) == cudaSuccess,
                  "SGI CUDA elapsed time failed");
      cuda_ms = elapsed;
    }
    return milliseconds(start_);
  }
  double cuda_ms = 0;
 private:
  bool cuda_;
  Clock::time_point start_;
  cudaEvent_t begin_ = nullptr, end_ = nullptr;
};
}

GlobalGIManager::GlobalGIManager(torch::DeviceType device, GlobalGIConfig config)
    : device_(device), config_(std::move(config)), bank_(config_) {}

void GlobalGIManager::resetForGaussianCount(int64_t count) {
  TORCH_CHECK(count >= 0, "Negative SGI Gaussian count");
  torch::NoGradGuard guard;
  views_.clear();
  view_bytes_ = 0;
  sum_ = torch::zeros({count}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  importance_ = torch::zeros_like(sum_);
  valid_ = torch::zeros_like(sum_, torch::kBool);
  initialized_estimate_ = false;
  valid_count_ = 0; valid_observed_count_ = -1;
  last_refresh_iteration_ = 0;
  ++version_; ++topology_version_;
  trackPeak();
}

torch::Tensor GlobalGIManager::checkedContribution(
    const torch::Tensor& raw, torch::Tensor& finite) const {
  TORCH_CHECK(raw.defined() && raw.dim() == 1 && raw.sizes() == sum_.sizes() &&
              raw.device() == sum_.device() && raw.scalar_type() == torch::kFloat32,
              "SGI contribution has stale topology, device or dtype");
  finite = torch::isfinite(raw) & (raw >= 0);
  // Non-finite rows remain unknown, never turn into supervised zero scores.
  return torch::where(finite, raw.detach(), torch::zeros_like(raw));
}

void GlobalGIManager::aggregate() {
  auto next = torch::zeros_like(sum_);
  for (const auto& view : views_) next.add_(view.second.contribution);
  auto mean = views_.empty() ? torch::zeros_like(next) : next / static_cast<double>(views_.size());
  trackPeak(bytes(next) + bytes(mean));
  sum_ = std::move(next); importance_ = std::move(mean);
  if (views_.empty()) valid_.zero_();
  valid_ = valid_ & torch::isfinite(importance_);
  ++version_;
}

void GlobalGIManager::appendGaussians(int64_t count, GITopologyKind kind) {
  TORCH_CHECK(count >= 0, "Negative SGI append count");
  if (!count) return;
  torch::NoGradGuard guard;
  const auto start = Clock::now();
  GITimer remap(config_.performance_logging && device_ == torch::kCUDA);
  // GaussianModel::densificationPostfix concatenates old then new rows.
  for (auto& view : views_) {
    auto next = torch::cat({view.second.contribution, torch::zeros({count}, sum_.options())});
    trackPeak(bytes(next));
    view_bytes_ = view_bytes_ - bytes(view.second.contribution) + bytes(next);
    view.second.contribution = std::move(next);
  }
  valid_ = torch::cat({valid_, torch::zeros({count}, valid_.options())});
  sum_ = torch::cat({sum_, torch::zeros({count}, sum_.options())});
  const double remap_ms = remap.finish();
  GITimer aggregation(config_.performance_logging && device_ == torch::kCUDA);
  aggregate();
  const double aggregate_ms = aggregation.finish();
  ++topology_version_;
  finishTopology(kind, remap_ms, aggregate_ms, remap.cuda_ms + aggregation.cuda_ms,
                 milliseconds(start));
}

void GlobalGIManager::pruneGaussians(const torch::Tensor& keep) {
  TORCH_CHECK(keep.defined() && keep.dim() == 1 && keep.sizes() == sum_.sizes() &&
              keep.device() == sum_.device() && keep.scalar_type() == torch::kBool,
              "SGI prune must use the exact GaussianModel survivor mask");
  torch::NoGradGuard guard;
  const auto start = Clock::now();
  GITimer remap(config_.performance_logging && device_ == torch::kCUDA);
  for (auto& view : views_) {
    auto next = view.second.contribution.index({keep});
    trackPeak(bytes(next));
    view_bytes_ = view_bytes_ - bytes(view.second.contribution) + bytes(next);
    view.second.contribution = std::move(next);
  }
  valid_ = valid_.index({keep});
  sum_ = sum_.index({keep});
  const double remap_ms = remap.finish();
  GITimer aggregation(config_.performance_logging && device_ == torch::kCUDA);
  aggregate();
  const double aggregate_ms = aggregation.finish();
  ++topology_version_;
  finishTopology(GITopologyKind::Prune, remap_ms, aggregate_ms,
                 remap.cuda_ms + aggregation.cuda_ms, milliseconds(start));
}

void GlobalGIManager::updateBank(std::vector<GlobalGIKeyframe> snapshot,
                                  const RenderContribution& render, double snapshot_ms) {
  GlobalGIBatch batch;
  std::set<uint64_t> present;
  for (const auto& frame : snapshot) present.insert(frame.id);
  for (auto id : bank_.keyframeIds())
    if (!present.count(id)) batch.removed.push_back(id);
  batch.upserts = std::move(snapshot);
  updateBankBatch(std::move(batch), render, snapshot_ms);
}

void GlobalGIManager::updateBankBatch(GlobalGIBatch batch,
                                  const RenderContribution& render, double snapshot_ms) {
  torch::NoGradGuard guard;
  const auto bank_start = Clock::now();
  ++bank_update_count_;
  const bool changed = bank_.updateBatch(std::move(batch));
  bank_snapshot_ms_ = snapshot_ms;
  bank_selection_ms_ = milliseconds(bank_start);
  bank_snapshot_total_ms_ += bank_snapshot_ms_;
  bank_selection_total_ms_ += bank_selection_ms_;
  bank_ms_ = bank_snapshot_ms_ + bank_selection_ms_; bank_total_ms_ += bank_ms_;
  // A reset clears caches even when the bank membership itself is unchanged.
  if (!changed && (initialized_estimate_ || bank_.members().empty())) {
    const bool cache_matches = views_.size() == bank_.members().size() &&
        std::all_of(bank_.members().begin(), bank_.members().end(),
                    [this](uint64_t id) { return views_.count(id) != 0; });
    if (cache_matches) return;
  }
  const auto start = Clock::now();
  std::set<uint64_t> members(bank_.members().begin(), bank_.members().end());
  GITimer rendering(config_.performance_logging && device_ == torch::kCUDA);
  std::map<uint64_t, GIViewCache> additions;
  auto next_valid = initialized_estimate_ ? valid_.clone() : torch::ones_like(valid_);
  for (auto id : members) {
    if (views_.count(id)) continue;
    torch::Tensor finite;
    auto contribution = checkedContribution(render(id), finite);
    next_valid = next_valid & finite;
    additions.emplace(id, GIViewCache{id, std::move(contribution)});
  }
  const double render_ms = rendering.finish();
  GITimer aggregation(config_.performance_logging && device_ == torch::kCUDA);
  auto next_sum = sum_.clone();
  for (auto it = views_.begin(); it != views_.end();) {
    if (!members.count(it->first)) {
      next_sum.sub_(it->second.contribution);
      view_bytes_ -= bytes(it->second.contribution);
      it = views_.erase(it);
    } else ++it;
  }
  size_t addition_bytes = 0;
  for (auto& view : additions) {
    next_sum.add_(view.second.contribution);
    addition_bytes += bytes(view.second.contribution);
  }
  trackPeak(addition_bytes + bytes(next_sum) + bytes(next_valid));
  incremental_count_ += additions.size();
  view_bytes_ += addition_bytes;
  views_.merge(additions);
  sum_ = std::move(next_sum);
  importance_ = views_.empty() ? torch::zeros_like(sum_) : sum_ / static_cast<double>(views_.size());
  valid_ = views_.empty() ? torch::zeros_like(valid_) : next_valid & torch::isfinite(importance_);
  if (!views_.empty()) initialized_estimate_ = true;
  ++version_;
  valid_observed_count_ = -1;
  const double aggregate_ms = aggregation.finish();
  incremental_render_ms_ += render_ms; incremental_aggregate_ms_ += aggregate_ms;
  incremental_total_ms_ += milliseconds(start);
  if (config_.performance_logging && config_.verbose_logging) {
    valid_count_ = valid_.sum().item<int64_t>();
    std::cout << "[GlobalGI] event=bank global_gi_incremental_render_ms=" << render_ms
              << " global_gi_incremental_aggregate_ms=" << aggregate_ms
              << " global_gi_incremental_cuda_ms=" << rendering.cuda_ms + aggregation.cuda_ms << '\n';
    logState("bank_state");
  }
}

bool GlobalGIManager::refreshIfDue(uint64_t iteration, const RenderContribution& render,
                                 bool force) {
  if (iteration == 0 || (!force && (iteration % config_.refresh_interval ||
      last_refresh_iteration_ == iteration))) return false;
  torch::NoGradGuard guard;
  GITimer timing(config_.performance_logging && device_ == torch::kCUDA);
  std::map<uint64_t, GIViewCache> next;
  auto next_sum = torch::zeros_like(sum_);
  auto next_valid = torch::ones_like(valid_);
  size_t next_bytes = bytes(next_sum) + bytes(next_valid);
  size_t next_view_bytes = 0;
  for (auto id : bank_.members()) {
    torch::Tensor finite;
    auto contribution = checkedContribution(render(id), finite);
    next_sum.add_(contribution); next_valid = next_valid & finite;
    next_bytes += bytes(contribution);
    next_view_bytes += bytes(contribution);
    trackPeak(next_bytes + bytes(finite));
    next.emplace(id, GIViewCache{id, std::move(contribution)});
  }
  auto mean = next.empty() ? torch::zeros_like(next_sum) : next_sum / static_cast<double>(next.size());
  if (next.empty()) next_valid.zero_();
  next_valid = next_valid & torch::isfinite(mean);
  trackPeak(next_bytes + bytes(mean));
  // Publish once, while the mapper holds the same model lock throughout rendering.
  views_.swap(next); sum_ = std::move(next_sum);
  view_bytes_ = next_view_bytes;
  importance_ = std::move(mean); valid_ = std::move(next_valid);
  initialized_estimate_ = !views_.empty();
  last_refresh_iteration_ = iteration; ++version_; ++full_count_;
  valid_observed_count_ = -1;
  const double elapsed = timing.finish();
  full_total_ms_ += elapsed; full_max_ms_ = std::max(full_max_ms_, elapsed);
  if (config_.performance_logging) {
    valid_count_ = valid_.sum().item<int64_t>();
    std::cout << "[GlobalGI] event=full_refresh iteration=" << iteration
              << " global_gi_full_refresh_views=" << views_.size()
              << " global_gi_full_refresh_ms=" << elapsed
              << " global_gi_full_refresh_cuda_ms=" << timing.cuda_ms << '\n';
    if (config_.verbose_logging) logState("refresh_state");
  }
  return true;
}

void GlobalGIManager::assertAligned(const torch::Tensor& xyz) const {
  TORCH_CHECK(sum_.defined() && importance_.defined() && valid_.defined() &&
      sum_.numel() == xyz.size(0) && importance_.numel() == xyz.size(0) &&
      valid_.numel() == xyz.size(0) && sum_.device() == xyz.device() &&
      importance_.device() == xyz.device() && valid_.device() == xyz.device(),
      "SGI aggregate is not aligned with GaussianModel");
  for (const auto& view : views_)
    TORCH_CHECK(view.second.contribution.dim() == 1 &&
        view.second.contribution.numel() == xyz.size(0) &&
        view.second.contribution.device() == xyz.device() &&
        !view.second.contribution.requires_grad(), "SGI view has stale Gaussian indices");
}

size_t GlobalGIManager::tensorBytes() const {
  return view_bytes_ + bytes(sum_) + bytes(importance_) + bytes(valid_);
}
void GlobalGIManager::trackPeak(size_t transient_bytes) {
  peak_bytes_ = std::max(peak_bytes_, tensorBytes() + transient_bytes);
}
void GlobalGIManager::recordGuidance(const torch::Tensor& observed, bool loss_enabled) {
  if (!config_.performance_logging) return;
  valid_count_ = valid_.sum().item<int64_t>();
  valid_observed_count_ = (valid_ & observed).sum().item<int64_t>();
  observed_count_ = observed.sum().item<int64_t>();
  guidance_enabled_ = loss_enabled && valid_observed_count_ > 0;
}
void GlobalGIManager::recordMapping(double wall_ms, uint64_t iteration) {
  mapping_wall_ms_ += wall_ms; ++mapping_count_; mapping_iteration_ = iteration;
  if (config_.performance_logging && (iteration == 1 || iteration % 1000 == 0))
    logState("mapping");
}
void GlobalGIManager::finishTopology(GITopologyKind kind, double remap_ms,
    double aggregate_ms, double cuda_ms, double total_ms) {
  ++topology_count_;
  const char* name = "append";
  if (kind == GITopologyKind::Clone) { ++clone_count_; name = "clone"; }
  else if (kind == GITopologyKind::Split) { ++split_count_; name = "split"; }
  else if (kind == GITopologyKind::Prune) { ++prune_count_; name = "prune"; }
  else ++append_count_;
  topology_total_ms_ += total_ms;
  topology_max_ms_ = std::max(topology_max_ms_, total_ms);
  valid_observed_count_ = -1;
  if (!config_.performance_logging) return;
  topology_times_.push_back(total_ms);
  if (!config_.verbose_logging) return;
  valid_count_ = valid_.sum().item<int64_t>();
  std::cout << "[GlobalGI] event=topology kind=" << name << " iteration=" << mapping_iteration_
            << " global_gi_topology_remap_ms=" << remap_ms
            << " global_gi_topology_aggregate_ms=" << aggregate_ms
            << " global_gi_topology_total_ms=" << total_ms
            << " global_gi_topology_cuda_ms=" << cuda_ms
            << " global_gi_topology_update_count=" << topology_count_
            << " global_gi_topology_cumulative_ms=" << topology_total_ms_
            << " gaussians=" << sum_.numel()
            << " global_gi_cache_memory_mb=" << tensorBytes() / 1048576.0
            << " global_gi_valid_gaussians=" << valid_count_
            << " global_gi_invalid_gaussians=" << sum_.numel() - valid_count_ << '\n';
}
void GlobalGIManager::logState(const char* event) const {
  auto times = (std::string(event) == "mapping" || std::string(event) == "final")
      ? topology_times_ : std::vector<double>{};
  double p95 = 0;
  if (!times.empty()) {
    const size_t index = (times.size() * 95 + 99) / 100 - 1;
    std::nth_element(times.begin(), times.begin() + index, times.end());
    p95 = times[index];
  }
  std::ostringstream out;
  out << std::fixed << std::setprecision(4) << "[GlobalGI] event=" << event
      << " iteration=" << mapping_iteration_ << " gaussians=" << sum_.numel()
      << " global_gi_total_kfs=" << bank_.totalKeyframes()
      << " global_gi_target_views=" << bank_.target()
      << " global_gi_anchor_views=" << bank_.anchors().size()
      << " global_gi_representative_views=" << bank_.representatives().size()
      << " global_gi_coverage_ratio=" << bank_.coverageRatio()
      << " global_gi_bank_update_ms=" << bank_ms_
      << " global_gi_bank_update_count=" << bank_update_count_
      << " global_gi_bank_total_ms=" << bank_total_ms_
      << " global_gi_bank_snapshot_ms=" << bank_snapshot_ms_
      << " global_gi_bank_selection_ms=" << bank_selection_ms_
      << " global_gi_bank_snapshot_total_ms=" << bank_snapshot_total_ms_
      << " global_gi_bank_selection_total_ms=" << bank_selection_total_ms_
      << " global_gi_map_wait_total_ms=" << map_wait_ms_
      << " global_gi_map_hold_total_ms=" << map_hold_ms_
        << " global_gi_association_wait_total_ms=" << association_wait_ms_
        << " global_gi_association_hold_total_ms=" << association_hold_ms_
        << " global_gi_association_writer_wait_total_ms=" << association_writer_wait_ms_
        << " tracking_gi_association_wait_total_ms=" << association_tracking_wait_ms_
        << " global_gi_snapshot_rebuilt_rows=" << snapshot_rebuilt_rows_
        << " global_gi_validated_snapshot_count=" << validated_snapshot_count_
      << " tracking_map_wait_total_ms=" << tracking_wait_ms_
      << " tracking_map_lock_count=" << tracking_lock_count_
      << " global_gi_incremental_count=" << incremental_count_
      << " global_gi_incremental_total_ms=" << incremental_total_ms_
      << " global_gi_incremental_render_total_ms=" << incremental_render_ms_
      << " global_gi_incremental_aggregate_total_ms=" << incremental_aggregate_ms_
      << " global_gi_topology_update_count=" << topology_count_
      << " global_gi_prune_count=" << prune_count_ << " global_gi_append_count=" << append_count_
      << " global_gi_clone_count=" << clone_count_ << " global_gi_split_count=" << split_count_
      << " global_gi_topology_cumulative_ms=" << topology_total_ms_
      << " global_gi_topology_total_ms_mean=" << (topology_count_ ? topology_total_ms_ / topology_count_ : 0)
      << " global_gi_topology_total_ms_max=" << topology_max_ms_
      << " global_gi_full_refresh_count=" << full_count_
      << " global_gi_full_refresh_total_ms=" << full_total_ms_
      << " global_gi_full_refresh_ms_mean=" << (full_count_ ? full_total_ms_ / full_count_ : 0)
      << " global_gi_full_refresh_ms_max=" << full_max_ms_
      << " global_gi_cache_memory_mb=" << tensorBytes() / 1048576.0
      << " global_gi_cache_peak_memory_mb=" << peak_bytes_ / 1048576.0
      << " global_gi_view_cache_estimate_mb=" << views_.size() * sum_.numel() * sizeof(float) / 1048576.0
      << " global_gi_valid_gaussians=" << valid_count_
      << " global_gi_invalid_gaussians=" << sum_.numel() - valid_count_
      << " mapping_iteration_ms_mean=" << (mapping_count_ ? mapping_wall_ms_ / mapping_count_ : 0)
      << " mapping_iterations_per_second=" << (mapping_wall_ms_ > 0 ? 1000 * mapping_count_ / mapping_wall_ms_ : 0)
      << " global_gi_overhead_fraction=" << (mapping_wall_ms_ > 0 ?
          (bank_total_ms_ + incremental_total_ms_ + topology_total_ms_ + full_total_ms_) / mapping_wall_ms_ : 0);
  if (std::string(event) == "mapping" || std::string(event) == "final")
    out << " global_gi_topology_total_ms_p95=" << p95;
  if (valid_observed_count_ >= 0)
    out << " global_gi_valid_observed_gaussians=" << valid_observed_count_
        << " global_gi_observed_gaussians=" << observed_count_
        << " global_gi_valid_observed_fraction=" << (observed_count_ ? double(valid_observed_count_) / observed_count_ : 0)
        << " global_gi_guidance_enabled=" << guidance_enabled_;
  std::cout << out.str() << std::endl;
}

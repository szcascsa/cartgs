#pragma once

#include <torch/torch.h>
#include <functional>
#include <map>
#include <vector>
#include "global_gi_bank.h"

enum class GITopologyKind { Append, Clone, Split, Prune };

struct GIViewCache {
  uint64_t keyframe_id;
  torch::Tensor contribution;
};

// All mutating calls and Selector reads run under GaussianMapper::mutex_render_.
// Append/prune use the exact GaussianModel operations, never inferred lengths.
class GlobalGIManager {
 public:
  using RenderContribution = std::function<torch::Tensor(uint64_t)>;
  GlobalGIManager(torch::DeviceType device, GlobalGIConfig config);
  void resetForGaussianCount(int64_t count);
  void appendGaussians(int64_t count, GITopologyKind kind);
  void pruneGaussians(const torch::Tensor& keep);
  void updateBank(std::vector<GlobalGIKeyframe> snapshot, const RenderContribution& render,
                  double snapshot_ms = 0);
  void updateBankBatch(GlobalGIBatch batch, const RenderContribution& render,
                       double snapshot_ms = 0);
  bool refreshIfDue(uint64_t completed_mapping_iterations, const RenderContribution& render,
                    bool force = false);
  void assertAligned(const torch::Tensor& xyz) const;
  void recordMapping(double wall_ms, uint64_t iteration);
  void recordMapLocks(double wait_ms, double hold_ms, double tracking_wait_ms,
                      uint64_t tracking_count) {
    map_wait_ms_ += wait_ms; map_hold_ms_ += hold_ms;
    tracking_wait_ms_ = tracking_wait_ms; tracking_lock_count_ = tracking_count;
  }
  void recordGuidance(const torch::Tensor& observed, bool loss_enabled);
  void recordAssociationLocks(double wait_ms, double hold_ms, uint64_t rebuilt_rows,
                              double writer_wait_ms, double tracking_wait_ms) {
    association_wait_ms_ += wait_ms; association_hold_ms_ += hold_ms;
    snapshot_rebuilt_rows_ = rebuilt_rows;
    association_writer_wait_ms_ = writer_wait_ms;
    association_tracking_wait_ms_ = tracking_wait_ms;
  }
  void recordValidatedSnapshot() { ++validated_snapshot_count_; }
  void setMappingIteration(uint64_t iteration) { mapping_iteration_ = iteration; }
  void report() const { if (config_.performance_logging) logState("final"); }
  const torch::Tensor& importance() const { return importance_; }
  const torch::Tensor& validMask() const { return valid_; }
  // Positive raw bank contribution is evidence that this row affected a GI view.
  torch::Tensor observedMask() const { return importance_ > 0; }
  const std::map<uint64_t, GIViewCache>& views() const { return views_; }
  const GlobalGIViewBank& bank() const { return bank_; }
  uint64_t version() const { return version_; }
  uint64_t topologyVersion() const { return topology_version_; }

 private:
  torch::Tensor checkedContribution(const torch::Tensor& raw, torch::Tensor& finite) const;
  void aggregate();
  size_t tensorBytes() const;
  void trackPeak(size_t transient_bytes = 0);
  void finishTopology(GITopologyKind kind, double remap_ms, double aggregate_ms,
                      double cuda_ms, double total_ms);
  void logState(const char* event) const;
  torch::DeviceType device_;
  GlobalGIConfig config_;
  GlobalGIViewBank bank_;
  std::map<uint64_t, GIViewCache> views_;
  torch::Tensor sum_, importance_, valid_;
  bool initialized_estimate_ = false;
  uint64_t version_ = 0, topology_version_ = 0, last_refresh_iteration_ = 0;
  uint64_t incremental_count_ = 0, full_count_ = 0, topology_count_ = 0;
  uint64_t bank_update_count_ = 0;
  uint64_t append_count_ = 0, clone_count_ = 0, split_count_ = 0, prune_count_ = 0;
  double bank_ms_ = 0, bank_total_ms_ = 0;
  double bank_snapshot_ms_ = 0, bank_selection_ms_ = 0;
  double bank_snapshot_total_ms_ = 0, bank_selection_total_ms_ = 0;
  double incremental_render_ms_ = 0, incremental_aggregate_ms_ = 0;
  double incremental_total_ms_ = 0, full_total_ms_ = 0, full_max_ms_ = 0;
  double topology_total_ms_ = 0, topology_max_ms_ = 0;
  std::vector<double> topology_times_;
  size_t peak_bytes_ = 0;
  size_t view_bytes_ = 0;
  int64_t valid_count_ = 0, valid_observed_count_ = -1;
  int64_t observed_count_ = 0;
  bool guidance_enabled_ = false;
  double mapping_wall_ms_ = 0;
  double map_wait_ms_ = 0, map_hold_ms_ = 0, tracking_wait_ms_ = 0;
  double association_wait_ms_ = 0, association_hold_ms_ = 0;
  double association_writer_wait_ms_ = 0, association_tracking_wait_ms_ = 0;
  uint64_t snapshot_rebuilt_rows_ = 0, validated_snapshot_count_ = 0;
  uint64_t tracking_lock_count_ = 0;
  uint64_t mapping_count_ = 0, mapping_iteration_ = 0;
};

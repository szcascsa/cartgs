#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct ConflictReplayConfig {
  int trigger_cancel_count = 1000;
  int source_view_count = 2;
  int views_per_source = 4;
  int remote_candidate_count = 32;
  int queue_capacity = 16;
  double translation_weight = 1.0, rotation_weight = 0.5;
  void validate() const;
};

// CPU-only current Twc snapshot; never stores Gaussian indices or training graphs.
struct ConflictReplayView {
  std::size_t keyframe_id = 0;
  std::array<double, 3> center{};
  std::array<double, 9> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
};

struct ConflictReplayTask {
  std::size_t keyframe_id = 0;
  std::uint64_t window_id = 0;
};

struct ConflictReplayBatch {
  std::uint64_t window_id = 0, would_cancel_count = 0;
  std::vector<std::pair<std::size_t, std::uint64_t>> sources;
  std::vector<std::vector<std::size_t>> selected_per_source;
  std::vector<std::pair<double, double>> distance_ranges;
  std::vector<std::size_t> enqueued;
  std::size_t duplicates = 0, queue_full = 0;
};

class ConflictReplayManager {
 public:
  explicit ConflictReplayManager(ConflictReplayConfig config = {});
  bool recordCancellation(std::size_t source_keyframe_id);
  ConflictReplayBatch enqueue(const std::vector<ConflictReplayView>& views,
                             const std::vector<std::size_t>& excluded);
  std::optional<ConflictReplayTask> pop();
  void reset();
  std::size_t queueSize() const { return queue_.size(); }
  std::uint64_t pendingCancellationCount() const { return window_count_; }
 private:
  ConflictReplayConfig config_;
  std::uint64_t window_count_ = 0, window_id_ = 0;
  std::unordered_map<std::size_t, std::uint64_t> cancel_counts_;
  std::deque<ConflictReplayTask> queue_;
  std::unordered_set<std::size_t> queued_;
};

#include "include/conflict_replay.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
bool finite(const ConflictReplayView& view) {
  for (double x : view.center) if (!std::isfinite(x)) return false;
  for (double x : view.rotation) if (!std::isfinite(x)) return false;
  return true;
}
double distance(const ConflictReplayView& a, const ConflictReplayView& b,
                double scale, const ConflictReplayConfig& config) {
  double translation = 0.0, trace = 0.0;
  for (std::size_t i = 0; i < 3; ++i)
    translation += (a.center[i] - b.center[i]) * (a.center[i] - b.center[i]);
  for (std::size_t i = 0; i < 9; ++i) trace += a.rotation[i] * b.rotation[i];
  return config.translation_weight * std::sqrt(translation) / scale +
      config.rotation_weight * std::acos(std::clamp((trace - 1.0) * 0.5, -1.0, 1.0)) /
          3.14159265358979323846;
}
}

void ConflictReplayConfig::validate() const {
  if (trigger_cancel_count <= 0 || source_view_count <= 0 || views_per_source <= 0 ||
      remote_candidate_count < views_per_source || queue_capacity <= 0 ||
      !std::isfinite(translation_weight) || !std::isfinite(rotation_weight) ||
      translation_weight < 0.0 || rotation_weight < 0.0 ||
      translation_weight + rotation_weight <= 0.0)
    throw std::invalid_argument("Invalid ConflictReplay configuration");
}

ConflictReplayManager::ConflictReplayManager(ConflictReplayConfig config) : config_(config) {
  config_.validate();
}

bool ConflictReplayManager::recordCancellation(std::size_t source_keyframe_id) {
  ++cancel_counts_[source_keyframe_id];
  return ++window_count_ >= static_cast<std::uint64_t>(config_.trigger_cancel_count);
}

ConflictReplayBatch ConflictReplayManager::enqueue(
    const std::vector<ConflictReplayView>& views, const std::vector<std::size_t>& excluded) {
  ConflictReplayBatch batch;
  if (window_count_ < static_cast<std::uint64_t>(config_.trigger_cancel_count)) return batch;
  batch.window_id = ++window_id_; batch.would_cancel_count = window_count_;
  std::unordered_map<std::size_t, const ConflictReplayView*> valid;
  for (const auto& view : views) if (finite(view)) valid.emplace(view.keyframe_id, &view);
  for (const auto& entry : cancel_counts_) if (valid.count(entry.first)) batch.sources.push_back(entry);
  std::sort(batch.sources.begin(), batch.sources.end(), [](const auto& a, const auto& b) {
    return a.second != b.second ? a.second > b.second : a.first < b.first;
  });
  if (batch.sources.size() > static_cast<std::size_t>(config_.source_view_count))
    batch.sources.resize(config_.source_view_count);
  cancel_counts_.clear(); window_count_ = 0;

  std::unordered_set<std::size_t> blocked(excluded.begin(), excluded.end());
  blocked.insert(queued_.begin(), queued_.end());
  for (const auto& source : batch.sources) blocked.insert(source.first);
  std::array<double, 3> low{}, high{};
  bool first = true;
  for (const auto& entry : valid) {
    const auto& center = entry.second->center;
    if (first) { low = high = center; first = false; }
    for (std::size_t axis = 0; axis < 3; ++axis) {
      low[axis] = std::min(low[axis], center[axis]);
      high[axis] = std::max(high[axis], center[axis]);
    }
  }
  double scale = 0.0;
  for (std::size_t axis = 0; axis < 3; ++axis) scale += (high[axis] - low[axis]) * (high[axis] - low[axis]);
  scale = std::max(std::sqrt(scale), 1e-6);

  for (const auto& source : batch.sources) {
    std::vector<std::pair<double, std::size_t>> remote;
    for (const auto& entry : valid) if (!blocked.count(entry.first))
      remote.emplace_back(distance(*valid.at(source.first), *entry.second, scale, config_), entry.first);
    const auto farther = [](const auto& a, const auto& b) {
      return a.first != b.first ? a.first > b.first : a.second < b.second;
    };
    const auto remote_count = std::min(remote.size(), static_cast<std::size_t>(config_.remote_candidate_count));
    std::partial_sort(remote.begin(), remote.begin() + remote_count, remote.end(), farther);
    remote.resize(remote_count);
    std::vector<std::size_t> selected;
    std::vector<double> nearest(remote.size(), std::numeric_limits<double>::infinity());
    std::vector<bool> used(remote.size(), false);
    const auto count = std::min(remote.size(), static_cast<std::size_t>(config_.views_per_source));
    double minimum = std::numeric_limits<double>::infinity(), maximum = 0.0;
    for (std::size_t pick = 0; pick < count; ++pick) {
      std::size_t best = 0;
      if (pick > 0) {
        double best_distance = -1.0;
        for (std::size_t i = 0; i < remote.size(); ++i) if (!used[i] &&
            (nearest[i] > best_distance || (nearest[i] == best_distance && remote[i].second < remote[best].second))) {
          best = i; best_distance = nearest[i];
        }
      }
      used[best] = true; selected.push_back(remote[best].second);
      minimum = std::min(minimum, remote[best].first); maximum = std::max(maximum, remote[best].first);
      for (std::size_t i = 0; i < remote.size(); ++i) if (!used[i])
        nearest[i] = std::min(nearest[i], distance(*valid.at(remote[i].second),
                                                    *valid.at(remote[best].second), scale, config_));
    }
    batch.selected_per_source.push_back(std::move(selected));
    batch.distance_ranges.emplace_back(count ? minimum : 0.0, maximum);
  }
  // Interleave independent source selections, then deduplicate without refilling.
  std::unordered_set<std::size_t> batch_ids;
  for (int pick = 0; pick < config_.views_per_source; ++pick)
    for (const auto& selected : batch.selected_per_source) if (static_cast<std::size_t>(pick) < selected.size()) {
      const auto id = selected[pick];
      if (!batch_ids.insert(id).second) { ++batch.duplicates; continue; }
      if (queue_.size() >= static_cast<std::size_t>(config_.queue_capacity)) { ++batch.queue_full; continue; }
      queue_.push_back({id, batch.window_id}); queued_.insert(id); batch.enqueued.push_back(id);
    }
  return batch;
}

std::optional<ConflictReplayTask> ConflictReplayManager::pop() {
  if (queue_.empty()) return std::nullopt;
  const auto task = queue_.front(); queue_.pop_front(); queued_.erase(task.keyframe_id);
  return task;
}

void ConflictReplayManager::reset() {
  window_count_ = 0; cancel_counts_.clear(); queue_.clear(); queued_.clear(); ++window_id_;
}

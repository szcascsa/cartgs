#pragma once

#include "include/global_gi_bank.h"

// Frozen pre-optimization policy for differential tests and CPU benchmarks.
// CPU-only bank policy. ORB pointers never outlive the mapper's snapshot lock.
class ReferenceGlobalGIViewBank {
 public:
  explicit ReferenceGlobalGIViewBank(GlobalGIConfig config) : config_(config) {
    config_.validate();
  }
  bool update(std::vector<GlobalGIKeyframe> snapshot) {
    std::map<uint64_t, std::vector<uint64_t>> next;
    std::set<uint64_t> changed;
    for (auto& frame : snapshot) {
      auto& points = frame.mappoints;
      std::sort(points.begin(), points.end());
      points.erase(std::unique(points.begin(), points.end()), points.end());
      const auto old = pool_.find(frame.id);
      if (old == pool_.end() || old->second != points) changed.insert(frame.id);
      next.emplace(frame.id, std::move(points));
    }
    bool anchor_changed = false;
    bool removed = false;
    for (const auto& entry : pool_) {
      if (!next.count(entry.first)) removed = true;
      if (!next.count(entry.first) || changed.count(entry.first)) {
        for (auto point : entry.second) {
          if (--all_points_.at(point) == 0) all_points_.erase(point);
          if (anchors_.count(entry.first) && --coverage_.at(point) == 0) coverage_.erase(point);
        }
      }
    }
    if (changed.empty() && !removed) return false;
    for (auto id : changed)
      for (auto point : next.at(id)) {
        ++all_points_[point];
        if (anchors_.count(id)) ++coverage_[point];
      }
    for (auto it = anchors_.begin(); it != anchors_.end();) {
      if (!next.count(*it)) { it = anchors_.erase(it); anchor_changed = true; }
      else { anchor_changed |= changed.count(*it) != 0; ++it; }
    }
    for (const auto& entry : pool_)
      if (!next.count(entry.first)) priorities_.erase({priority(entry.first), entry.first});
    for (const auto& entry : next)
      if (!pool_.count(entry.first)) priorities_.insert({priority(entry.first), entry.first});
    pool_ = std::move(next);
    target_ = pool_.empty() ? 0 : std::min(pool_.size(), std::max<size_t>(1,
        static_cast<size_t>(std::ceil(config_.view_ratio * pool_.size() - 1e-12))));
    if (config_.max_views) target_ = std::min(target_, config_.max_views);
    anchor_target_ = static_cast<size_t>(std::ceil(
        config_.coverage_anchor_fraction * target_ - 1e-12));
    while (anchors_.size() > anchor_target_) {
      auto worst = anchors_.begin();
      for (auto it = anchors_.begin(); it != anchors_.end(); ++it)
        if (uniqueGain(*it) < uniqueGain(*worst)) worst = it;
      removeAnchor(*worst); anchor_changed = true;
    }
    while (anchors_.size() < anchor_target_) {
      uint64_t best = 0; size_t best_gain = 0; bool found = false;
      for (const auto& entry : pool_) {
        if (anchors_.count(entry.first)) continue;
        size_t gain = 0;
        for (auto point : entry.second) gain += coverage_.count(point) == 0;
        if (!found || gain > best_gain) { best = entry.first; best_gain = gain; found = true; }
      }
      if (!found) break;
      addAnchor(best);
    }
    // Association changes in an anchor can change the value of old candidates.
    // New/changed candidates otherwise suffice for streaming replacement.
    for (const auto& entry : pool_) {
      if (anchors_.empty() || anchors_.count(entry.first) ||
          (!anchor_changed && !changed.count(entry.first))) continue;
      uint64_t replace = 0; int64_t best_delta = 0;
      for (auto anchor : anchors_) {
        int64_t delta = -static_cast<int64_t>(uniqueGain(anchor));
        const auto& removed = pool_.at(anchor);
        for (auto point : entry.second) {
          const auto count = coverage_.find(point);
          if (count == coverage_.end() || (count->second == 1 &&
              std::binary_search(removed.begin(), removed.end(), point))) ++delta;
        }
        if (delta > best_delta) { replace = anchor; best_delta = delta; }
      }
      if (best_delta > 0) { removeAnchor(replace); addAnchor(entry.first); }
    }
    representatives_.clear();
    for (const auto& candidate : priorities_) {
      if (representatives_.size() == target_ - anchor_target_) break;
      if (!anchors_.count(candidate.second)) representatives_.insert(candidate.second);
    }
    std::vector<uint64_t> members(anchors_.begin(), anchors_.end());
    members.insert(members.end(), representatives_.begin(), representatives_.end());
    std::sort(members.begin(), members.end());
    const bool membership_changed = members != members_;
    members_ = std::move(members);
    coverage_ratio_ = all_points_.empty() ? 0.0 : double(coverage_.size()) / all_points_.size();
    return membership_changed;
  }
  const std::vector<uint64_t>& members() const { return members_; }
  const std::set<uint64_t>& anchors() const { return anchors_; }
  const std::set<uint64_t>& representatives() const { return representatives_; }
  size_t totalKeyframes() const { return pool_.size(); }
  size_t target() const { return target_; }
  double coverageRatio() const { return coverage_ratio_; }
  uint64_t priority(uint64_t id) const {
    uint64_t z = (id ^ config_.sampling_seed) + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
 private:
  size_t uniqueGain(uint64_t anchor) const {
    size_t result = 0;
    for (auto point : pool_.at(anchor)) result += coverage_.at(point) == 1;
    return result;
  }
  void removeAnchor(uint64_t id) {
    for (auto point : pool_.at(id)) if (--coverage_.at(point) == 0) coverage_.erase(point);
    anchors_.erase(id);
  }
  void addAnchor(uint64_t id) {
    anchors_.insert(id);
    for (auto point : pool_.at(id)) ++coverage_[point];
  }
  GlobalGIConfig config_;
  std::map<uint64_t, std::vector<uint64_t>> pool_;
  std::set<uint64_t> anchors_, representatives_;
  std::set<std::pair<uint64_t, uint64_t>> priorities_;
  std::unordered_map<uint64_t, size_t> coverage_;
  std::unordered_map<uint64_t, size_t> all_points_;
  std::vector<uint64_t> members_;
  size_t target_ = 0, anchor_target_ = 0;
  double coverage_ratio_ = 0;
};

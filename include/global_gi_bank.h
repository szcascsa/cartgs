#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct GlobalGIConfig {
  // One source selector avoids competing RecentWindow/GlobalGI enable flags.
  std::string mode = "disabled";
  double view_ratio = 0.30;
  double coverage_anchor_fraction = 0.80;
  uint64_t sampling_seed = 0;
  uint64_t refresh_interval = 1000;
  size_t max_views = 0;
  bool performance_logging = true;
  bool verbose_logging = false;
  bool marginal_shadow = false;
  bool marginal_diagnostics = false;
  bool hierarchical_representative_enabled = false;
  bool hierarchical_representative_diagnostics = false;

  bool streaming() const { return mode == "streaming" || mode == "streaming_marginal_geo"; }
  void validate() const {
    if (mode != "disabled" && mode != "recent" && !streaming())
      throw std::invalid_argument("GlobalGI.mode must be disabled, recent, streaming or streaming_marginal_geo");
    if (marginal_shadow && mode != "streaming")
      throw std::invalid_argument("Marginal shadow requires Raw streaming teacher");
    if (hierarchical_representative_enabled && (mode != "streaming" || marginal_shadow))
      throw std::invalid_argument("Hierarchical representatives require Raw streaming without marginal shadow");
    if (!std::isfinite(view_ratio) || view_ratio <= 0 || view_ratio > 1 ||
        !std::isfinite(coverage_anchor_fraction) ||
        coverage_anchor_fraction < 0 || coverage_anchor_fraction > 1 ||
        refresh_interval == 0)
      throw std::invalid_argument("Invalid GlobalGI configuration");
  }
};

struct GlobalGIKeyframe {
  uint64_t id = 0;
  std::vector<uint64_t> mappoints;
};

struct GlobalGIBatch {
  std::vector<GlobalGIKeyframe> upserts;
  std::vector<uint64_t> removed;
};

// Selection runs once after a complete boundary batch, preserving full-input
// ordering and replacement semantics without copying the unchanged rows.
class GlobalGIViewBank {
 public:
  explicit GlobalGIViewBank(GlobalGIConfig config) : config_(config) {
    config_.validate();
  }
  bool update(std::vector<GlobalGIKeyframe> snapshot) {
    GlobalGIBatch batch;
    std::set<uint64_t> present;
    for (const auto& frame : snapshot) present.insert(frame.id);
    for (const auto& entry : pool_)
      if (!present.count(entry.first)) batch.removed.push_back(entry.first);
    batch.upserts = std::move(snapshot);
    return updateBatch(std::move(batch));
  }
  bool updateBatch(GlobalGIBatch batch) {
    std::map<uint64_t, std::vector<uint64_t>> updates;
    std::set<uint64_t> changed, removed, upsert_ids;
    for (auto& frame : batch.upserts) {
      upsert_ids.insert(frame.id);
      auto& points = frame.mappoints;
      std::sort(points.begin(), points.end());
      points.erase(std::unique(points.begin(), points.end()), points.end());
      const auto old = pool_.find(frame.id);
      if (old == pool_.end() || old->second != points) {
        changed.insert(frame.id);
        updates.emplace(frame.id, std::move(points));
      }
    }
    for (auto id : batch.removed)
      if (pool_.count(id) && !upsert_ids.count(id)) removed.insert(id);
    if (changed.empty() && removed.empty()) return false;
    bool anchor_changed = false;
    std::set<uint64_t> affected = changed;
    affected.insert(removed.begin(), removed.end());
    for (auto id : affected) {
      const auto old = pool_.find(id);
      if (old != pool_.end()) {
        for (auto point : old->second) {
          if (--all_points_.at(point) == 0) all_points_.erase(point);
          if (anchors_.count(id)) removeCoverage(point, id);
        }
      }
    }
    for (auto id : changed)
      for (auto point : updates.at(id)) {
        ++all_points_[point];
        if (anchors_.count(id)) addCoverage(point, id);
      }
    for (auto it = anchors_.begin(); it != anchors_.end();) {
      if (removed.count(*it)) {
        unique_gains_.erase(*it); it = anchors_.erase(it); anchor_changed = true;
      }
      else { anchor_changed |= changed.count(*it) != 0; ++it; }
    }
    for (auto id : removed) {
      priorities_.erase({priority(id), id});
      pool_.erase(id);
    }
    for (auto& entry : updates) {
      if (!pool_.count(entry.first)) priorities_.insert({priority(entry.first), entry.first});
      pool_[entry.first] = std::move(entry.second);
    }
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
    std::unordered_map<uint64_t, size_t> retained_unique;
    retained_unique.reserve(anchors_.size());
    for (const auto& entry : pool_) {
      if (anchors_.empty() || anchors_.count(entry.first) ||
          (!anchor_changed && !changed.count(entry.first))) continue;
      size_t uncovered = 0;
      retained_unique.clear();
      for (auto point : entry.second) {
        const auto coverage = coverage_.find(point);
        if (coverage == coverage_.end()) ++uncovered;
        else if (coverage->second.count == 1) ++retained_unique[coverage->second.owners_xor];
      }
      uint64_t replace = 0; int64_t best_delta = 0;
      for (auto anchor : anchors_) {
        const auto retained = retained_unique.find(anchor);
        // Exact gain: new coverage + retained unique coverage - removed unique coverage.
        const int64_t delta = static_cast<int64_t>(uncovered) +
            (retained == retained_unique.end() ? 0 : static_cast<int64_t>(retained->second)) -
            static_cast<int64_t>(uniqueGain(anchor));
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
  std::vector<uint64_t> keyframeIds() const {
    std::vector<uint64_t> ids;
    for (const auto& entry : pool_) ids.push_back(entry.first);
    return ids;
  }
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
  struct Coverage {
    size_t count = 0;
    uint64_t owners_xor = 0;
  };
  size_t uniqueGain(uint64_t anchor) const { return unique_gains_.at(anchor); }
  void addCoverage(uint64_t point, uint64_t anchor) {
    auto& coverage = coverage_[point];
    if (coverage.count == 0) ++unique_gains_.at(anchor);
    else if (coverage.count == 1) --unique_gains_.at(coverage.owners_xor);
    ++coverage.count;
    coverage.owners_xor ^= anchor;
  }
  void removeCoverage(uint64_t point, uint64_t anchor) {
    auto& coverage = coverage_.at(point);
    if (coverage.count == 1) --unique_gains_.at(coverage.owners_xor);
    --coverage.count;
    coverage.owners_xor ^= anchor;
    if (coverage.count == 0) coverage_.erase(point);
    else if (coverage.count == 1) ++unique_gains_.at(coverage.owners_xor);
  }
  void removeAnchor(uint64_t id) {
    for (auto point : pool_.at(id)) removeCoverage(point, id);
    anchors_.erase(id);
    unique_gains_.erase(id);
  }
  void addAnchor(uint64_t id) {
    anchors_.insert(id);
    unique_gains_.emplace(id, 0);
    for (auto point : pool_.at(id)) addCoverage(point, id);
  }
  GlobalGIConfig config_;
  std::map<uint64_t, std::vector<uint64_t>> pool_;
  std::set<uint64_t> anchors_, representatives_;
  std::set<std::pair<uint64_t, uint64_t>> priorities_;
  std::unordered_map<uint64_t, Coverage> coverage_;
  std::unordered_map<uint64_t, size_t> unique_gains_;
  std::unordered_map<uint64_t, size_t> all_points_;
  std::vector<uint64_t> members_;
  size_t target_ = 0, anchor_target_ = 0;
  double coverage_ratio_ = 0;
};

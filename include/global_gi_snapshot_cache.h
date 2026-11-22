#pragma once

#include "global_gi_bank.h"

// Metadata and row reads belong to one ORB GI read transaction. The cache owns
// only IDs, never ORB pointers. Every boundary is applied as one bank batch.
class GlobalGISnapshotCache {
 public:
  using Metadata = std::pair<uint64_t, uint64_t>; // KF ID, association revision
  template<class ReadPoints>
  GlobalGIBatch update(uint64_t generation, uint64_t bad_revision,
                      const std::vector<Metadata>& frames, ReadPoints read_points) {
    GlobalGIBatch batch;
    if (generation != generation_) {
      for (const auto& row : rows_) batch.removed.push_back(row.first);
      rows_.clear();
      generation_ = generation;
    }
    std::set<uint64_t> present;
    for (const auto& frame : frames) {
      present.insert(frame.first);
      auto old = rows_.find(frame.first);
      if (old != rows_.end() && old->second.revision == frame.second &&
          old->second.bad_revision == bad_revision) continue;
      auto ids = read_points(frame.first);
      std::sort(ids.begin(), ids.end());
      ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      if (old == rows_.end() || old->second.points != ids)
        batch.upserts.push_back({frame.first, ids});
      rows_[frame.first] = {frame.second, bad_revision, std::move(ids)};
      ++rebuilt_rows_;
    }
    for (auto it = rows_.begin(); it != rows_.end();) {
      if (!present.count(it->first)) {
        batch.removed.push_back(it->first);
        it = rows_.erase(it);
      } else ++it;
    }
    return batch;
  }
  std::vector<GlobalGIKeyframe> snapshot() const {
    std::vector<GlobalGIKeyframe> result;
    for (const auto& row : rows_) result.push_back({row.first, row.second.points});
    return result;
  }
  uint64_t rebuiltRows() const { return rebuilt_rows_; }
 private:
  struct Row {
    uint64_t revision, bad_revision;
    std::vector<uint64_t> points;
  };
  uint64_t generation_ = 0, rebuilt_rows_ = 0;
  std::map<uint64_t, Row> rows_;
};

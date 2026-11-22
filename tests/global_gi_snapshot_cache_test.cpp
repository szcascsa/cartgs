#include "include/global_gi_snapshot_cache.h"
#include "global_gi_bank_reference.h"
#include <iostream>
#include <random>

int main() {
  GlobalGISnapshotCache cache;
  GlobalGIViewBank bank(GlobalGIConfig{});
  ReferenceGlobalGIViewBank reference(GlobalGIConfig{});
  std::map<uint64_t, std::vector<uint64_t>> points;
  std::map<uint64_t, uint64_t> revisions;
  uint64_t generation = 1, bad_revision = 0, reads = 0;
  std::mt19937 rng(271);
  for (int boundary = 0; boundary < 600; ++boundary) {
    if (boundary == 300) { ++generation; points.clear(); revisions.clear(); }
    // Several changes form one boundary, including deletions and replacement.
    if (boundary % 3 != 2) {
      for (int event = 0; event < 5; ++event) {
        const auto id = rng() % 50;
        if (rng() % 5 == 0) { points.erase(id); ++revisions[id]; }
        else {
          points[id] = {rng() % 100, rng() % 100, rng() % 100};
          ++revisions[id];
        }
      }
      if (boundary % 7 == 0) {
        ++bad_revision;
        // Invalid MP may be absent from its own observations: invalidate all rows.
        for (auto& row : points)
          row.second.erase(std::remove(row.second.begin(), row.second.end(), 42), row.second.end());
      }
    }
    std::vector<GlobalGISnapshotCache::Metadata> metadata;
    std::vector<GlobalGIKeyframe> authoritative;
    for (const auto& row : points) {
      metadata.emplace_back(row.first, revisions.at(row.first));
      authoritative.push_back({row.first, row.second});
    }
    const auto previous_reads = reads;
    auto batch = cache.update(generation, bad_revision, metadata, [&](uint64_t id) {
      ++reads; return points.at(id);
    });
    bank.updateBatch(std::move(batch));
    reference.update(authoritative);
    if (bank.members() != reference.members() || bank.anchors() != reference.anchors() ||
        bank.representatives() != reference.representatives() ||
        bank.coverageRatio() != reference.coverageRatio() ||
        bank.totalKeyframes() != reference.totalKeyframes())
      throw std::runtime_error("Batch cache differs at boundary " + std::to_string(boundary));
    if (boundary % 3 == 2 && reads != previous_reads)
      throw std::runtime_error("Unchanged boundary rebuilt association rows");
  }
  std::cout << "PASS: initial/no-change/new/changed/bad/replaced/removed/batched/reset bank state\n";
}

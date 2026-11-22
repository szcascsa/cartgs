#include "include/conflict_replay.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
ConflictReplayView view(std::size_t id, double x, double y = 0) {
  ConflictReplayView result; result.keyframe_id = id; result.center = {x, y, 0}; return result;
}
void thresholdAndIndependentSources() {
  ConflictReplayConfig config; config.remote_candidate_count = 4;
  ConflictReplayManager manager(config);
  for (int i = 0; i < 999; ++i) check(!manager.recordCancellation(i < 600 ? 1 : 2), "Early trigger");
  check(manager.recordCancellation(2), "Missing 1000th trigger");
  std::vector<ConflictReplayView> views{view(1, 0), view(2, 100)};
  for (std::size_t i = 3; i < 15; ++i) views.push_back(view(i, (i - 3) * 9));
  const auto batch = manager.enqueue(views, {});
  check(batch.sources.size() == 2 && batch.sources[0].first == 1 && batch.sources[0].second == 600,
        "Sources are not ranked by current-KF cancellation count");
  check(batch.selected_per_source[0].size() == 4 && batch.selected_per_source[1].size() == 4,
        "Did not independently select four per source");
  for (auto id : batch.selected_per_source[0]) check(id >= 11, "Source 1 did not use its farthest pool");
  for (auto id : batch.selected_per_source[1]) check(id <= 6, "Source 2 did not use its farthest pool");
  check(manager.queueSize() == 8 && manager.pendingCancellationCount() == 0, "Window/queue mismatch");
  check(manager.enqueue(views, {}).enqueued.empty(), "Window retriggered without new cancellations");
}
void diversityDedupAndCapacity() {
  ConflictReplayConfig config; config.trigger_cancel_count = 2;
  config.views_per_source = 2; config.remote_candidate_count = 4; config.queue_capacity = 2;
  ConflictReplayManager manager(config);
  manager.recordCancellation(1); manager.recordCancellation(2);
  const auto batch = manager.enqueue({view(1, 0), view(2, 0), view(3, 100), view(4, 99),
      view(5, -90), view(6, 20), view(7, 1000)}, {7});
  check(batch.selected_per_source[0] == std::vector<std::size_t>({3, 5}), "FPS did not disperse far views");
  check(batch.enqueued.size() == 2 && batch.duplicates == 2 && manager.queueSize() == 2,
        "Independent selections were refilled instead of deduplicated");
  check(manager.pop()->keyframe_id == 3 && manager.pop()->keyframe_id == 5 && !manager.pop(), "FIFO failed");
  manager.recordCancellation(1); manager.recordCancellation(2);
  auto invalid = view(8, std::numeric_limits<double>::quiet_NaN());
  const auto empty = manager.enqueue({invalid}, {});
  check(empty.enqueued.empty() && manager.pendingCancellationCount() == 0, "Invalid window was not cleared");
  manager.recordCancellation(1); manager.reset();
  check(manager.queueSize() == 0 && manager.pendingCancellationCount() == 0, "Reset retained stale work");
  config.queue_capacity = 1;
  ConflictReplayManager capped(config); capped.recordCancellation(1); capped.recordCancellation(2);
  const auto full = capped.enqueue({view(1, 0), view(2, 0), view(3, 100), view(4, -100)}, {});
  check(capped.queueSize() == 1 && full.queue_full == 1, "Queue exceeded capacity or hid overflow");
}
void queuedViewsAndTieOrder() {
  ConflictReplayConfig config; config.trigger_cancel_count = 2;
  config.views_per_source = 1; config.remote_candidate_count = 1;
  ConflictReplayManager manager(config);
  manager.recordCancellation(2); manager.recordCancellation(1);
  const std::vector<ConflictReplayView> views{view(1, 0), view(2, 0), view(3, 100), view(4, -99)};
  auto batch = manager.enqueue(views, {});
  check(batch.sources[0].first == 1 && batch.sources[1].first == 2 &&
            batch.enqueued == std::vector<std::size_t>({3}), "Ties or independent dedup are not deterministic");
  manager.recordCancellation(1); manager.recordCancellation(2);
  batch = manager.enqueue(views, {});
  check(batch.enqueued == std::vector<std::size_t>({4}) && manager.queueSize() == 2,
        "Already queued view was selected again");
  check(manager.pop()->window_id == 1 && manager.pop()->window_id == 2, "FIFO lost window provenance");
  manager.recordCancellation(1); manager.recordCancellation(2);
  batch = manager.enqueue(views, {3, 4});
  check(batch.enqueued.empty() && batch.selected_per_source[0].empty() &&
            manager.pendingCancellationCount() == 0, "Ineligible window retained or randomly refilled tasks");
}
void invalidConfiguration() {
  for (int scenario = 0; scenario < 4; ++scenario) {
    ConflictReplayConfig config;
    if (scenario == 0) config.trigger_cancel_count = 0;
    if (scenario == 1) config.remote_candidate_count = config.views_per_source - 1;
    if (scenario == 2) config.queue_capacity = 0;
    if (scenario == 3) config.rotation_weight = std::numeric_limits<double>::quiet_NaN();
    bool rejected = false;
    try { ConflictReplayManager invalid(config); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid Conflict Replay configuration was accepted");
  }
}
}
int main() { thresholdAndIndependentSources(); diversityDedupAndCapacity(); queuedViewsAndTieOrder(); invalidConfiguration(); }

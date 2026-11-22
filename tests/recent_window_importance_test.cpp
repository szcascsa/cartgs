#include "include/recent_window_importance.h"
#include "include/gaussian_keyframe.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void compareRebuild(RecentWindowImportanceManager& manager, double beta) {
  auto expected = torch::zeros({manager.size()});
  double denominator = 0;
  const auto entries = manager.entries();
  auto observed = torch::zeros({manager.size()}, torch::kBool);
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto weight = std::pow(beta, entries.size() - 1 - i);
    expected += entries[i].importance * weight;
    denominator += weight;
    if (weight > 0) observed = observed | entries[i].observed;
  }
  if (denominator > 0) expected /= denominator;
  check(torch::allclose(manager.getImportance(), expected, 1e-5, 2e-5),
        "Incremental RecentWindow differs from weighted full rebuild");
  check(manager.getImportance().unsafeGetTensorImpl() ==
            manager.getImportance().unsafeGetTensorImpl(), "GI cache not reused");
  check(manager.observedMask().equal(observed), "Recent observation union is stale");
}
void rollover(int64_t size, float beta) {
  RecentWindowConfig config;
  config.enabled = true;
  config.size = size; config.ema_decay = beta;
  RecentWindowImportanceManager manager(torch::kCPU, config);
  manager.resetForGaussianCount(3);
  auto counts = torch::ones({3}, torch::TensorOptions().dtype(torch::kInt32));
  for (size_t id = 1; id <= 600; ++id) {
    manager.pushKeyframe(std::make_shared<GaussianKeyframe>(id), 3);
    compareRebuild(manager, beta);
    auto previous = manager.getImportance();
    manager.updateContribution(id, torch::tensor({1.0f, float(id % 7 + 1), 3.0f}), counts);
    check(previous.unsafeGetTensorImpl() != manager.getImportance().unsafeGetTensorImpl(),
          "Contribution update retained stale normalized GI");
    compareRebuild(manager, beta);
    if (id % 13 == 0) {
      const auto older = manager.entries().front().keyframe_id;
      manager.updateContribution(older, torch::zeros({3}), counts);
      compareRebuild(manager, beta);
      manager.updateContribution(older, torch::tensor({3.0f, 2.0f, 1.0f}), counts);
      compareRebuild(manager, beta);
    }
  }
  manager.appendGaussians(1);
  compareRebuild(manager, beta);
  manager.pruneGaussians(torch::tensor({true, false, true, true}));
  compareRebuild(manager, beta);
  manager.invalidateContributions();
  check(!manager.hasValidContribution() && manager.getImportance().equal(torch::zeros({3})),
        "Invalidation retained a contribution");
  const auto last = manager.entries().back().keyframe_id;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  manager.updateContribution(last, torch::tensor({nan, 2.0f, 4.0f}),
                            torch::tensor({0, 1, 1}, counts.options()));
  check(torch::allclose(manager.entries().back().importance,
                       torch::tensor({0.0f, 2.0f / 3.0f, 4.0f / 3.0f})),
        "Unobserved non-finite GI must not invalidate observed rows");
  manager.updateContribution(last, torch::tensor({nan, 2.0f, 4.0f}), counts);
  check(!manager.hasValidContribution(), "Observed non-finite GI must invalidate entry");
  compareRebuild(manager, beta);
  manager.resetForGaussianCount(0);
  manager.updateContribution(last, torch::empty({0}), torch::empty({0}, counts.options()));
  check(manager.getImportance().numel() == 0, "Empty GI population is not supported");
}
void observationLifecycle() {
  RecentWindowConfig config;
  config.enabled = true;
  config.size = 2;
  RecentWindowImportanceManager manager(torch::kCPU, config);
  manager.resetForGaussianCount(3);
  manager.pushKeyframe(std::make_shared<GaussianKeyframe>(1), 3);
  manager.updateContribution(1, torch::tensor({1.0f, 2.0f, 99.0f}), torch::tensor({1, 1, 0}));
  manager.pushKeyframe(std::make_shared<GaussianKeyframe>(2), 3);
  manager.updateContribution(2, torch::tensor({99.0f, 2.0f, 3.0f}), torch::tensor({0, 1, 1}));
  check(manager.observedMask().all().item<bool>(), "Observation union lost shared rows");
  manager.pushKeyframe(std::make_shared<GaussianKeyframe>(3), 3);
  check(manager.observedMask().equal(torch::tensor({false, true, true})),
        "Evicted keyframe still supervised historical-only rows");
  manager.updateContribution(2, torch::tensor({0.0f, 0.0f, 3.0f}), torch::tensor({0, 0, 1}));
  check(manager.observedMask().equal(torch::tensor({false, false, true})),
        "Re-rendered keyframe retained its old observation mask");
  manager.appendGaussians(1);
  check(!manager.observedMask()[3].item<bool>(), "Appended row inherited observation");
  manager.pruneGaussians(torch::tensor({false, true, true, true}));
  manager.assertAligned(torch::zeros({3, 3}));
  check(manager.observedMask().equal(torch::tensor({false, true, false})),
        "Prune changed observation index correspondence");
  manager.invalidateContributions();
  check(!manager.observedMask().any().item<bool>(), "Geometry invalidation kept observations");
}
void temporalOnlyLifecycle() {
  RecentWindowConfig config;
  config.enabled = true;
  config.size = 2; config.collect_importance = false;
  RecentWindowImportanceManager manager(torch::kCPU, config);
  for (size_t id = 1; id <= 3; ++id)
    manager.pushKeyframe(std::make_shared<GaussianKeyframe>(id), 10);
  check(manager.keyframeIds() == std::vector<size_t>({2, 3}), "SGI changed Guard/Replay temporal membership");
  check(!manager.getImportance().defined(), "SGI allocated a redundant RecentWindow GI cache");
  manager.appendGaussians(5);
  manager.pruneGaussians(torch::ones({15}, torch::kBool));
  manager.resetForGaussianCount(0);
  check(manager.keyframeIds() == std::vector<size_t>({2, 3}), "Topology changed temporal history");
}
}
int main() {
  temporalOnlyLifecycle();
  observationLifecycle();
  for (const float beta : {0.0f, 0.5f, 0.9f, 1.0f})
    for (const int64_t size : {1, 2, 16}) rollover(size, beta);
  return 0;
}

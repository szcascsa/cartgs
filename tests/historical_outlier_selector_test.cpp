#include "historical_outlier_selector.h"

#include <cassert>
#include <memory>

namespace {
std::shared_ptr<GaussianKeyframe> makeKeyframe(std::size_t id, double x) {
  auto keyframe = std::make_shared<GaussianKeyframe>(id);
  keyframe->setPose(Eigen::Quaterniond::Identity(), Eigen::Vector3d(x, 0.0, 0.0));
  return keyframe;
}
}

int main() {
  HistoricalOutlierConfig config;
  config.count = 2;
  config.diversity_enabled = true;
  HistoricalOutlierSelector selector(config);

  std::vector<std::shared_ptr<GaussianKeyframe>> keyframes = {
      makeKeyframe(1, 0.0), makeKeyframe(2, 1.0), makeKeyframe(3, 10.0),
      makeKeyframe(4, -10.0)};
  const auto selected = selector.select(keyframes, {1, 2});
  assert(selected.size() == 2);
  for (const auto& view : selected) {
    assert(view.keyframe);
    assert(view.keyframe_id != 1 && view.keyframe_id != 2);
    assert(view.outlier_score >= 0.0);
  }

  assert(selector.select(keyframes, {}).empty());
  assert(selector.select(keyframes, {1, 2, 3, 4}).empty());
  return 0;
}

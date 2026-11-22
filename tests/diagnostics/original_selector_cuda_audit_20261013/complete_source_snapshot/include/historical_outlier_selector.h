#pragma once

#include <Eigen/Geometry>

#include <cstddef>
#include <memory>
#include <vector>

#include "gaussian_keyframe.h"

struct HistoricalOutlierConfig {
  int count = 4;
  bool diversity_enabled = true;
  float translation_weight = 1.0f;
  float rotation_weight = 0.5f;
  float diversity_recent_weight = 0.5f;
};

struct HistoricalOutlierView {
  std::size_t keyframe_id = 0;
  std::shared_ptr<GaussianKeyframe> keyframe;
  double outlier_score = 0.0;
};

class HistoricalOutlierSelector {
 public:
  explicit HistoricalOutlierSelector(HistoricalOutlierConfig config = {});

  std::vector<HistoricalOutlierView> select(
      const std::vector<std::shared_ptr<GaussianKeyframe>>& all_keyframes,
      const std::vector<std::size_t>& recent_keyframe_ids) const;

 private:
  double poseDistance(const GaussianKeyframe& a,
                      const GaussianKeyframe& b,
                      double translation_scale) const;
  HistoricalOutlierConfig config_;
};

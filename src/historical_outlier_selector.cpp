#include "include/historical_outlier_selector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

HistoricalOutlierSelector::HistoricalOutlierSelector(
    HistoricalOutlierConfig config)
    : config_(std::move(config)) {
  if (config_.count < 0 || !std::isfinite(config_.translation_weight) ||
      !std::isfinite(config_.rotation_weight) ||
      !std::isfinite(config_.diversity_recent_weight) ||
      config_.translation_weight < 0.0f || config_.rotation_weight < 0.0f ||
      config_.diversity_recent_weight < 0.0f ||
      config_.diversity_recent_weight > 1.0f)
    throw std::invalid_argument("Invalid HistoricalOutlier configuration");
}

double HistoricalOutlierSelector::poseDistance(
    const GaussianKeyframe& a, const GaussianKeyframe& b,
    double translation_scale) const {
  const auto pose_a = a.getPosef().inverse();
  const auto pose_b = b.getPosef().inverse();
  const double translation =
      (pose_a.translation() - pose_b.translation()).norm() /
      std::max(translation_scale, 1e-6);
  const auto relative = pose_a.so3().inverse() * pose_b.so3();
  const double angle = relative.log().norm() / kPi;
  return config_.translation_weight * translation +
         config_.rotation_weight * angle;
}

std::vector<HistoricalOutlierView> HistoricalOutlierSelector::select(
    const std::vector<std::shared_ptr<GaussianKeyframe>>& all_keyframes,
    const std::vector<std::size_t>& recent_keyframe_ids) const {
  if (config_.count == 0 || recent_keyframe_ids.empty()) return {};
  std::unordered_set<std::size_t> recent_ids(recent_keyframe_ids.begin(),
                                             recent_keyframe_ids.end());
  std::vector<std::shared_ptr<GaussianKeyframe>> recent;
  std::vector<std::shared_ptr<GaussianKeyframe>> historical;
  for (const auto& keyframe : all_keyframes) {
    if (!keyframe) continue;
    if (recent_ids.count(keyframe->fid_)) recent.push_back(keyframe);
    else historical.push_back(keyframe);
  }
  if (recent.empty() || historical.empty()) return {};

  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (const auto& keyframe : recent)
    center += keyframe->getPose().inverse().translation();
  center /= static_cast<double>(recent.size());
  double translation_scale = 0.0;
  for (const auto& keyframe : recent)
    translation_scale = std::max(
        translation_scale,
        (keyframe->getPose().inverse().translation() - center).norm());
  for (const auto& keyframe : historical)
    translation_scale = std::max(
        translation_scale,
        (keyframe->getPose().inverse().translation() - center).norm());
  translation_scale = std::max(translation_scale, 1e-3);

  std::vector<double> recent_distance(historical.size(), 0.0);
  for (std::size_t i = 0; i < historical.size(); ++i) {
    for (const auto& recent_kf : recent)
      recent_distance[i] = std::max(
          recent_distance[i],
          poseDistance(*historical[i], *recent_kf, translation_scale));
  }

  std::vector<HistoricalOutlierView> selected;
  std::vector<bool> used(historical.size(), false);
  const auto limit = std::min<std::size_t>(
      static_cast<std::size_t>(config_.count), historical.size());
  for (std::size_t pick = 0; pick < limit; ++pick) {
    std::size_t best = historical.size();
    double best_score = -1.0;
    for (std::size_t i = 0; i < historical.size(); ++i) {
      if (used[i]) continue;
      double diversity = recent_distance[i];
      if (config_.diversity_enabled && !selected.empty()) {
        diversity = std::numeric_limits<double>::infinity();
        for (const auto& selected_view : selected) {
          diversity = std::min(
              diversity,
              poseDistance(*historical[i], *selected_view.keyframe,
                           translation_scale));
        }
      }
      const double score = config_.diversity_enabled && !selected.empty()
          ? config_.diversity_recent_weight * recent_distance[i] +
                (1.0 - config_.diversity_recent_weight) * diversity
          : recent_distance[i];
      if (score > best_score) {
        best = i;
        best_score = score;
      }
    }
    if (best == historical.size()) break;
    used[best] = true;
    selected.push_back({historical[best]->fid_, historical[best],
                        recent_distance[best]});
  }
  return selected;
}

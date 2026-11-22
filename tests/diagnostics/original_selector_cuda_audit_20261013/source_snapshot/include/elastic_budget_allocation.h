#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

struct ProtectionConfig {
  int age_ready = 100;
  int optimization_ready = 80;
  int unique_keyframe_ready = 4;
  int stale_age = 1000;
  float max_budget_fraction = 0.35f;
  bool performance_logging = true;
  void validate() const {
    if (age_ready <= 0 || optimization_ready <= 0 || unique_keyframe_ready <= 0 ||
        stale_age < age_ready || !std::isfinite(max_budget_fraction) ||
        max_budget_fraction < 0 || max_budget_fraction > 1)
      throw std::invalid_argument("Invalid Protection evidence thresholds or budget fraction");
  }
};

struct ElasticBudgetAllocation {
  double total = 0.0, protection_demand = 0.0, protection = 0.0, mature = 0.0;
};

inline double computeProtectionBudget(double demand, double total, double fraction) {
  if (!std::isfinite(demand) || !std::isfinite(total) || !std::isfinite(fraction) ||
      demand < 0 || total < 0 || fraction < 0 || fraction > 1)
    throw std::invalid_argument("Invalid Protection budget");
  return std::min(demand, fraction * total);
}
inline double computeMatureTarget(double total, int64_t actual_protected, int64_t mature) {
  return std::clamp(total - actual_protected, 0.0, static_cast<double>(mature));
}

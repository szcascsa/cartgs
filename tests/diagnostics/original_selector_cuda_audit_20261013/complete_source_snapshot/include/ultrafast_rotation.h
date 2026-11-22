#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

// Only the normal Mapping optimizer decision completes a slot. No KF state.
struct UltrafastRotation {
  static constexpr std::array<float, 4> ratios = {0.01f, 0.05f, 0.10f, 0.15f};
  static constexpr std::array<const char*, 5> names = {
      "LOD_001", "LOD_005", "LOD_010", "LOD_015", "FULL"};
  bool enabled = false;
  std::uint64_t step = 0;
  std::array<std::uint64_t, 5> counts{};
  std::array<double, 5> elapsed_sum{}, render_sum{};
  std::vector<double> elapsed;
  std::uint64_t full_render_count = 0, lod_render_count = 0;
  std::uint64_t selector_updates = 0, tf_updates = 0;
  bool online_ended = false;
  std::uint64_t online_end_mapping_iteration = 0, online_end_rotation_step = 0;

  std::size_t phase() const { return step % names.size(); }
  bool full() const { return phase() == ratios.size(); }
  float ratio() const { return full() ? 1.0f : ratios[phase()]; }
  bool allowsNormalStep(bool online, std::uint64_t mapping, std::uint64_t limit) const {
    return !online_ended && online && (limit == 0 || mapping < limit);
  }
  void endOnline(std::uint64_t mapping) {
    if (online_ended) return;
    online_ended = true;
    online_end_mapping_iteration = mapping;
    online_end_rotation_step = step;
  }
  std::uint64_t tailSteps() const {
    return online_ended ? step - online_end_rotation_step : 0;
  }
  void complete(double ms, double render_ms, bool rendered,
                bool selector_updated, bool tf_updated) {
    const auto index = phase();
    ++counts[index];
    elapsed_sum[index] += ms;
    render_sum[index] += render_ms;
    elapsed.push_back(ms);
    if (rendered) ++(full() ? full_render_count : lod_render_count);
    selector_updates += selector_updated;
    tf_updates += tf_updated;
    ++step;
  }
  bool balanced() const {
    const auto range = std::minmax_element(counts.begin(), counts.end());
    return *range.second - *range.first <= 1 &&
        std::accumulate(counts.begin(), counts.end(), std::uint64_t{0}) == step;
  }
  double mean() const {
    return elapsed.empty() ? 0.0 :
        std::accumulate(elapsed.begin(), elapsed.end(), 0.0) / elapsed.size();
  }
  double p95() const {
    if (elapsed.empty()) return 0.0;
    auto sorted = elapsed;
    std::sort(sorted.begin(), sorted.end());
    return sorted[static_cast<std::size_t>(std::ceil(sorted.size() * 0.95)) - 1];
  }
};

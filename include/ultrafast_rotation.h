#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

// Only the normal Mapping optimizer decision completes a slot. No KF state.
struct UltrafastRotation {
  static constexpr std::array<float, 4> ratios = {0.01f, 0.05f, 0.10f, 0.15f};
  static constexpr std::array<const char*, 5> names = {
      "LOD_001", "LOD_005", "LOD_010", "LOD_015", "FULL"};
  bool enabled = false;
  bool rotate_tail = true;
  bool serial_full_lod = false;
  bool serial_lod_detach_master = true;
  std::uint64_t step = 0;
  std::vector<float> active_ratios{ratios.begin(),ratios.end()};
  std::vector<std::uint64_t> counts=std::vector<std::uint64_t>(5);
  std::vector<double> elapsed_sum=std::vector<double>(5), render_sum=std::vector<double>(5);
  std::vector<double> elapsed;
  std::uint64_t full_render_count = 0, lod_render_count = 0;
  std::uint64_t selector_updates = 0, tf_updates = 0;
  bool online_ended = false;
  std::uint64_t online_end_mapping_iteration = 0, online_end_rotation_step = 0;
  std::uint64_t online_end_selector_updates = 0, online_end_tf_updates = 0;
  std::vector<std::uint64_t> tail_ratio_counts=std::vector<std::uint64_t>(4);
  std::vector<std::uint64_t> tail_phase_counts=std::vector<std::uint64_t>(5);
  std::uint64_t tail_full_renders = 0, tail_recovery_steps = 0;
  std::uint64_t tail_lod_attempts = 0, tail_lod_renders = 0;
  int post_topology_full_steps = 0;
  std::uint64_t recovery_remaining = 0, recovery_windows = 0;
  std::uint64_t recovery_steps = 0, online_recovery_steps = 0, online_end_recovery_steps = 0;

  void configureRatios(const std::vector<float>& values) {
    active_ratios=values;counts.assign(values.size()+1,0);
    elapsed_sum.assign(counts.size(),0);render_sum.assign(counts.size(),0);
    tail_ratio_counts.assign(values.size(),0);tail_phase_counts.assign(counts.size(),0);
  }
  std::string phaseName(std::size_t index) const {
    if(index==active_ratios.size())return "FULL";
    auto value=std::to_string(int(std::lround(active_ratios[index]*1000)));
    return "LOD_"+std::string(value.size()<3?3-value.size():0,'0')+value;
  }
  std::string countKey(std::size_t index) const {
    return index==active_ratios.size()?"full":"lod_"+phaseName(index).substr(4);
  }
  std::size_t phase() const { return (step - online_recovery_steps - tail_recovery_steps) % (active_ratios.size()+(serial_full_lod?0:1)); }
  bool full() const { return serial_full_lod || phase() == active_ratios.size(); }
  bool recovering() const { return recovery_remaining > 0; }
  void beginRecovery() {
    if (!post_topology_full_steps) return;
    recovery_remaining = post_topology_full_steps;
    ++recovery_windows;
  }
  // Extra Full mappings pause the ordinary ratio cursor but keep the real clock.
  void completeRecovery(double ms, bool record_timing) {
    if (online_ended && !serial_full_lod && rotate_tail) { ++tail_recovery_steps; ++tail_full_renders; }
    else ++online_recovery_steps;
    ++full_render_count; ++step;
    if (record_timing) elapsed.push_back(ms);
  }
  float ratio() const { return serial_full_lod || !full() ? active_ratios[phase()] : 1.0f; }
  void endOnline(std::uint64_t mapping) {
    if (online_ended) return;
    online_ended = true;
    online_end_mapping_iteration = mapping;
    online_end_rotation_step = step;
    online_end_recovery_steps = online_recovery_steps;
    online_end_selector_updates = selector_updates;
    online_end_tf_updates = tf_updates;
  }
  std::uint64_t tailSteps() const {
    return online_ended ? step - online_end_rotation_step : 0;
  }
  std::size_t tailRatioIndex() const {
    if (serial_full_lod) return (online_end_rotation_step + tail_lod_attempts) % active_ratios.size();
    if (rotate_tail) return phase();
    const auto next=(online_end_rotation_step-online_end_recovery_steps)%(active_ratios.size()+1);
    return ((next==active_ratios.size()?0:next)+tail_lod_attempts)%active_ratios.size();
  }
  void completeSerial(double ms, double render_ms, bool lod_rendered,
                      bool selector_updated, bool tf_updated, bool record_timing = true) {
    const auto index = phase();
    ++counts[index]; ++full_render_count;
    lod_render_count += lod_rendered;
    selector_updates += selector_updated; tf_updates += tf_updated;
    if (record_timing) {
      elapsed_sum[index] += ms; render_sum[index] += render_ms; elapsed.push_back(ms);
    }
    ++step;
  }
  void completeTail(double ms, bool lod_attempted, bool lod_rendered,
                    bool selector_updated, bool tf_updated, bool record_timing = true) {
    if (lod_attempted) { ++tail_ratio_counts[tailRatioIndex()]; ++tail_lod_attempts; }
    ++full_render_count; ++tail_full_renders;
    if (lod_rendered) { ++lod_render_count; ++tail_lod_renders; }
    selector_updates += selector_updated;
    tf_updates += tf_updated;
    if (record_timing) elapsed.push_back(ms);
    ++step;
  }
  // Normal mode1 tail continues the same five-slot cursor as online Mapping.
  // Serial Full+LoD retains its separate, explicitly requested schedule.
  void completeRotatingTail(double ms, double render_ms, bool rendered, bool lod_attempted,
                            bool selector_updated, bool tf_updated, bool record_timing = true) {
    const auto index = phase();
    ++tail_phase_counts[index];
    if (!full() && lod_attempted) { ++tail_ratio_counts[index]; ++tail_lod_attempts; }
    if (rendered) {
      if (full()) { ++full_render_count; ++tail_full_renders; }
      else { ++lod_render_count; ++tail_lod_renders; }
    }
    selector_updates += selector_updated; tf_updates += tf_updated;
    if (record_timing) elapsed.push_back(ms);
    ++step;
  }
  void complete(double ms, double render_ms, bool rendered,
                bool selector_updated, bool tf_updated, bool record_timing = true) {
    const auto index = phase();
    ++counts[index];
    if (record_timing) {
      elapsed_sum[index] += ms;
      render_sum[index] += render_ms;
      elapsed.push_back(ms);
    }
    if (rendered) ++(full() ? full_render_count : lod_render_count);
    selector_updates += selector_updated;
    tf_updates += tf_updated;
    ++step;
  }
  bool balanced() const {
    const auto range = std::minmax_element(counts.begin(), serial_full_lod ? counts.end()-1 : counts.end());
    return *range.second - *range.first <= 1 &&
        std::accumulate(counts.begin(), counts.end(), std::uint64_t{0}) ==
            (online_ended ? online_end_rotation_step - online_end_recovery_steps : step - online_recovery_steps) &&
        std::accumulate(tail_ratio_counts.begin(), tail_ratio_counts.end(), std::uint64_t{0}) == tail_lod_attempts &&
        (!online_ended || serial_full_lod || !rotate_tail ||
         std::accumulate(tail_phase_counts.begin(), tail_phase_counts.end(), std::uint64_t{0}) +
             tail_recovery_steps == tailSteps());
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

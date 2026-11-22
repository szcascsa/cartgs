#pragma once

#include <cstdint>
#include <stdexcept>

// Tail length counts completed normal mappings, without owning any training clock.
struct TailOptimizationSchedule {
  bool enabled = false;
  int steps = -1;
  bool active = false;
  std::uint64_t start_iteration = 0, start_mapping = 0, target_steps = 0;

  void validate() const {
    if (steps < -1)
      throw std::invalid_argument("Training.tail_optimization_steps must be -1 or nonnegative");
  }
  bool begin(std::uint64_t iteration, std::uint64_t mapping) {
    if (!enabled || active) return false;
    start_iteration = iteration;
    start_mapping = mapping;
    target_steps = steps == -1 ? iteration / 2 : static_cast<std::uint64_t>(steps);
    active = true;
    return true;
  }
  std::uint64_t completed(std::uint64_t mapping) const {
    return active && mapping >= start_mapping ? mapping - start_mapping : 0;
  }
  bool complete(std::uint64_t mapping) const {
    return active && completed(mapping) >= target_steps;
  }
  std::uint64_t mappingLimit(std::uint64_t configured, bool has_slam) const {
    return enabled && has_slam ? 0 : configured;
  }
  bool allowsOptimization(int iteration, int limit) const {
    return active || limit == -1 || iteration < limit;
  }
};

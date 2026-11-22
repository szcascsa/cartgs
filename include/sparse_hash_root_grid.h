#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <vector>

namespace hierarchical_si {
using RootKey = std::array<int64_t, 3>;
using Position = std::array<double, 3>;
constexpr int64_t kTargetOccupiedRoots = 2048;

struct RootSearchSample {
  double h = 0;
  int64_t occupied = 0;
};

struct SparseRootGrid {
  double h = 0, h_seed = 0, d50 = 0, bracket_low = 0, bracket_high = 0;
  double search_ms = 0, grouping_ms = 0;
  bool crossing_bracketed = false;
  int64_t unique_positions = 0;
  std::vector<RootSearchSample> samples;
  // Only occupied triplets are allocated. Map order defines deterministic traversal.
  std::map<RootKey, std::vector<int64_t>> roots;
};

RootKey sparseRootKey(const Position& xyz, double h);

// This API deliberately accepts no importance, coverage, GT or quality information.
SparseRootGrid makeSparseRootGrid(const std::vector<Position>& candidate_xyz,
    const std::vector<double>& candidate_radius, const std::vector<int64_t>& source_rows);
}  // namespace hierarchical_si

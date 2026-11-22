#pragma once

#include <torch/torch.h>
#include <array>
#include <filesystem>
#include <string>
#include <vector>
#include "sparse_hash_root_grid.h"

namespace hierarchical_si {
constexpr int kGridBits = 11;
constexpr int kInitialCells = 1 << kGridBits;  // Legacy bin count; sparse backend targets occupied roots.
constexpr int kRepresentatives = 3;
constexpr int kMaxLevel = 3;
constexpr double kSplitCoverage = 0.80;
constexpr double kEpsilon = 1e-12;
enum class RootBackend { LegacyAabb2048, SparseHashTarget2048 };

struct CellSelection {
  std::vector<int64_t> representatives;  // Local indices, in greedy order.
  std::vector<double> gains;
  torch::Tensor cover;
  double mass = 0, covered_mass = 0, quality = 1;
};

struct Node {
  int level = 1;
  int64_t root = 0;  // Diagnostic root ordinal; identity is root_key for sparse mode.
  RootKey root_key{};
  std::string path;
  std::array<double, 3> lower{}, upper{};
  int64_t occupancy = 0;
  bool leaf = true;
  double mass = 0, covered_mass = 0, quality = 1, residual_mass = 0;
  double conservation_absolute = 0, conservation_relative = 0;
  std::vector<int64_t> representatives;
  std::vector<double> representative_raw, gains;
};

struct Result {
  RootBackend backend = RootBackend::SparseHashTarget2048;
  torch::Tensor importance, leaf_assignment;
  std::array<int, 3> dimensions{1, 1, kInitialCells};
  std::array<double, 3> lower{}, upper{}, cell_size{};
  std::vector<Node> nodes;
  int64_t population = 0, candidates = 0;
  double raw_sum = 0, output_sum = 0, conservation_absolute = 0, conservation_relative = 0;
  double build_ms = 0, similarity_ms = 0, selection_ms = 0;
  double h_search_ms = 0, root_grouping_ms = 0, recursive_split_ms = 0;
  double h_root = 0, d50 = 0, h_seed = 0, bracket_low = 0, bracket_high = 0;
  bool crossing_bracketed = false;
  std::vector<RootSearchSample> search_samples;
};

// FP64 cell-level tensor operations, with bounded candidate-row blocks. No N*N scene matrix.
CellSelection selectCell(const torch::Tensor& xyz, const torch::Tensor& radius,
    const torch::Tensor& raw, const torch::Tensor& source_rows,
    double* similarity_ms = nullptr, double* selection_ms = nullptr);

// Stateless: every invocation rebuilds from current geometry/scale/raw/masks.
Result build(const torch::Tensor& xyz, const torch::Tensor& activated_scale,
    const torch::Tensor& raw, const torch::Tensor& valid, const torch::Tensor& observed,
    RootBackend backend = RootBackend::SparseHashTarget2048);

// Disabled path returns the original Tensor handle without accessing geometry or masks.
torch::Tensor score(bool enabled, const torch::Tensor& xyz, const torch::Tensor& activated_scale,
    const torch::Tensor& raw, const torch::Tensor& valid, const torch::Tensor& observed,
    Result* result = nullptr);

// Called only by an explicit diagnostic flag. Tree includes provisional split nodes.
void writeDiagnostics(const Result& result, const std::filesystem::path& directory,
    const std::string& stem);
}  // namespace hierarchical_si

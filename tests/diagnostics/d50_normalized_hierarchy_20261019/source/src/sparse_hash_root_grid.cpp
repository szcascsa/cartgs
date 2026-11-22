#include "include/sparse_hash_root_grid.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace hierarchical_si {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
int64_t countOccupied(const std::vector<Position>& xyz, double h) {
  std::vector<RootKey> keys; keys.reserve(xyz.size());
  for (const auto& point : xyz) keys.push_back(sparseRootKey(point, h));
  std::sort(keys.begin(), keys.end());
  return std::unique(keys.begin(), keys.end()) - keys.begin();
}
}  // namespace

RootKey sparseRootKey(const Position& xyz, double h) {
  require(h > 0 && std::isfinite(h), "Sparse root h must be finite and positive");
  RootKey result;
  for (int a = 0; a < 3; ++a) {
    require(std::isfinite(xyz[a]), "Sparse root xyz must be finite");
    const double value = std::floor(xyz[a] / h);  // Origin is always (0,0,0).
    // Use long double bounds to avoid INT64_MAX rounding up when converted to double.
    require(std::isfinite(value) && static_cast<long double>(value) >= std::numeric_limits<int64_t>::min() &&
        static_cast<long double>(value) <= std::numeric_limits<int64_t>::max(), "Sparse root key exceeds int64 range");
    result[a] = static_cast<int64_t>(value);
  }
  return result;
}

SparseRootGrid makeSparseRootGrid(const std::vector<Position>& xyz,
    const std::vector<double>& radius, const std::vector<int64_t>& rows) {
  require(xyz.size() == radius.size() && xyz.size() == rows.size(), "Sparse root candidate arrays mismatch");
  SparseRootGrid result;
  if (xyz.empty()) return result;
  auto start = Clock::now();
  std::vector<double> sorted_radius = radius;
  double max_coordinate = 0;
  for (size_t i = 0; i < xyz.size(); ++i) {
    require(radius[i] > 0 && std::isfinite(radius[i]), "Sparse root radius must be finite and positive");
    for (double coordinate : xyz[i]) {
      require(std::isfinite(coordinate), "Sparse root xyz must be finite");
      max_coordinate = std::max(max_coordinate, std::abs(coordinate));
    }
  }
  std::sort(sorted_radius.begin(), sorted_radius.end());
  size_t middle = sorted_radius.size() / 2;
  double median = sorted_radius.size() % 2 ? sorted_radius[middle] :
      sorted_radius[middle - 1] * .5 + sorted_radius[middle] * .5;
  result.d50 = 2 * median; result.h_seed = 8 * result.d50;
  require(std::isfinite(result.h_seed) && result.h_seed > 0, "Sparse root numerical seed outside supported range");
  // Safety bounds protect int64 keys and FP64 divisions; never allocate by coordinate span.
  const double min_h = std::max(std::numeric_limits<double>::min() * 1024, std::ldexp(max_coordinate, -61));
  const double max_h = std::numeric_limits<double>::max() / 1024;
  require(min_h <= max_h, "Sparse root coordinate range outside supported FP64 scale");
  auto distinct = xyz;
  std::sort(distinct.begin(), distinct.end());
  result.unique_positions = std::unique(distinct.begin(), distinct.end()) - distinct.begin();
  double best_h = 0; int64_t best_error = std::numeric_limits<int64_t>::max();
  std::map<double, int64_t> evaluated;
  auto evaluate = [&](double h) {
    auto it = evaluated.find(h);
    if (it != evaluated.end()) return it->second;
    int64_t count = countOccupied(xyz, h);
    evaluated.emplace(h, count); result.samples.push_back({h, count});
    int64_t error = std::abs(count - kTargetOccupiedRoots);
    if (error < best_error || (error == best_error && h > best_h)) { best_error = error; best_h = h; }
    return count;
  };
  double current = std::clamp(result.h_seed, min_h, max_h);
  int64_t count = evaluate(current);
  double low = current, high = current;
  if (count > kTargetOccupiedRoots) {
    for (int step = 0; step < 256; ++step) {
      double next = std::min(current * 2, max_h);
      if (next <= current) break;
      low = current; high = next; current = next; count = evaluate(current);
      if (count <= kTargetOccupiedRoots) { result.crossing_bracketed = true; break; }
    }
  } else if (count < kTargetOccupiedRoots) {
    for (int step = 0; step < 256; ++step) {
      if (count == result.unique_positions) break;  // Target unattainable: every distinct xyz already has a cell.
      double next = std::max(current * .5, min_h);
      if (next >= current) break;
      low = next; high = current; current = next; count = evaluate(current);
      if (count >= kTargetOccupiedRoots) { result.crossing_bracketed = true; break; }
    }
  } else {
    low = std::max(current * .5, min_h); high = std::min(current * 2, max_h);
    evaluate(low); evaluate(high); result.crossing_bracketed = true;
  }
  // If target exceeds all unique xyz, still search around the seed for the coarsest tested optimum.
  if (low == high) { low = std::max(current * .5, min_h); high = std::min(current * 2, max_h); }
  result.bracket_low = low; result.bracket_high = high;
  auto log_samples = [&](double lo, double hi) {
    for (int i = 0; i < 128; ++i) {
      double h = i == 0 ? lo : i == 127 ? hi : std::exp(std::log(lo) +
          (std::log(hi) - std::log(lo)) * (static_cast<double>(i) / 127));
      evaluate(h);
    }
  };
  log_samples(low, high);
  // One fixed refinement over tested neighbors of the current winner; never inspect hierarchy.
  auto winner = evaluated.find(best_h);
  auto left = winner == evaluated.begin() ? winner : std::prev(winner);
  auto right = std::next(winner); if (right == evaluated.end()) right = winner;
  log_samples(left->first, right->first);
  result.h = best_h; result.search_ms = milliseconds(start);
  start = Clock::now();
  for (size_t i = 0; i < xyz.size(); ++i) result.roots[sparseRootKey(xyz[i], result.h)].push_back(rows[i]);
  for (auto& root : result.roots) std::sort(root.second.begin(), root.second.end());
  result.grouping_ms = milliseconds(start);
  require(static_cast<int64_t>(result.roots.size()) == evaluated.at(best_h), "Sparse occupied count/grouping mismatch");
  return result;
}

SparseRootGrid makeD50NormalizedRootGrid(const std::vector<Position>& xyz,
    const std::vector<double>& radius, const std::vector<int64_t>& rows) {
  require(xyz.size() == radius.size() && xyz.size() == rows.size(), "D50 root arrays mismatch");
  SparseRootGrid result;
  if (xyz.empty()) return result;
  const auto start = Clock::now();
  auto sorted_radius = radius;
  for (size_t i = 0; i < xyz.size(); ++i) {
    require(radius[i] > 0 && std::isfinite(radius[i]), "D50 radius must be finite and positive");
    for (double value : xyz[i]) require(std::isfinite(value), "D50 xyz must be finite");
  }
  std::sort(sorted_radius.begin(), sorted_radius.end());
  const size_t middle = sorted_radius.size() / 2;
  const double median = sorted_radius.size() % 2 ? sorted_radius[middle] :
      sorted_radius[middle - 1] * .5 + sorted_radius[middle] * .5;
  result.d50 = 2 * median;
  result.h = 32 * result.d50;
  require(result.h > 0 && std::isfinite(result.h), "32*d50 outside finite positive range");
  for (size_t i = 0; i < xyz.size(); ++i)
    result.roots[sparseRootKey(xyz[i], result.h)].push_back(rows[i]);
  for (auto& item : result.roots) std::sort(item.second.begin(), item.second.end());
  result.grouping_ms = milliseconds(start);
  return result;
}
}  // namespace hierarchical_si

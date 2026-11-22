#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

struct ConstraintDualSolution {
  std::vector<double> multipliers;
  bool converged = false;
  double violation = 0.0;
};

namespace gradient_constraint_detail {
// Complete pivoting reveals redundant rows without adding a ridge or relaxing
// any constraint. Free coordinates are zero for compatible rank-deficient systems.
inline bool solveActiveSystem(std::vector<std::vector<double>> a,
                              std::vector<double> b, std::vector<double>& x,
                              double tolerance) {
  const auto n = b.size();
  std::vector<std::size_t> columns(n);
  std::iota(columns.begin(), columns.end(), 0);
  double scale = 0.0, rhs_scale = 1.0;
  for (std::size_t i = 0; i < n; ++i) {
    rhs_scale = std::max(rhs_scale, std::abs(b[i]));
    for (double value : a[i]) scale = std::max(scale, std::abs(value));
  }
  std::size_t rank = 0;
  for (; rank < n; ++rank) {
    std::size_t row = rank, column = rank;
    double pivot = 0.0;
    for (std::size_t i = rank; i < n; ++i)
      for (std::size_t j = rank; j < n; ++j)
        if (std::abs(a[i][j]) > pivot) {
          pivot = std::abs(a[i][j]); row = i; column = j;
        }
    if (pivot <= 1e-12 * std::max(scale, 1e-20)) break;
    std::swap(a[rank], a[row]); std::swap(b[rank], b[row]);
    for (auto& values : a) std::swap(values[rank], values[column]);
    std::swap(columns[rank], columns[column]);
    for (std::size_t i = rank + 1; i < n; ++i) {
      const double factor = a[i][rank] / a[rank][rank];
      a[i][rank] = 0.0;
      for (std::size_t j = rank + 1; j < n; ++j) a[i][j] -= factor * a[rank][j];
      b[i] -= factor * b[rank];
    }
  }
  for (std::size_t i = rank; i < n; ++i)
    if (std::abs(b[i]) > tolerance * rhs_scale) return false;
  std::vector<double> permuted(n, 0.0);
  for (std::size_t k = rank; k > 0; --k) {
    const auto i = k - 1;
    double value = b[i];
    for (std::size_t j = i + 1; j < rank; ++j) value -= a[i][j] * permuted[j];
    permuted[i] = value / a[i][i];
    if (!std::isfinite(permuted[i])) return false;
  }
  x.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) x[columns[i]] = permuted[i];
  return true;
}
}

// H = normalized A A^T, r = normalized A d0. Solve the nonnegative dual
// with an active set, retaining distinct constraints even when nearly parallel.
inline ConstraintDualSolution solveConstraintDual(
    const std::vector<std::vector<double>>& h, const std::vector<double>& r,
    int max_iterations = 256, double tolerance = 1e-7) {
  const auto n = r.size();
  ConstraintDualSolution result;
  result.multipliers.assign(n, 0.0);
  if (h.size() != n) return result;
  for (std::size_t i = 0; i < n; ++i) {
    if (h[i].size() != n || !std::isfinite(r[i])) return result;
    for (double value : h[i]) if (!std::isfinite(value)) return result;
  }
  std::vector<bool> active(n, false);
  std::vector<double> residual(n);
  auto measure = [&] {
    double kkt = 0.0;
    result.violation = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      residual[i] = r[i];
      for (std::size_t j = 0; j < n; ++j)
        residual[i] -= h[i][j] * result.multipliers[j];
      result.violation = std::max(result.violation, residual[i]);
      kkt = std::max(kkt, result.multipliers[i] > 0.0
                            ? std::abs(residual[i]) : std::max(0.0, residual[i]));
    }
    result.converged = kkt <= tolerance;
  };
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    measure();
    if (result.converged) return result;
    std::size_t entering = n;
    double worst = tolerance;
    for (std::size_t i = 0; i < n; ++i)
      if (!active[i] && residual[i] > worst) { entering = i; worst = residual[i]; }
    if (entering == n) return result;  // May be primal-feasible without optimal dual KKT.
    active[entering] = true;
    for (int inner = 0; inner < max_iterations; ++inner) {
      std::vector<std::size_t> indices;
      for (std::size_t i = 0; i < n; ++i) if (active[i]) indices.push_back(i);
      std::vector<std::vector<double>> matrix(indices.size(), std::vector<double>(indices.size()));
      std::vector<double> rhs, solution;
      for (std::size_t i = 0; i < indices.size(); ++i) {
        rhs.push_back(r[indices[i]]);
        for (std::size_t j = 0; j < indices.size(); ++j) matrix[i][j] = h[indices[i]][indices[j]];
      }
      if (!gradient_constraint_detail::solveActiveSystem(matrix, rhs, solution, tolerance)) {
        measure(); return result;
      }
      std::vector<double> trial(n, 0.0);
      bool nonnegative = true;
      for (std::size_t i = 0; i < indices.size(); ++i) {
        trial[indices[i]] = solution[i];
        nonnegative = nonnegative && solution[i] >= 0.0;
      }
      if (nonnegative) { result.multipliers = std::move(trial); break; }
      double alpha = 1.0;
      for (auto i : indices)
        if (trial[i] < 0.0) alpha = std::min(alpha,
            result.multipliers[i] / (result.multipliers[i] - trial[i]));
      for (std::size_t i = 0; i < n; ++i) {
        result.multipliers[i] += alpha * (trial[i] - result.multipliers[i]);
        if (active[i] && result.multipliers[i] <= 1e-12) {
          result.multipliers[i] = 0.0; active[i] = false;
        }
      }
    }
  }
  measure();
  return result;
}

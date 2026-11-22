#include "include/gradient_constraint_solver.h"
#include <stdexcept>
#include <limits>
#include <random>

void check(bool ok) { if (!ok) throw std::runtime_error("Constraint solver regression"); }
int main() {
  auto one = solveConstraintDual({{1.0}}, {2.0});
  check(one.converged && std::abs(one.multipliers[0] - 2.0) < 1e-8);
  auto two = solveConstraintDual({{1.0, 0.0}, {0.0, 1.0}}, {2.0, -1.0});
  check(two.converged && two.multipliers[0] == 2.0 && two.multipliers[1] == 0.0);
  auto parallel = solveConstraintDual({{1.0, 1.0}, {1.0, 1.0}}, {2.0, 2.0});
  check(parallel.converged && std::abs(parallel.multipliers[0] + parallel.multipliers[1] - 2.0) < 1e-8);
  check(solveConstraintDual({{0.0}}, {0.0}).converged);
  check(!solveConstraintDual({{0.0}}, {1.0}).converged);
  check(!solveConstraintDual({{1.0}}, {std::numeric_limits<double>::quiet_NaN()}).converged);
  check(!solveConstraintDual({{1.0, 0.8}, {0.8, 1.0}}, {1.0, 1.0}, 1).converged);
  const double cosine = 1.0 - 1e-8;
  auto opposed = solveConstraintDual({{1.0, -cosine}, {-cosine, 1.0}}, {1e-4, 1e-4});
  check(opposed.converged && opposed.violation <= 1e-7);
  auto redundant = solveConstraintDual({{1.0, -1.0, 1.0}, {-1.0, 1.0, -1.0}, {1.0, -1.0, 1.0}},
                                      {1.0, -1.0, 1.0});
  check(redundant.converged && redundant.violation <= 1e-7);
  // Homogeneous quality constraints always admit d=0. Exercise PSD Gram
  // matrices with more constraints than dimensions (not just nonsingular QPs).
  std::mt19937 rng(42);
  std::normal_distribution<double> draw;
  for (int trial = 0; trial < 200; ++trial) {
    const int count = 2 + trial % 7, dims = 2 + trial % 4;
    std::vector<std::vector<double>> a(count, std::vector<double>(dims));
    std::vector<double> delta(dims), r(count, 0.0);
    for (auto& value : delta) value = draw(rng);
    for (auto& row : a) {
      double norm = 0.0;
      for (auto& value : row) { value = draw(rng); norm += value * value; }
      for (auto& value : row) value /= std::sqrt(norm);
    }
    std::vector<std::vector<double>> gram(count, std::vector<double>(count, 0.0));
    for (int i = 0; i < count; ++i)
      for (int k = 0; k < dims; ++k) {
        r[i] += a[i][k] * delta[k];
        for (int j = 0; j < count; ++j) gram[i][j] += a[i][k] * a[j][k];
      }
    const auto solution = solveConstraintDual(gram, r);
    check(solution.converged && solution.violation <= 1e-7);
    for (auto value : solution.multipliers) check(value >= 0.0 && std::isfinite(value));
  }
}

#include "include/gradient_constraint.h"
#include "include/gradient_constraint_solver.h"

#include <torch/csrc/autograd/autograd.h>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace {
using Blocks = std::vector<torch::Tensor>;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
torch::Tensor dot(const Blocks& a, const Blocks& b, const torch::TensorOptions& options) {
  auto result = torch::zeros({}, options.dtype(torch::kFloat64));
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i].defined() && b[i].defined() && a[i].numel() > 0)
      result = result + torch::dot(a[i].reshape({-1}).to(torch::kFloat64),
                                  b[i].reshape({-1}).to(torch::kFloat64));
  return result;
}
Blocks gradient(const torch::Tensor& loss, const AdamStepPreview& preview) {
  Blocks result(preview.parameters().size()), inputs;
  std::vector<std::size_t> indices;
  if (!loss.requires_grad()) return result;
  for (std::size_t i = 0; i < preview.parameters().size(); ++i) {
    if (preview.movable()[i] && preview.parameters()[i].requires_grad()) {
      inputs.push_back(preview.parameters()[i]); indices.push_back(i);
    }
  }
  if (!inputs.empty()) {
    auto grads = torch::autograd::grad({loss}, inputs, {}, false, false, true);
    for (std::size_t i = 0; i < grads.size(); ++i)
      if (grads[i].defined()) result[indices[i]] = grads[i].detach();
  }
  return result;
}
std::vector<double> values(const Blocks& scalars) {
  auto cpu = torch::stack(scalars).to(torch::kCPU).to(torch::kDouble).contiguous();
  const auto* data = cpu.data_ptr<double>();
  return {data, data + cpu.numel()};
}
bool finite(const std::vector<double>& data) {
  for (double x : data) if (!std::isfinite(x)) return false;
  return true;
}
}

void GradientConstraintConfig::validate() const {
  if (check_interval < 1 || memory_views_per_check < 1 || memory_ratios_per_check < 1 ||
      max_candidate_checks < 1 || max_candidate_checks > 3 || log_interval < 1 ||
      tf_warmup_steps < 0 || !std::isfinite(cancel_fallback_scale) ||
      cancel_fallback_scale <= 0.0 || cancel_fallback_scale > 1.0)
    throw std::invalid_argument("Invalid GradientConstraint configuration");
}
GradientConstraintManager::GradientConstraintManager(GradientConstraintConfig config)
    : config_(config) { config_.validate(); }

GradientConstraintReport GradientConstraintManager::step(
    torch::optim::Adam& optimizer, const GuardEvaluation& evaluation,
    const std::vector<std::size_t>& cuda_tf_groups) const {
  const auto start = Clock::now();
  GradientConstraintReport report;
  auto finish = [&](std::string reason, GuardFailureKind failure = GuardFailureKind::None) {
    report.reason = std::move(reason); report.failure_kind = failure;
    report.milliseconds = elapsed(start); return report;
  };
  AdamStepPreview preview(optimizer, cuda_tf_groups);
  if (preview.parameters().empty()) {
    report.status = "NOOP"; return finish("no_current_gradients");
  }
  const bool finite_preview = preview.finite();
  report.preview_ms = elapsed(start);
  if (!finite_preview) return finish("nonfinite_adam", GuardFailureKind::NonFinite);
  if (evaluation.objectives.empty()) return finish("empty_guard_objectives", GuardFailureKind::InvalidState);
  const auto opts = preview.parameters().front().options().requires_grad(false);
  const std::size_t n = evaluation.objectives.size();

  auto evaluate = [&](const std::vector<torch::Tensor>* candidate) {
    torch::NoGradGuard no_grad;
    Blocks losses;
    for (std::size_t j = 0; j < n; ++j) {
      auto loss = evaluation.evaluate(j, preview, candidate);
      TORCH_CHECK(loss.numel() == 1, "Guard objectives must be scalar");
      losses.push_back(loss.detach());
    }
    return values(losses);
  };
  const auto baseline_start = Clock::now();
  report.before = evaluate(nullptr);
  report.baseline_ms = elapsed(baseline_start);
  if (!finite(report.before)) return finish("nonfinite_baseline", GuardFailureKind::NonFinite);

  bool candidate_nonfinite = false;
  auto commit = [&](const Blocks& candidate, double alpha, const char* reason, bool fallback) {
    report.update_ratio_by_group = preview.updateRatios(candidate);
    preview.commit(candidate);
    report.committed = true; report.committed_after = report.after;
    report.alpha = alpha; report.reason = reason;
    report.fallback_committed = fallback;
    report.status = fallback ? "FALLBACK" : alpha == 1.0 ? "ACCEPT" : "SHRINK";
  };

  // Raw candidate first: most harmless updates need no history backward or QP.
  // Every candidate uses the same live population/noise and fresh scratch weights.
  auto validate = [&](const Blocks& direction, double alpha, const char* reason) {
    torch::NoGradGuard no_grad;
    const auto validation_start = Clock::now();
    auto candidate = preview.candidate(direction, alpha);
    if (!preview.finiteCandidate(candidate)) {
      candidate_nonfinite = true;
      report.validation_ms += elapsed(validation_start); return false;
    }
    if (!preview.changed(candidate)) {
      report.status = "NOOP"; report.reason = "zero_update";
      report.after = report.before;
      report.validation_ms += elapsed(validation_start); return false;
    }
    ++report.candidates;
    if (evaluation.prepare_candidate) evaluation.prepare_candidate(preview, candidate);
    report.after = evaluate(&candidate);
    report.last_checked_after = report.after;
    if (report.candidates == 1) report.raw_after = report.after;
    candidate_nonfinite = !finite(report.after);
    bool accepted = finite(report.after);
    for (std::size_t j = 0; j < n && accepted; ++j)
      accepted = report.after[j] <= report.before[j] + 1e-7 + 1e-5 * std::abs(report.before[j]);
    if (!accepted) {
      report.validation_ms += elapsed(validation_start); return false;
    }
    commit(candidate, alpha, reason, false);
    report.validation_ms += elapsed(validation_start);
    return true;
  };
  if (validate(preview.delta(), 1.0, "raw_validated") || report.status == "NOOP")
    return finish(report.reason);
  if (candidate_nonfinite) return finish("nonfinite_candidate", GuardFailureKind::NonFinite);
  auto cancel = [&](const char* reason, GuardFailureKind failure = GuardFailureKind::QualityConflict) {
    report.would_cancel = true;
    if (!config_.cancel_fallback_enable) return finish(reason, failure);
    report.fallback_attempted = true;
    torch::NoGradGuard no_grad;
    const auto check_start = Clock::now();
    auto candidate = preview.candidate(preview.delta(), config_.cancel_fallback_scale);
    if (!preview.finiteCandidate(candidate)) {
      report.validation_ms += elapsed(check_start);
      return finish("nonfinite_fallback", GuardFailureKind::NonFinite);
    }
    if (!preview.changed(candidate)) {
      report.status = "NOOP"; report.after = report.before;
      report.validation_ms += elapsed(check_start);
      return finish("zero_fallback", failure);
    }
    ++report.fallback_checks;
    if (evaluation.prepare_candidate) evaluation.prepare_candidate(preview, candidate);
    report.after = evaluate(&candidate);
    report.validation_ms += elapsed(check_start);
    if (!finite(report.after)) return finish("nonfinite_fallback_loss", GuardFailureKind::NonFinite);
    commit(candidate, config_.cancel_fallback_scale, reason, true);
    return finish(reason, failure);
  };
  if (report.candidates >= config_.max_candidate_checks)
    return cancel("historical_regression");

  const auto history_start = Clock::now();
  std::vector<Blocks> gradients;
  // Fresh graphs; autograd::grad never writes normal .grad or retains rasterizer
  // buffers for another objective. These backwards only run after raw rejection.
  {
    torch::AutoGradMode enable_grad(true);
    Blocks history_losses;
    for (std::size_t j = 0; j < n; ++j) {
      auto loss = evaluation.evaluate(j, preview, nullptr);
      history_losses.push_back(loss.detach());
      gradients.push_back(gradient(loss, preview));
    }
    if (!finite(values(history_losses))) {
      report.history_ms = elapsed(history_start); return finish("nonfinite_history_loss", GuardFailureKind::NonFinite);
    }
  }
  report.history_ms = elapsed(history_start);
  torch::NoGradGuard no_grad;
  const auto projection_start = Clock::now();
  Blocks statistics;
  for (std::size_t i = 0; i < n; ++i) {
    statistics.push_back(dot(gradients[i], preview.delta(), opts));
    for (std::size_t j = 0; j <= i; ++j)
      statistics.push_back(dot(gradients[i], gradients[j], opts));
  }
  const auto packed = values(statistics);
  report.projection_ms = elapsed(projection_start);
  if (!finite(packed)) return finish("nonfinite_history_gradient", GuardFailureKind::NonFinite);
  std::vector<double> ad(n);
  std::vector<std::vector<double>> gram(n, std::vector<double>(n));
  std::size_t offset = 0;
  for (std::size_t i = 0; i < n; ++i) {
    ad[i] = packed[offset++];
    for (std::size_t j = 0; j <= i; ++j) gram[i][j] = gram[j][i] = packed[offset++];
  }
  std::vector<std::size_t> rows;
  std::vector<double> norms(n);
  for (std::size_t i = 0; i < n; ++i) {
    norms[i] = std::sqrt(std::max(0.0, gram[i][i]));
    if (norms[i] >= 1e-15) rows.push_back(i);
  }
  std::vector<std::vector<double>> h(rows.size(), std::vector<double>(rows.size()));
  std::vector<double> r(rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto a = rows[i];
    r[i] = ad[a] / norms[a];
    if (r[i] > 1e-7) ++report.conflicts;
    for (std::size_t j = 0; j < rows.size(); ++j)
      h[i][j] = gram[a][rows[j]] / (norms[a] * norms[rows[j]]);
  }
  const auto dual = solveConstraintDual(h, r);
  report.projection_converged = dual.converged;
  report.projection_violation = dual.violation;
  auto direction = preview.delta();
  for (auto& block : direction)
    block = block.scalar_type() == torch::kFloat64 ? block.clone() : block.to(torch::kFloat64);
  bool projected = false;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const double coefficient = dual.multipliers[i] / norms[rows[i]];
    if (coefficient == 0.0) continue;
    projected = true;
    for (std::size_t b = 0; b < direction.size(); ++b)
      if (gradients[rows[i]][b].defined())
        direction[b].add_(gradients[rows[i]][b].to(torch::kFloat64), -coefficient);
  }
  for (std::size_t b = 0; b < direction.size(); ++b)
    direction[b] = direction[b].to(preview.delta()[b].scalar_type());
  // Primal feasibility of the actual FP32 direction, not dual optimality,
  // determines whether real candidate validation is allowed.
  statistics.clear();
  for (const auto& g : gradients) statistics.push_back(dot(g, direction, opts));
  const auto actual_ad = values(statistics);
  report.projection_ms = elapsed(projection_start);
  if (!finite(actual_ad)) return finish("nonfinite_projection", GuardFailureKind::NonFinite);
  for (auto i : rows)
    if (actual_ad[i] / norms[i] > 1e-6)
      return cancel(dual.converged ? "projection_residual" : "projection_not_converged",
                    GuardFailureKind::FiniteProjectionFailure);
  report.projection_ms = elapsed(projection_start);
  constexpr double alphas[] = {1.0, 0.5, 0.25};
  for (double alpha : alphas) {
    if (alpha == 1.0 && !projected) continue;  // Raw candidate already failed.
    if (report.candidates >= config_.max_candidate_checks) break;
    if (validate(direction, alpha, projected ? "projected_validated" : "raw_shrunk_validated") ||
        report.status == "NOOP")
      return finish(report.reason);
    if (candidate_nonfinite) return finish("nonfinite_candidate", GuardFailureKind::NonFinite);
  }
  return cancel("historical_regression");
}

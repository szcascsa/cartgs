#include "include/gradient_constraint.h"
#include <stdexcept>
#include <limits>
#include <cmath>

namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void shrinkAndCancel(double lr, const char* expected) {
  auto p = torch::ones({1}, torch::TensorOptions().requires_grad(true));
  auto absent = torch::ones({1}, torch::TensorOptions().requires_grad(true));
  torch::optim::Adam optimizer({p, absent}, torch::optim::AdamOptions(lr));
  p.mutable_grad() = torch::ones_like(p);
  const auto before = p.detach().clone(), normal_grad = p.grad().clone();
  GuardEvaluation e;
  e.objectives = {{"full"}, {"lod"}};
  e.evaluate = [&](std::size_t, const AdamStepPreview& preview, const std::vector<torch::Tensor>* candidate) {
    const auto value = candidate ? preview.value(p, *candidate) : p;
    return value.square().mean() + absent.square().mean();
  };
  auto r = GradientConstraintManager().step(optimizer, e);
  check(r.status == expected, "Wrong finite-step acceptance/shrink/cancel decision");
  check(r.candidates <= 3, "Guard exceeded configured candidate render budget");
  check(p.grad().equal(normal_grad) && !absent.grad().defined(), "Historical gradients polluted .grad");
  check(optimizer.state().count(absent.unsafeGetTensorImpl()) == 0, "History updated grad=None parameter");
  if (!r.committed) check(p.equal(before) && optimizer.state().empty(), "Rejected transaction was committed");
  else {
    check(r.alpha == 0.5 && r.after[0] <= r.before[0], "Backtracking failed to protect actual loss");
    const auto& state = static_cast<const torch::optim::AdamParamState&>(*optimizer.state().at(p.unsafeGetTensorImpl()));
    check(state.step() == 1, "Accepted shrink advanced Adam more than once");
  }
}
void rawAcceptAndBudgetDrift() {
  auto p = torch::ones({1}, torch::TensorOptions().requires_grad(true));
  torch::optim::Adam optimizer({p}, torch::optim::AdamOptions(0.1));
  p.mutable_grad() = torch::ones_like(p);
  int historical_graphs = 0;
  GuardEvaluation e;
  e.objectives = {{"full"}, {"lod"}};
  e.evaluate = [&](std::size_t, const AdamStepPreview& preview, const std::vector<torch::Tensor>* candidate) {
    if (torch::GradMode::is_enabled()) ++historical_graphs;
    const auto value = candidate ? preview.value(p, *candidate) : p;
    return value.square().mean();
  };
  // A synthetic count deviation grows as p decreases; it is NOT a guard objective.
  const auto budget_before = (1.0 - p.detach()).abs().item<double>();
  const auto r = GradientConstraintManager().step(optimizer, e);
  check(r.committed && r.status == "ACCEPT" && r.reason == "raw_validated" && r.candidates == 1,
        "Harmless raw update did not take the fast path");
  check(historical_graphs == 0 && r.history_ms == 0.0 && r.projection_ms == 0.0,
        "Raw acceptance still computed historical autograd/projection");
  check((1.0 - p.detach()).abs().item<double>() > budget_before,
        "Test did not exercise a quality-improving count deviation");
}
void projectedQualityAndZeroLR(torch::ScalarType dtype) {
  const auto options = torch::TensorOptions().dtype(dtype).requires_grad(true);
  auto p = torch::ones({2}, options);
  auto frozen = torch::ones({1}, options);
  torch::optim::Adam optimizer({p}, torch::optim::AdamOptions(0.1));
  optimizer.add_param_group(std::vector<torch::Tensor>{frozen});
  optimizer.param_groups()[1].options().set_lr(0.0);
  p.mutable_grad() = torch::tensor({-1.0f, 1.0f}, options.requires_grad(false));
  frozen.mutable_grad() = torch::ones_like(frozen);
  GuardEvaluation e; e.objectives = {{"full"}, {"lod"}};
  e.evaluate = [&](std::size_t, const AdamStepPreview& preview, const std::vector<torch::Tensor>* candidate) {
    const auto value = candidate ? preview.value(p, *candidate) : p;
    return value[0].square();
  };
  const auto r = GradientConstraintManager().step(optimizer, e);
  check(r.committed && r.reason == "projected_validated" && r.conflicts > 0 &&
            p[0].item<float>() <= 1.000001f && p[1].item<float>() < 1.0f,
        "Projection did not remove the harmful component while retaining learning");
  check(frozen.item<float>() == 1.0f && r.update_ratio_by_group[1] == 0.0,
        "Projection moved a zero-LR parameter");
  check(std::abs(r.update_ratio_by_group[0] - std::sqrt(0.5)) < 1e-5,
        "Projection mutated the original Adam direction (FP64 aliasing)");
  auto& state = static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(p.unsafeGetTensorImpl()));
  const auto old_moment = state.exp_avg().clone(); const auto old_step = state.step();
  const auto old_p = p.clone();
  e.evaluate = [&](std::size_t, const AdamStepPreview&, const std::vector<torch::Tensor>*) {
    return torch::full({}, std::numeric_limits<float>::quiet_NaN());
  };
  const auto cancelled = GradientConstraintManager().step(optimizer, e);
  check(!cancelled.committed && cancelled.status == "CANCEL" && p.equal(old_p) &&
            state.step() == old_step && state.exp_avg().equal(old_moment),
        "Nonfinite baseline changed an existing Adam transaction");
}
void noOpDoesNotCommit() {
  auto p = torch::ones({1}, torch::TensorOptions().requires_grad(true));
  torch::optim::Adam optimizer({p}, torch::optim::AdamOptions(0.1));
  p.mutable_grad() = -torch::ones_like(p);
  GuardEvaluation e; e.objectives = {{"full"}};
  e.evaluate = [&](std::size_t, const AdamStepPreview& preview, const std::vector<torch::Tensor>* candidate) {
    return (candidate ? preview.value(p, *candidate) : p).square().mean();
  };
  auto r = GradientConstraintManager().step(optimizer, e);
  check(r.status == "NOOP" && !r.committed && p.item<float>() == 1.0f && optimizer.state().empty(),
        "Zero projected update was counted/committed as learning");
  p.mutable_grad() = torch::zeros_like(p);
  r = GradientConstraintManager().step(optimizer, e);
  check(r.status == "NOOP" && !r.committed && optimizer.state().empty(),
        "Zero raw update advanced Adam state");
  p.mutable_grad() = torch::Tensor();
  r = GradientConstraintManager().step(optimizer, e);
  check(r.status == "NOOP" && !r.committed, "No-gradient update was reported as a cancellation");
}

void fallbackUsesOriginalAdamOnce() {
  auto p = torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto absent = torch::ones({1}, p.options());
  torch::optim::Adam optimizer({p, absent}, torch::optim::AdamOptions(12.0));
  p.mutable_grad() = torch::ones_like(p);
  AdamStepPreview reference(optimizer);
  const auto expected = reference.candidate(reference.delta(), 0.2)[0];
  GuardEvaluation evaluation; evaluation.objectives = {{"full"}, {"lod"}};
  evaluation.evaluate = [&](std::size_t, const AdamStepPreview& preview, const std::vector<torch::Tensor>* candidate) {
    return (candidate ? preview.value(p, *candidate) : p).square().mean();
  };
  GradientConstraintConfig config; config.cancel_fallback_enable = true;
  const auto result = GradientConstraintManager(config).step(optimizer, evaluation);
  check(result.status == "FALLBACK" && result.would_cancel && result.fallback_attempted &&
            result.fallback_committed && result.committed && result.fallback_checks == 1 &&
            result.candidates == 3 && torch::allclose(p, expected),
        "Fallback did not commit 0.2 of the ORIGINAL Adam displacement exactly once");
  check(result.raw_after[0] > result.last_checked_after[0] &&
            result.committed_after[0] == result.after[0] && result.after[0] > result.before[0],
        "Raw/rejected/committed diagnostics were confused or fallback still enforced quality gate");
  const auto& state = static_cast<const torch::optim::AdamParamState&>(*optimizer.state().at(p.unsafeGetTensorImpl()));
  check(state.step() == 1 && torch::allclose(state.exp_avg(), torch::full_like(p, 0.1)),
        "Fallback scaled gradients/moments or advanced Adam more than once");
  check(optimizer.state().count(absent.unsafeGetTensorImpl()) == 0 &&
            std::abs(result.update_ratio_by_group[0] - 0.2) < 1e-8,
        "Fallback changed grad=None parameters or reported the wrong displacement");
}

void numericalFailuresNeverFallback() {
  auto p = torch::ones({1}, torch::TensorOptions().requires_grad(true));
  torch::optim::Adam optimizer({p}, torch::optim::AdamOptions(12.0));
  p.mutable_grad() = torch::ones_like(p);
  GuardEvaluation evaluation; evaluation.objectives = {{"full"}};
  evaluation.evaluate = [&](std::size_t, const AdamStepPreview&, const std::vector<torch::Tensor>* candidate) {
    if (candidate) return torch::full({}, std::numeric_limits<float>::quiet_NaN());
    return p.square().mean();
  };
  GradientConstraintConfig config; config.cancel_fallback_enable = true;
  auto result = GradientConstraintManager(config).step(optimizer, evaluation);
  check(result.status == "CANCEL" && result.failure_kind == GuardFailureKind::NonFinite &&
            !result.would_cancel && !result.fallback_attempted && !result.committed &&
            p.item<float>() == 1.0f && optimizer.state().empty(),
        "Nonfinite raw candidate was softened into fallback/source statistics");
  // A finite would-CANCEL may be counted, but a nonfinite fallback must never commit.
  config.max_candidate_checks = 1;
  evaluation.evaluate = [&](std::size_t, const AdamStepPreview& preview, const std::vector<torch::Tensor>* candidate) {
    const auto value = candidate ? preview.value(p, *candidate) : p;
    if (candidate && value.item<float>() > -2.0f)
      return torch::full({}, std::numeric_limits<float>::quiet_NaN());
    return value.square().mean();
  };
  result = GradientConstraintManager(config).step(optimizer, evaluation);
  check(result.would_cancel && result.fallback_attempted && !result.committed &&
            result.failure_kind == GuardFailureKind::NonFinite && p.item<float>() == 1.0f &&
            optimizer.state().empty(), "Nonfinite fallback committed parameter or Adam state");
  p.mutable_grad() = torch::full_like(p, std::numeric_limits<float>::infinity());
  result = GradientConstraintManager(config).step(optimizer, evaluation);
  check(!result.would_cancel && !result.fallback_attempted && !result.committed && optimizer.state().empty(),
        "Nonfinite original Adam gradient entered fallback");
}
}
int main() {
  shrinkAndCancel(3.0, "SHRINK"); shrinkAndCancel(12.0, "CANCEL");
  rawAcceptAndBudgetDrift();
  projectedQualityAndZeroLR(torch::kFloat32); projectedQualityAndZeroLR(torch::kFloat64);
  noOpDoesNotCommit();
  fallbackUsesOriginalAdamOnce();
  numericalFailuresNeverFallback();
}

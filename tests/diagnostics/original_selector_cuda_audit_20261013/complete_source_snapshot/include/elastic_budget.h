#pragma once
#include "gaussian_protection.h"

struct ElasticBudgetPopulation {
  torch::Tensor mature, need;
  int64_t mature_count = 0;
};
struct ElasticBudgetPlan {
  ElasticBudgetAllocation target;
  torch::Tensor admitted, probability;
  int64_t protected_count = 0;
  float mature_ratio = 0.0f;
};

// Cached uniforms give nested weighted admission at fixed evidence and state.
class ElasticBudgetManager {
 public:
  std::shared_ptr<ProtectionProfiler> profiler;
  bool performance_logging = true;
  void reset(int64_t count, const torch::TensorOptions& options) {
    torch::NoGradGuard guard;
    uniform_ = torch::rand({count}, options.dtype(torch::kFloat32).requires_grad(false));
  }
  void append(int64_t count, const torch::TensorOptions& options) {
    torch::NoGradGuard guard;
    if (!uniform_.defined()) { reset(count, options); return; }
    uniform_ = torch::cat({uniform_, torch::rand({count}, uniform_.options())});
  }
  void prune(const torch::Tensor& survivors) {
    torch::NoGradGuard guard;
    TORCH_CHECK(uniform_.defined() && uniform_.numel() == survivors.numel(), "Protection uniforms are not topology-aligned");
    uniform_ = uniform_.index({survivors});
  }
  void refresh() { torch::NoGradGuard guard; if (uniform_.defined()) uniform_ = torch::rand_like(uniform_); }
  const torch::Tensor& uniform() const { return uniform_; }
  void setUniform(const torch::Tensor& value) { uniform_ = value.detach().clone(); }
  void useUniform(const torch::Tensor& value) { uniform_ = value; }
  ElasticBudgetPopulation population(const torch::Tensor& mature, const torch::Tensor& need,
                                    int64_t known_mature_count = -1) const {
    torch::NoGradGuard guard;
    TORCH_CHECK(uniform_.defined() && uniform_.numel() == mature.numel() && need.numel() == mature.numel(),
                "Protection budget population is not aligned");
    return {mature, need * (~mature).to(need.scalar_type()),
        known_mature_count < 0 ? mature.sum().item<int64_t>() : known_mature_count};
  }
  ElasticBudgetPlan plan(const ElasticBudgetPopulation& population, float ratio,
                        const ProtectionConfig& config) const {
    torch::NoGradGuard guard;
    TORCH_CHECK(std::isfinite(ratio) && ratio >= 0 && ratio <= 1, "Invalid global LoD ratio");
    config.validate();
    auto* timer = performance_logging ? profiler.get() : nullptr;
    ElasticBudgetPlan result;
    result.target.total = static_cast<double>(ratio) * population.mature.numel();
    torch::Tensor demand;
    {
      ProtectionProfiler::Scope budget(timer, ProtectionProfiler::Budget, uniform_.is_cuda());
      demand = population.need.sum();
      // Bp <= D, hence lambda <= 1. Clipping never activates for P in [0,1];
      // this is the exact solution, with no bisection or CPU Gaussian loop.
      const auto protection_budget = demand.clamp_max(config.max_budget_fraction * result.target.total);
      result.probability = population.need * (protection_budget / demand.clamp_min(1e-12));
    }
    {
      ProtectionProfiler::Scope admission(timer, ProtectionProfiler::Admission, uniform_.is_cuda());
      result.admitted = ~population.mature & (uniform_ < result.probability);
      // The actual sample is required by the remaining Mature quota. Read the
      // demand and sample size in one transfer, instead of synchronizing twice.
      const auto stats = torch::stack({demand, result.admitted.sum().to(demand.scalar_type())}).cpu();
      result.target.protection_demand = stats[0].item<double>();
      result.protected_count = stats[1].item<int64_t>();
    }
    result.target.protection = computeProtectionBudget(result.target.protection_demand,
        result.target.total, config.max_budget_fraction);
    result.target.mature = computeMatureTarget(result.target.total, result.protected_count, population.mature_count);
    result.mature_ratio = population.mature_count ? result.target.mature / population.mature_count : 0;
    return result;
  }
 private:
  torch::Tensor uniform_;
};

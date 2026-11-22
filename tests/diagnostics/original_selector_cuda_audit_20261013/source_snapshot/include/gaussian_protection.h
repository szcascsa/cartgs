#pragma once
#include "elastic_budget_allocation.h"
#include "protection_profiler.h"
#include "protection_evidence_cuda.h"
#include <torch/torch.h>
#include <limits>
#include <map>

enum class GaussianProtectionState { WARMUP, UNDER_OBSERVED, STALE, MATURE };

struct ProtectionClassification {
  torch::Tensor age, age_confidence, optimization_confidence, keyframe_confidence;
  torch::Tensor maturity, need, state, mature;
};

// Evidence belongs to Gaussian rows. Only normal Full Mapping opens and commits
// a transaction here. Auxiliary optimizers cannot increment these counters.
class GaussianProtection {
 public:
  ProtectionConfig config;
  int64_t mapping_iteration = 0;
  torch::Tensor birth_iter, optimized_count, unique_keyframe_count, supporting_keyframe_ids, cached_uniform;
  std::shared_ptr<ProtectionProfiler> profiler = std::make_shared<ProtectionProfiler>();
  torch::Tensor transitions;

  ProtectionProfiler* timing() const { return config.performance_logging ? profiler.get() : nullptr; }
  void reset(int64_t count, const torch::TensorOptions& options) {
    torch::NoGradGuard guard;
    config.validate();
    auto longs = options.dtype(torch::kInt64).requires_grad(false);
    auto ints = options.dtype(torch::kInt32).requires_grad(false);
    birth_iter = torch::full({count}, mapping_iteration, longs);
    optimized_count = torch::zeros({count}, ints);
    unique_keyframe_count = torch::zeros({count}, ints);
    supporting_keyframe_ids = torch::full({count, config.unique_keyframe_ready}, -1, longs);
    cached_uniform = torch::rand({count}, options.dtype(torch::kFloat32).requires_grad(false));
    pending_visibility_ = torch::Tensor();
    transitions = torch::zeros({5}, longs);
  }
  void assertAligned(int64_t count) const {
    TORCH_CHECK(birth_iter.numel() == count && optimized_count.numel() == count &&
        unique_keyframe_count.numel() == count && supporting_keyframe_ids.dim() == 2 &&
        supporting_keyframe_ids.size(0) == count &&
        supporting_keyframe_ids.size(1) == config.unique_keyframe_ready && cached_uniform.numel() == count,
        "Protection metadata is not Gaussian-topology aligned");
  }
  void append(int64_t count, const torch::Tensor& parents = {}) {
    torch::NoGradGuard guard;
    auto unique = parents.defined() ? unique_keyframe_count.index_select(0, parents)
        : torch::zeros({count}, unique_keyframe_count.options());
    auto support = parents.defined() ? supporting_keyframe_ids.index_select(0, parents)
        : torch::full({count, config.unique_keyframe_ready}, -1, supporting_keyframe_ids.options());
    TORCH_CHECK(unique.numel() == count, "Protection parent mapping size mismatch");
    birth_iter = torch::cat({birth_iter, torch::full({count}, mapping_iteration, birth_iter.options())});
    optimized_count = torch::cat({optimized_count, torch::zeros({count}, optimized_count.options())});
    unique_keyframe_count = torch::cat({unique_keyframe_count, unique});
    supporting_keyframe_ids = torch::cat({supporting_keyframe_ids, support});
    cached_uniform = torch::cat({cached_uniform, torch::rand({count}, cached_uniform.options())});
  }
  void prune(const torch::Tensor& survivors) {
    torch::NoGradGuard guard;
    assertAligned(survivors.numel());
    birth_iter = birth_iter.index({survivors});
    optimized_count = optimized_count.index({survivors});
    unique_keyframe_count = unique_keyframe_count.index({survivors});
    supporting_keyframe_ids = supporting_keyframe_ids.index({survivors});
    cached_uniform = cached_uniform.index({survivors});
    if (pending_visibility_.defined()) {
      const auto retained = survivors.index_select(0, pending_visibility_);
      const auto mapping = survivors.to(torch::kInt64).cumsum(0) - 1;
      pending_visibility_ = mapping.index_select(0, pending_visibility_).masked_select(retained);
    }
  }
  ProtectionClassification classify(int64_t iteration) const {
    torch::NoGradGuard guard;
    ProtectionProfiler::Scope timing_scope(timing(), ProtectionProfiler::Maturity, birth_iter.is_cuda());
    ProtectionClassification c;
    c.age = (iteration - birth_iter).clamp_min(0);
    c.age_confidence = (c.age.to(torch::kFloat32) / config.age_ready).clamp(0, 1);
    c.optimization_confidence = (optimized_count.to(torch::kFloat32) / config.optimization_ready).clamp(0, 1);
    c.keyframe_confidence = (unique_keyframe_count.to(torch::kFloat32) / config.unique_keyframe_ready).clamp(0, 1);
    c.maturity = torch::minimum(torch::minimum(c.age_confidence, c.optimization_confidence), c.keyframe_confidence);
    c.need = 1 - c.maturity;
    c.mature = (c.age >= config.age_ready) & (optimized_count >= config.optimization_ready) &
        (unique_keyframe_count >= config.unique_keyframe_ready);
    c.state = torch::full_like(optimized_count, int(GaussianProtectionState::UNDER_OBSERVED));
    c.state.masked_fill_(c.age >= config.stale_age, int(GaussianProtectionState::STALE));
    c.state.masked_fill_(c.age < config.age_ready, int(GaussianProtectionState::WARMUP));
    c.state.masked_fill_(c.mature, int(GaussianProtectionState::MATURE));
    return c;
  }
  ProtectionClassification classify() const { return classify(mapping_iteration); }
  torch::Tensor matureMask() const {
    torch::NoGradGuard guard;
    return (mapping_iteration - birth_iter >= config.age_ready) &
        (optimized_count >= config.optimization_ready) & (unique_keyframe_count >= config.unique_keyframe_ready);
  }
  void advance(int64_t iteration) {
    ProtectionProfiler::Scope timing_scope(timing(), ProtectionProfiler::Maturity, birth_iter.is_cuda());
    TORCH_CHECK(iteration >= mapping_iteration, "Protection mapping clock must be monotonic");
    if (iteration == mapping_iteration) return;
    if (birth_iter.is_cuda()) {
      advanceProtectionClockCUDA(birth_iter, optimized_count, unique_keyframe_count, transitions,
          mapping_iteration, iteration, config.age_ready, config.optimization_ready,
          config.unique_keyframe_ready, config.stale_age);
      mapping_iteration = iteration;
      return;
    }
    if (birth_iter.defined() && birth_iter.numel()) {
      torch::NoGradGuard guard;
      const auto age_before = mapping_iteration - birth_iter, age_after = iteration - birth_iter;
      const auto ready = (optimized_count >= config.optimization_ready) &
          (unique_keyframe_count >= config.unique_keyframe_ready);
      const auto leaving_warmup = (age_before < config.age_ready) & (age_after >= config.age_ready);
      const auto becoming_stale = (age_before >= config.age_ready) & (age_before < config.stale_age) &
          (age_after >= config.stale_age) & ~ready;
      transitions += torch::stack({(leaving_warmup & ~ready & (age_after < config.stale_age)).sum(),
          (leaving_warmup & ready).sum(), torch::zeros({}, birth_iter.options()),
          becoming_stale.sum(), torch::zeros({}, birth_iter.options())});
    }
    mapping_iteration = iteration;
  }
  void beginFullMapping(const torch::Tensor& visible_indices) {
    torch::NoGradGuard guard;
    ProtectionProfiler::Scope timing_scope(timing(), ProtectionProfiler::Evidence, birth_iter.is_cuda());
    pending_visibility_ = visible_indices;
  }
  void finishFullMapping(bool committed, int64_t keyframe_id) {
    torch::NoGradGuard guard;
    ProtectionProfiler::Scope evidence(timing(), ProtectionProfiler::Evidence, birth_iter.is_cuda());
    if (!committed || !pending_visibility_.defined()) { pending_visibility_ = torch::Tensor(); return; }
    const auto visible = pending_visibility_;
    if (birth_iter.is_cuda()) {
      ProtectionProfiler::Scope unique_timing(timing(), ProtectionProfiler::UniqueKF, true);
      commitProtectionEvidenceCUDA(visible, birth_iter, optimized_count, unique_keyframe_count,
          supporting_keyframe_ids, transitions, mapping_iteration, keyframe_id,
          config.age_ready, config.optimization_ready, config.stale_age);
      pending_visibility_ = torch::Tensor();
      return;
    }
    const auto old_ready = (optimized_count.index_select(0, visible) >= config.optimization_ready) &
        (unique_keyframe_count.index_select(0, visible) >= config.unique_keyframe_ready);
    const auto opt = optimized_count.index_select(0, visible);
    optimized_count.index_copy_(0, visible, opt + (opt < std::numeric_limits<int32_t>::max()).to(torch::kInt32));
    {
      ProtectionProfiler::Scope unique_timing(timing(), ProtectionProfiler::UniqueKF, birth_iter.is_cuda());
      const auto counts = unique_keyframe_count.index_select(0, visible);
      const auto ids = supporting_keyframe_ids.index_select(0, visible);
      const auto add = ~ids.eq(keyframe_id).any(1) & (counts < config.unique_keyframe_ready);
      const auto local = torch::nonzero(add).reshape({-1});
      const auto rows = visible.index_select(0, local);
      supporting_keyframe_ids.index_put_({rows, counts.index_select(0, local).to(torch::kInt64)}, keyframe_id);
      unique_keyframe_count.index_copy_(0, rows, counts.index_select(0, local) + 1);
    }
    const auto became_ready = ~old_ready & (optimized_count.index_select(0, visible) >= config.optimization_ready) &
        (unique_keyframe_count.index_select(0, visible) >= config.unique_keyframe_ready);
    const auto age = mapping_iteration - birth_iter.index_select(0, visible);
    transitions.index_put_({2}, transitions[2] + (became_ready & (age >= config.age_ready) & (age < config.stale_age)).sum());
    transitions.index_put_({4}, transitions[4] + (became_ready & (age >= config.stale_age)).sum());
    pending_visibility_ = torch::Tensor();
  }
  double metadataMB() const {
    return (birth_iter.numel() * birth_iter.element_size() + optimized_count.numel() * optimized_count.element_size() +
        unique_keyframe_count.numel() * unique_keyframe_count.element_size() +
        supporting_keyframe_ids.numel() * supporting_keyframe_ids.element_size() +
        cached_uniform.numel() * cached_uniform.element_size()) / 1048576.0;
  }
  std::map<std::string, double> statistics(bool enabled = true) const {
    torch::NoGradGuard guard;
    auto c = classify();
    if (!enabled) {
      c.mature = torch::ones_like(c.mature);
      c.maturity = torch::ones_like(c.maturity);
      c.need = torch::zeros_like(c.need);
      c.state.fill_(int(GaussianProtectionState::MATURE));
    }
    const auto stale = c.state == int(GaussianProtectionState::STALE);
    const auto stale_f = stale.to(torch::kFloat64);
    const auto stale_n = stale_f.sum().clamp_min(1);
    auto mean = [](const torch::Tensor& value) {
      return value.to(torch::kFloat64).sum() / std::max<int64_t>(1, value.numel());
    };
    std::vector<std::string> names = {"warmup_gaussians", "under_observed_gaussians", "stale_gaussians", "mature_gaussians",
        "protected_gaussians", "mean_maturity_confidence", "mean_protection_need", "mean_age_confidence",
        "mean_optimization_confidence", "mean_keyframe_confidence", "mean_optimized_count", "mean_unique_keyframe_support",
        "stale_mean_age", "stale_mean_optimized_count", "stale_mean_unique_keyframes"};
    std::vector<torch::Tensor> values;
    for (int state = 0; state < 4; ++state) values.push_back((c.state == state).sum().to(torch::kFloat64));
    values.insert(values.end(), {(~c.mature).sum().to(torch::kFloat64), mean(c.maturity), mean(c.need),
        mean(c.age_confidence), mean(c.optimization_confidence), mean(c.keyframe_confidence),
        mean(optimized_count), mean(unique_keyframe_count),
        (c.age * stale_f).sum() / stale_n, (optimized_count * stale_f).sum() / stale_n,
        (unique_keyframe_count * stale_f).sum() / stale_n});
    const char* transition_names[] = {"warmup_to_under_observed", "warmup_to_mature", "under_observed_to_mature",
        "under_observed_to_stale", "stale_to_mature"};
    for (int i = 0; i < 5; ++i) { names.push_back(transition_names[i]); values.push_back(transitions[i].to(torch::kFloat64)); }
    const auto cpu = torch::stack(values).cpu();
    std::map<std::string, double> result;
    for (size_t i = 0; i < names.size(); ++i) result[names[i]] = cpu[i].item<double>();
    result["stale_count"] = result["stale_gaussians"];
    return result;
  }
  void report(std::ostream& out, bool enabled = true) const {
    out << "[Protection STATE] mapping_iter=" << mapping_iteration;
    for (const auto& entry : statistics(enabled)) out << ' ' << entry.first << '=' << entry.second;
    out << '\n';
    if (config.performance_logging) profiler->report(out, metadataMB());
  }
 private:
  torch::Tensor pending_visibility_;
};

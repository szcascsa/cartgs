#include "include/drift_stability.h"
#include "include/frozen_teacher_target_cache.h"
#include "include/historical_population.h"
#include "include/transform_field_cuda_optimizer.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>

namespace {

torch::Tensor selectorProbability(
    const improvements::selector::GumbelNetwork& network,
    const GaussianAttrState& state, const torch::Tensor& indices,
    float ratio, double temperature) {
  const auto selected = state.select(indices);
  return network.ptr()->forwardProbability(
      selected.xyz, selected.rotation, selected.log_scaling.exp(), ratio,
      temperature);
}

double weightedTF(const std::array<double, 3>& values,
                  const DriftStabilityConfig& config) {
  return config.position_weight * values[0] +
         config.scale_weight * values[1] +
         config.rotation_weight * values[2];
}

std::array<double, 3> asArray(const torch::Tensor& tensor) {
  const auto host = tensor.to(torch::kCPU);
  return {host[0].item<double>(), host[1].item<double>(),
          host[2].item<double>()};
}

void blendReference(const torch::nn::Module& old, torch::nn::Module& candidate,
                    double decay) {
  torch::NoGradGuard no_grad;
  const auto previous = old.named_parameters();
  for (auto& item : candidate.named_parameters()) {
    TORCH_CHECK(previous[item.key()].sizes() == item.value().sizes(),
                "Stability EMA parameter shape mismatch");
    item.value().mul_(1.0 - decay).add_(previous[item.key()], decay);
  }
  // cloneFrozen already copied buffers (including TF domain) and disabled grads.
  candidate.eval();
}

}  // namespace

DriftStabilityManager::DriftStabilityManager(DriftStabilityConfig config)
    : config_(std::move(config)), rng_(config_.seed) {
  TORCH_CHECK(config_.probe_interval > 0 &&
                  config_.min_bad_views > 0 && config_.max_gaussians > 0 &&
                  config_.batch_size > 0 &&
                  config_.consecutive_trigger_count > 0 &&
                  config_.reference_min_age >= 0 &&
                  config_.reference_horizon > 0 &&
                  config_.reference_max_age > 0 &&
                  config_.reference_max_age >= config_.reference_min_age &&
                  config_.reference_max_age >= config_.reference_horizon &&
                  config_.non_regression_refresh_count > 0 &&
                  config_.epochs > 0,
              "Invalid Historical-Outlier Distill integer configuration");
  TORCH_CHECK(std::isfinite(config_.reference_ema_decay) &&
                  config_.reference_ema_decay >= 0.0f &&
                  config_.reference_ema_decay < 1.0f,
              "Stability reference EMA decay must be in [0,1)");
  for (float value : {config_.function_threshold,
                      config_.psnr_regression_threshold,
                      config_.budget_relative_tolerance,
                      config_.position_weight, config_.scale_weight,
                      config_.rotation_weight, config_.selector_learning_rate,
                      config_.tf_grid_learning_rate,
                      config_.tf_mlp_learning_rate})
    TORCH_CHECK(std::isfinite(value) && value >= 0.0f,
                "Invalid Historical-Outlier Distill numeric configuration");
  TORCH_CHECK(config_.selector_learning_rate > 0.0f &&
                  config_.tf_grid_learning_rate > 0.0f &&
                  config_.tf_mlp_learning_rate > 0.0f,
              "Stability Distill learning rates must be positive");
}

bool DriftStabilityManager::shouldProbe(uint64_t mapping_iter) const {
  return config_.enabled && mapping_iter > 0 &&
         mapping_iter % static_cast<uint64_t>(config_.probe_interval) == 0;
}

void DriftStabilityManager::refreshReference(
    const improvements::selector::GumbelNetwork& selector,
    const TransformField& tf, uint64_t mapping_iter, uint64_t change_version,
    uint64_t skip_probe_iter, bool refresh_selector, bool refresh_tf,
    const char* reason) {
  if (!config_.enabled) return;
  torch::NoGradGuard no_grad;
  if (refresh_selector) {
    reference_.selector_teacher = nullptr;
    if (selector)
      reference_.selector_teacher =
          improvements::selector::GumbelNetwork(selector->cloneFrozen());
    reference_.selector_reference_iter = mapping_iter;
    reference_.selector_full_reset_iter = mapping_iter;
    selector_over_count_.clear();
    selector_stable_count_ = 0;
  }
  if (refresh_tf) {
    reference_.tf_teacher = nullptr;
    if (tf) reference_.tf_teacher = TransformField(tf->cloneFrozen());
    reference_.tf_reference_iter = mapping_iter;
    reference_.tf_full_reset_iter = mapping_iter;
    tf_over_count_.clear();
    tf_stable_count_ = 0;
  }
  reference_.change_version = change_version;
  if (skip_probe_iter) skip_probe_iter_ = skip_probe_iter;
  std::cout << "[HistoricalDistill HARD_RESET] iter=" << mapping_iter
            << " change_version=" << change_version
            << " selector=" << (refresh_selector && bool(selector))
            << " tf=" << (refresh_tf && bool(tf))
            << " reason=" << reason << std::endl;
}

bool DriftStabilityManager::prepareProbe(
    const improvements::selector::GumbelNetwork& selector,
    const TransformField& tf, uint64_t mapping_iter, uint64_t change_version,
    bool deferred) {
  if (!config_.enabled || (!deferred && !shouldProbe(mapping_iter))) return false;
  if (skip_probe_iter_ == mapping_iter) {
    skip_probe_iter_ = 0;
    std::cout << "[HistoricalProbe SKIP] iter=" << mapping_iter
              << " reason=structural_change_same_iteration" << std::endl;
    return false;
  }
  if (skip_probe_iter_ < mapping_iter) skip_probe_iter_ = 0;
  if (reference_.change_version != change_version) {
    refreshReference(selector, tf, mapping_iter, change_version, 0, true, true,
                     "version_mismatch");
    return false;
  }
  const bool empty = !reference_.selector_teacher && !reference_.tf_teacher;
  const bool missing_selector = selector && !reference_.selector_teacher;
  const bool missing_tf = tf && !reference_.tf_teacher;
  if (missing_selector || missing_tf)
    refreshReference(selector, tf, mapping_iter, change_version, 0,
                     missing_selector, missing_tf, "initialized");
  return !empty;
}

void DriftStabilityManager::clearProbeState() {
  selector_over_count_.clear();
  tf_over_count_.clear();
  selector_stable_count_ = 0;
  tf_stable_count_ = 0;
}

torch::Tensor DriftStabilityManager::shuffle(const torch::Tensor& indices) {
  if (!indices.defined() || indices.numel() <= 1) return indices;
  std::vector<int64_t> order(indices.numel());
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), rng_);
  auto permutation = torch::tensor(
      order, torch::TensorOptions().dtype(torch::kLong)).to(indices.device());
  return indices.index_select(0, permutation);
}

torch::Tensor DriftStabilityManager::selectorLoss(
    const torch::Tensor& indices, const GaussianAttrState& current,
    const improvements::selector::GumbelNetwork& student,
    const improvements::selector::GumbelNetwork& teacher, float ratio,
    double temperature, FrozenTeacherTargetCache* targets) const {
  const auto selected = current.select(indices);
  const auto logits = student.ptr()->forwardKeepLogit(
      selected.xyz, selected.rotation, selected.log_scaling.exp(), ratio, temperature);
  torch::Tensor target;
  {
    torch::NoGradGuard no_grad;
    auto predict = [&](const torch::Tensor& missing) {
      return selectorProbability(teacher, current, missing, ratio, temperature);
    };
    target = targets ? targets->get(indices, predict) : predict(indices);
  }
  return torch::nn::functional::binary_cross_entropy_with_logits(logits, target);
}

torch::Tensor DriftStabilityManager::tfLossTensor(
    const torch::Tensor& indices, const GaussianAttrState& current,
    const TransformField& student, const TransformField& teacher,
    float ratio, double diagonal, FrozenTeacherTargetCache* targets) const {
  const auto selected = current.select(indices);
  GaussianAttrState target;
  {
    torch::NoGradGuard no_grad;
    if (targets) {
      auto packed = targets->get(indices, [&](const torch::Tensor& missing) {
        auto output = teacher.ptr()->applyLogResidual(current.select(missing), ratio);
        return torch::cat({output.xyz, output.log_scaling, output.rotation}, 1);
      });
      target = {packed.slice(1, 0, 3), packed.slice(1, 3, 6), packed.slice(1, 6, 10)};
    } else {
      target = teacher.ptr()->applyLogResidual(selected, ratio);
    }
  }
  const auto output = student.ptr()->applyLogResidual(selected, ratio);
  auto position = (output.xyz - target.xyz).abs().mean() /
                  std::max(diagonal, 1e-6);
  auto scale = (output.log_scaling - target.log_scaling).abs().mean();
  auto dot = (output.rotation * target.rotation).sum(1).abs().clamp(0.0, 1.0);
  return torch::stack({position, scale, (1.0 - dot).mean()});
}

std::array<double, 3> DriftStabilityManager::tfLoss(
    const torch::Tensor& indices, const GaussianAttrState& current,
    const TransformField& student, const TransformField& teacher,
    float ratio, double diagonal, FrozenTeacherTargetCache* targets) const {
  torch::NoGradGuard no_grad;
  return asArray(tfLossTensor(indices, current, student, teacher, ratio,
                              diagonal, targets));
}

std::vector<int64_t> DriftStabilityManager::distillModule(
    bool selector_module, const std::vector<torch::Tensor>& populations,
    const GaussianAttrState& current,
    const improvements::selector::GumbelNetwork& selector, const TransformField& tf,
    const std::vector<float>& ratios, double temperature, double diagonal,
    uint64_t mapping_iter, HistoricalProbeCache* probe_cache) {
  const auto n = ratios.size();
  std::vector<int64_t> steps(n, 0);
  std::vector<torch::Tensor> indices(n);
  std::vector<std::unique_ptr<FrozenTeacherTargetCache>> targets(n);
  for (std::size_t r = 0; r < n; ++r) {
    if (!populations[r].defined() || populations[r].numel() == 0) continue;
    indices[r] = shuffle(populations[r]).slice(
        0, 0, std::min(populations[r].numel(), config_.max_gaussians));
    targets[r] = std::make_unique<FrozenTeacherTargetCache>(
        current.xyz.size(0), current.xyz.options());
    if (probe_cache) targets[r]->get(indices[r], [&](const torch::Tensor& ids) {
      if (selector_module)
        return probe_cache->probability(reference_.selector_teacher, ratios[r], ids);
      const auto value = probe_cache->transformed(reference_.tf_teacher, ratios[r], ids);
      return torch::cat({value.xyz, value.log_scaling, value.rotation}, 1);
    });
  }
  // One temporary optimizer per module/event, shared by all triggered ratios.
  torch::optim::Adam optimizer(
      selector_module ? selector->parameters() : tf->mlpParameters(),
      torch::optim::AdamOptions(selector_module ? config_.selector_learning_rate
                                               : config_.tf_mlp_learning_rate)
          .weight_decay(0.0));
  if (!selector_module) {
    optimizer.add_param_group(tf->gridParameters());
    optimizer.param_groups()[1].options().set_lr(config_.tf_grid_learning_rate);
  }
  const auto loss_for = [&](std::size_t r, const torch::Tensor& ids) {
    if (selector_module)
      return selectorLoss(ids, current, selector, reference_.selector_teacher,
                          ratios[r], temperature, targets[r].get());
    auto values = tfLossTensor(ids, current, tf, reference_.tf_teacher,
                               ratios[r], diagonal, targets[r].get());
    return config_.position_weight * values[0] + config_.scale_weight * values[1] +
           config_.rotation_weight * values[2];
  };
  std::vector<double> previous(n, 0.0);
  const auto log_epoch = [&](int64_t epoch, const std::vector<int64_t>& prior_steps) {
    torch::NoGradGuard no_grad;
    for (std::size_t r = 0; r < n; ++r) {
      if (!targets[r]) continue;
      double loss = 0.0, probability_l1 = 0.0;
      std::array<double, 3> components{};
      for (int64_t begin = 0; begin < indices[r].numel(); begin += config_.batch_size) {
        const auto ids = indices[r].slice(0, begin,
            std::min(begin + config_.batch_size, indices[r].numel()));
        loss += loss_for(r, ids).item<double>() * ids.numel();
        if (selector_module) {
          const auto target = targets[r]->get(ids, [&](const torch::Tensor& missing) {
            return selectorProbability(reference_.selector_teacher, current, missing,
                                       ratios[r], temperature);
          });
          probability_l1 += (selectorProbability(selector, current, ids, ratios[r],
                                                 temperature) - target)
                                .abs().mean().item<double>() * ids.numel();
        } else {
          const auto values = tfLoss(ids, current, tf, reference_.tf_teacher,
                                      ratios[r], diagonal, targets[r].get());
          for (int j = 0; j < 3; ++j) components[j] += values[j] * ids.numel();
        }
      }
      loss /= indices[r].numel();
      std::ostringstream line;
      line << std::fixed << std::setprecision(4)
           << "[HistoricalDistill " << (selector_module ? "SELECTOR" : "TF") << "]"
           << " iter=" << mapping_iter << " ratio=" << ratios[r]
           << " epoch=" << epoch << "/" << config_.epochs
           << " count=" << indices[r].numel() << " loss=" << loss
           << " loss_type=" << (selector_module ? "soft_BCE" : "TF")
           << " delta_prev=" << (epoch ? loss - previous[r] : 0.0)
           << " steps=" << steps[r] - prior_steps[r];
      if (selector_module) line << " probability_l1=" << probability_l1 / indices[r].numel();
      else line << " xyz/scale/rotation=" << components[0] / indices[r].numel()
                << "/" << components[1] / indices[r].numel()
                << "/" << components[2] / indices[r].numel();
      std::cout << line.str() << std::endl;
      previous[r] = loss;
    }
  };
  log_epoch(0, steps);
  for (int64_t epoch = 0; epoch < config_.epochs; ++epoch) {
    const auto prior_steps = steps;
    int64_t largest = 0;
    for (auto& ids : indices) if (ids.defined()) {
      ids = shuffle(ids);
      largest = std::max(largest, ids.numel());
    }
    // Interleave batches; do not finish all epochs of one ratio before the next.
    for (int64_t begin = 0; begin < largest; begin += config_.batch_size) {
      for (std::size_t offset = 0; offset < n; ++offset) {
        const auto r = (offset + static_cast<std::size_t>(epoch)) % n;
        if (!indices[r].defined() || begin >= indices[r].numel()) continue;
        const auto ids = indices[r].slice(0, begin,
            std::min(begin + config_.batch_size, indices[r].numel()));
        optimizer.zero_grad();
        auto loss = loss_for(r, ids);
        if (!torch::isfinite(loss).item<bool>()) {
          optimizer.zero_grad();
          return steps;  // Caller validates and restores the pre-event parameters.
        }
        if (!loss.requires_grad()) continue;  // TF outside-domain identity.
        loss.backward();
        if (!selector_module && tf->config().backend == "cuda")
          tf_cuda::tfCudaAdamDirect(optimizer, {0, 1});
        else optimizer.step();
        ++steps[r];
      }
    }
    log_epoch(epoch + 1, prior_steps);
  }
  optimizer.zero_grad();
  return steps;
}

DriftProbeReport DriftStabilityManager::maybeProbeAndDistill(
    const GaussianAttrState& current_input,
    const improvements::selector::GumbelNetwork& selector,
    const TransformField& tf, const std::vector<float>& ratios,
    double temperature, const AABB3D& aabb,
    const std::vector<HistoricalViewObservation>& views,
    uint64_t mapping_iter, uint64_t change_version,
    const QualityEvaluator& evaluate_quality, HistoricalProbeCache* probe_cache,
    bool deferred) {
  DriftProbeReport report;
  if (!prepareProbe(selector, tf, mapping_iter, change_version, deferred)) return report;
  // Hard expiry retires the target BEFORE any further teaching. A fresh clone
  // is pending: budget + module-specific quality + consecutive drift evidence
  // must qualify it again. Expiry is never counted as recovery or an EMA success.
  const bool selector_expired = reference_.selector_teacher &&
      mapping_iter - reference_.selector_full_reset_iter >=
          static_cast<uint64_t>(config_.reference_max_age);
  const bool tf_expired = reference_.tf_teacher &&
      mapping_iter - reference_.tf_full_reset_iter >=
          static_cast<uint64_t>(config_.reference_max_age);
  if (selector_expired || tf_expired) {
    refreshReference(selector, tf, mapping_iter, change_version, 0,
                     selector_expired, tf_expired, "expired_pending_qualification");
    report.reference_refreshed = true;
  }
  const auto has_population = [](const HistoricalViewObservation& view) {
    return !view.guard_only && view.visible_mature_indices.defined() &&
           view.visible_mature_indices.numel() > 0;
  };
  if (ratios.empty() || std::none_of(views.begin(), views.end(), has_population)) {
    std::cout << "[HistoricalProbe SKIP] iter=" << mapping_iter
              << " reason=no_historical_visible_mature" << std::endl;
    return report;
  }
  TORCH_CHECK(evaluate_quality, "DSD requires actual quality/budget evidence");
  const GaussianAttrState current{current_input.xyz.detach(),
      current_input.log_scaling.detach(), current_input.rotation.detach()};
  const double diagonal = aabb.defined() ? aabb.diagonal().item<double>() : 1.0;
  const auto old_selector = reference_.selector_teacher;
  const auto old_tf = reference_.tf_teacher;
  const auto selector_age = mapping_iter - reference_.selector_reference_iter;
  const auto tf_age = mapping_iter - reference_.tf_reference_iter;
  const auto n = ratios.size();
  report.ran = true;
  report.view_count = views.size();
  report.ratios.resize(n);
  if (selector_over_count_.size() != n) selector_over_count_.assign(n, 0);
  if (tf_over_count_.size() != n) tf_over_count_.assign(n, 0);
  const auto budget_error = [](const HistoricalRenderQuality& q) {
    if (!std::isfinite(q.mature_target) || !std::isfinite(q.mature_expected) ||
        q.mature_target < 0.0 || q.mature_expected < 0.0)
      return std::numeric_limits<double>::infinity();
    return std::abs(q.mature_expected - q.mature_target) / std::max(q.mature_target, 1.0);
  };
  const auto budget_ok = [&](const HistoricalRenderQuality& q) {
    return budget_error(q) <= config_.budget_relative_tolerance;
  };
  const auto probability = [&](const improvements::selector::GumbelNetwork& model,
                               const torch::Tensor& ids, float ratio) {
    return probe_cache ? probe_cache->probability(model, ratio, ids)
                       : selectorProbability(model, current, ids, ratio, temperature);
  };
  const auto drift = [&](bool is_selector, const torch::Tensor& ids, float ratio) {
    torch::NoGradGuard no_grad;
    if (is_selector)
      return (probability(selector, ids, ratio) - probability(old_selector, ids, ratio))
          .abs().mean().item<double>();
    if (!probe_cache) return weightedTF(tfLoss(ids, current, tf, old_tf, ratio, diagonal), config_);
    const auto a = probe_cache->transformed(tf, ratio, ids);
    const auto b = probe_cache->transformed(old_tf, ratio, ids);
    return weightedTF(asArray(torch::stack({
        (a.xyz - b.xyz).abs().mean() / std::max(diagonal, 1e-6),
        (a.log_scaling - b.log_scaling).abs().mean(),
        (1.0 - (a.rotation * b.rotation).sum(1).abs().clamp(0.0, 1.0)).mean()})), config_);
  };
  // All baselines refer to the CURRENT other module, never to the old pair.
  const auto measure = [&](std::size_t v, std::size_t r, bool teacher, bool is_selector) {
    return evaluate_quality(views[v].keyframe_id, ratios[r],
        teacher && is_selector ? old_selector : selector,
        teacher && !is_selector ? old_tf : tf);
  };
  std::vector<torch::Tensor> selector_populations(n), tf_populations(n);
  std::vector<std::vector<bool>> selector_bad(n, std::vector<bool>(views.size(), false));
  std::vector<std::vector<bool>> tf_bad = selector_bad;
  const auto find_harm = [&](bool is_selector, std::size_t r, std::vector<bool>& bad,
                              torch::Tensor& population, double& regression,
                              bool& qualified) {
    std::vector<torch::Tensor> sets;
    double weight_sum = 0.0;
    double current_psnr = 0.0, teacher_psnr = 0.0;
    HistoricalRenderQuality sample_current, sample_teacher;
    regression = 0.0;
    qualified = true;
    int64_t count = 0;
    std::fill(bad.begin(), bad.end(), false);
    for (std::size_t v = 0; v < views.size(); ++v) {
      if (!has_population(views[v])) continue;
      const auto before = measure(v, r, false, is_selector);
      const auto teacher = measure(v, r, true, is_selector);
      sample_current = before;
      sample_teacher = teacher;
      const bool valid = std::isfinite(before.psnr) && std::isfinite(teacher.psnr) &&
                         budget_ok(before) && budget_ok(teacher);
      qualified = qualified && valid;
      if (!valid) continue;
      const auto weight = std::max(views[v].outlier_score, 1e-6);
      const auto drop = teacher.psnr - before.psnr;
      weight_sum += weight;
      current_psnr += weight * before.psnr;
      teacher_psnr += weight * teacher.psnr;
      regression += weight * std::max(0.0, drop);
      if (drop > config_.psnr_regression_threshold) {
        bad[v] = true;
        ++count;
        sets.push_back(views[v].visible_mature_indices);
      }
    }
    regression = weight_sum > 0.0 ? regression / weight_sum : 0.0;
    population = torch::Tensor();
    if (qualified && count >= config_.min_bad_views &&
        regression > config_.psnr_regression_threshold && !sets.empty())
      population = historicalPopulationUnion(sets, current.xyz.size(0));
    std::ostringstream line;
    line << std::fixed << std::setprecision(4) << "[HistoricalProbe QUALITY] iter=" << mapping_iter
         << " module=" << (is_selector ? "SELECTOR" : "TF") << " ratio=" << ratios[r]
         << " qualified=" << qualified;
    if (weight_sum > 0.0)
      line << " psnr(current/reference)=" << current_psnr / weight_sum << "/"
           << teacher_psnr / weight_sum;
    else line << " psnr(current/reference)=UNQUALIFIED";
    line << " mature_expected(current/reference)=" << sample_current.mature_expected << "/"
         << sample_teacher.mature_expected << " mature_target=" << sample_current.mature_target
         << " mature_actual(current/reference)=" << sample_current.active_mature_count << "/"
         << sample_teacher.active_mature_count;
    std::cout << line.str() << std::endl;
    return count;
  };
  std::cout << "[HistoricalProbe BEGIN] iter=" << mapping_iter
            << " total_gaussians=" << current.xyz.size(0) << " views=" << views.size()
            << " ref_age(S/TF)=" << selector_age << "/" << tf_age
            << " quality_rule=shared_uniform_hard_selection" << std::endl;
  for (std::size_t r = 0; r < n; ++r) {
    auto& result = report.ratios[r];
    result.ratio = ratios[r];
    double weight_sum = 0.0;
    for (const auto& view : views) {
      if (!has_population(view)) continue;
      const auto weight = std::max(view.outlier_score, 1e-6);
      if (selector && old_selector)
        result.selector_drift += weight * drift(true, view.visible_mature_indices, ratios[r]);
      if (tf && old_tf)
        result.tf_drift += weight * drift(false, view.visible_mature_indices, ratios[r]);
      weight_sum += weight;
    }
    result.selector_drift /= weight_sum;
    result.tf_drift /= weight_sum;
    if (!std::isfinite(result.selector_drift) || !std::isfinite(result.tf_drift)) {
      clearProbeState();
      std::cout << "[HistoricalProbe SKIP] iter=" << mapping_iter
                << " reason=nonfinite_function_drift" << std::endl;
      return report;
    }
    selector_over_count_[r] = selector && old_selector &&
        result.selector_drift > config_.function_threshold ? selector_over_count_[r] + 1 : 0;
    tf_over_count_[r] = tf && old_tf &&
        result.tf_drift > config_.function_threshold ? tf_over_count_[r] + 1 : 0;
    double selector_regression = 0.0, tf_regression = 0.0;
    bool tf_qualified = false;
    if (selector_over_count_[r] >= config_.consecutive_trigger_count) {
      result.quality_evaluated = true;
      result.selector_bad_views = find_harm(true, r, selector_bad[r], selector_populations[r],
                                            selector_regression, result.selector_budget_qualified);
      result.selector_triggered = selector_populations[r].defined();
    }
    if (tf_over_count_[r] >= config_.consecutive_trigger_count) {
      result.quality_evaluated = true;
      result.tf_bad_views = find_harm(false, r, tf_bad[r], tf_populations[r],
                                      tf_regression, tf_qualified);
      result.tf_triggered = tf_populations[r].defined();
    }
    result.bad_views = std::max(result.selector_bad_views, result.tf_bad_views);
    result.weighted_psnr_regression = std::max(selector_regression, tf_regression);
    std::ostringstream line;
    line << std::fixed << std::setprecision(4) << "[HistoricalProbe] ratio=" << ratios[r]
         << " drift(S/TF)=" << result.selector_drift << "/" << result.tf_drift
         << " regression(S/TF)=" << selector_regression << "/" << tf_regression
         << " bad_views(S/TF)=" << result.selector_bad_views << "/" << result.tf_bad_views
         << " qualified(S/TF)=" << result.selector_budget_qualified << "/" << tf_qualified
         << " hits(S/TF)=" << selector_over_count_[r] << "/" << tf_over_count_[r]
         << " action=" << (result.selector_triggered ? (result.tf_triggered ? "SELECTOR+TF" : "SELECTOR")
                              : result.tf_triggered ? "TF" : "SKIP")
         << " reason=" << (!result.quality_evaluated ? "function_or_consecutive_gate"
              : result.selector_triggered || result.tf_triggered ? "module_quality_advantage"
              : !result.selector_budget_qualified && !tf_qualified ? "budget_or_invalid_quality"
              : "module_quality_gate");
    std::cout << line.str() << std::endl;
  }

  std::vector<std::vector<HistoricalRenderQuality>> event_before(views.size());
  const auto run_module = [&](bool is_selector, const std::vector<torch::Tensor>& populations,
                              const std::vector<std::vector<bool>>& bad) -> std::string {
    if (std::none_of(populations.begin(), populations.end(),
                    [](const auto& ids) { return ids.defined() && ids.numel() > 0; }))
      return "SKIP";
    // Save all view/ratio baselines before any shared-network update, including
    // recent guard views and ratios which did not request trigger QUALITY.
    std::vector<std::vector<HistoricalRenderQuality>> before(views.size());
    for (std::size_t v = 0; v < views.size(); ++v)
      for (std::size_t r = 0; r < n; ++r) {
        auto q = measure(v, r, false, is_selector);
        if (!std::isfinite(q.psnr) || !std::isfinite(budget_error(q))) {
          std::cout << "[HistoricalDistill RESULT] iter=" << mapping_iter
                    << " module=" << (is_selector ? "SELECTOR" : "TF")
                    << " outcome=REJECTED reason=invalid_guard_baseline steps=0" << std::endl;
          return "REJECTED";
        }
        before[v].push_back(q);
      }
    double pre_drift = 0.0;
    int64_t count = 0;
    for (std::size_t r = 0; r < n; ++r) if (populations[r].defined()) {
      pre_drift += drift(is_selector, populations[r], ratios[r]) * populations[r].numel();
      count += populations[r].numel();
      report.gaussian_count = std::max(report.gaussian_count, populations[r].numel());
    }
    auto parameters = is_selector ? selector->parameters() : tf->parameters();
    std::vector<torch::Tensor> saved;
    {
      torch::NoGradGuard guard;
      for (const auto& parameter : parameters) saved.push_back(parameter.detach().clone());
    }
    const auto steps = distillModule(is_selector, populations, current, selector, tf, ratios,
                                    temperature, diagonal, mapping_iter, probe_cache);
    for (std::size_t r = 0; r < n; ++r) {
      if (is_selector) report.ratios[r].selector_steps = steps[r];
      else report.ratios[r].tf_steps = steps[r];
    }
    const auto invalidate = [&]() {
      if (probe_cache) probe_cache->invalidateStudent(selector, tf, is_selector, !is_selector);
    };
    invalidate();
    const auto restore = [&]() {
      torch::NoGradGuard guard;
      for (std::size_t i = 0; i < parameters.size(); ++i) parameters[i].copy_(saved[i]);
      invalidate();
    };
    bool safe = true, recovered = true;
    double post_drift = 0.0, gain = 0.0, gain_weight = 0.0, worst_new_drop = 0.0;
    double worst_remaining = 0.0;
    {
      torch::NoGradGuard guard;
      std::vector<torch::Tensor> finite;
      for (const auto& parameter : parameters) finite.push_back(torch::isfinite(parameter).all());
      safe = torch::stack(finite).all().item<bool>();
    }
    for (std::size_t r = 0; r < n && safe; ++r) if (populations[r].defined()) {
      const auto value = drift(is_selector, populations[r], ratios[r]);
      safe = std::isfinite(value);
      recovered = recovered && value <= config_.function_threshold;
      post_drift += value * populations[r].numel();
    }
    if (safe) for (std::size_t v = 0; v < views.size(); ++v) {
      for (std::size_t r = 0; r < n; ++r) {
        const auto post = measure(v, r, false, is_selector);
        const auto drop = before[v][r].psnr - post.psnr;
        worst_new_drop = std::max(worst_new_drop, drop);
        safe = safe && std::isfinite(post.psnr) &&
            drop <= config_.psnr_regression_threshold &&
            std::isfinite(event_before[v][r].psnr) &&
            event_before[v][r].psnr - post.psnr <= config_.psnr_regression_threshold &&
            budget_error(post) <= std::max(double(config_.budget_relative_tolerance),
                                           budget_error(before[v][r])) + 1e-6;
        if (populations[r].defined() && bad[r][v]) {
          const auto target = measure(v, r, true, is_selector);
          const auto remaining = target.psnr - post.psnr;
          worst_remaining = std::max(worst_remaining, remaining);
          recovered = recovered && std::isfinite(target.psnr) &&
              remaining <= config_.psnr_regression_threshold;
          const auto weight = std::max(views[v].outlier_score, 1e-6);
          gain += weight * (post.psnr - before[v][r].psnr);
          gain_weight += weight;
        }
      }
    }
    gain = gain_weight > 0.0 ? gain / gain_weight : 0.0;
    const bool stepped = std::accumulate(steps.begin(), steps.end(), int64_t{0}) > 0;
    const bool improved = stepped && gain > 1e-4 && post_drift < pre_drift;
    const std::string outcome = !safe ? "REJECTED" :
        improved ? (recovered ? "RECOVERED" : "IMPROVED") : "NO_PROGRESS";
    if (outcome == "REJECTED" || outcome == "NO_PROGRESS") restore();
    std::ostringstream line;
    line << std::fixed << std::setprecision(4) << "[HistoricalDistill RESULT] iter=" << mapping_iter
         << " module=" << (is_selector ? "SELECTOR" : "TF") << " outcome=" << outcome
         << " psnr_gain=" << gain << " worst_new_drop=" << worst_new_drop
         << " worst_remaining_drop=" << worst_remaining
         << " drift=" << pre_drift / std::max(count, int64_t{1}) << "->"
         << post_drift / std::max(count, int64_t{1})
         << " rollback=" << (outcome == "REJECTED" || outcome == "NO_PROGRESS");
    std::cout << line.str() << std::endl;
    return outcome;
  };
  const bool any_trigger = std::any_of(report.ratios.begin(), report.ratios.end(),
      [](const auto& r) { return r.selector_triggered || r.tf_triggered; });
  if (any_trigger)
    std::cout << "[HistoricalDistill BEGIN] iter=" << mapping_iter
              << " epochs=" << config_.epochs << " batch_size=" << config_.batch_size << std::endl;
  if (any_trigger)
    for (std::size_t v = 0; v < views.size(); ++v)
      for (std::size_t r = 0; r < n; ++r)
        event_before[v].push_back(measure(v, r, false, true));
  report.selector_outcome = run_module(true, selector_populations, selector_bad);
  // A committed Selector update changes the TF counterfactual. Recheck its
  // quality/population before teaching TF; never reuse the pre-Selector gate.
  if (report.selector_outcome == "IMPROVED" || report.selector_outcome == "RECOVERED") {
    for (std::size_t r = 0; r < n; ++r) if (report.ratios[r].tf_triggered) {
      double regression;
      bool qualified;
      report.ratios[r].tf_bad_views = find_harm(false, r, tf_bad[r], tf_populations[r],
                                                regression, qualified);
      report.ratios[r].tf_triggered = tf_populations[r].defined();
      if (!report.ratios[r].tf_triggered) {
        std::ostringstream line;
        line << std::fixed << std::setprecision(4)
             << "[HistoricalDistill RECHECK] iter=" << mapping_iter << " ratio=" << ratios[r]
             << " module=TF action=SKIP reason=quality_after_selector_update";
        std::cout << line.str() << std::endl;
      }
    }
  }
  report.tf_outcome = run_module(false, tf_populations, tf_bad);
  report.selector_recovery_passed = report.selector_outcome == "RECOVERED";
  report.tf_recovery_passed = report.tf_outcome == "RECOVERED";
  for (std::size_t r = 0; r < n; ++r) {
    double weight_sum = 0.0;
    for (const auto& view : views) if (has_population(view)) {
      const auto weight = std::max(view.outlier_score, 1e-6);
      if (selector && old_selector)
        report.ratios[r].selector_post_drift += weight * drift(true, view.visible_mature_indices, ratios[r]);
      if (tf && old_tf)
        report.ratios[r].tf_post_drift += weight * drift(false, view.visible_mature_indices, ratios[r]);
      weight_sum += weight;
    }
    report.ratios[r].selector_post_drift /= weight_sum;
    report.ratios[r].tf_post_drift /= weight_sum;
  }

  const auto update_reference = [&](bool is_selector, uint64_t age, const std::string& outcome) {
    if ((is_selector && (!selector || !old_selector)) || (!is_selector && (!tf || !old_tf))) return;
    auto& confirmations = is_selector ? selector_stable_count_ : tf_stable_count_;
    const bool recovered = outcome == "RECOVERED";
    // Partially recovered updates must not immediately replace their own target.
    if (outcome == "IMPROVED" || outcome == "REJECTED" || outcome == "NO_PROGRESS" ||
        (!recovered && age < static_cast<uint64_t>(config_.reference_min_age))) {
      confirmations = 0;
      return;
    }
    bool qualified = true, old_budget_valid = true;
    double quality_delta = 0.0;
    for (std::size_t v = 0; v < views.size(); ++v)
      for (std::size_t r = 0; r < n; ++r) {
        const auto now = measure(v, r, false, is_selector);
        const auto old = measure(v, r, true, is_selector);
        qualified = qualified && std::isfinite(now.psnr) && budget_ok(now);
        const bool old_ok = budget_ok(old);
        old_budget_valid = old_budget_valid && old_ok;
        // An over-budget Selector cannot veto a budget-compliant replacement
        // using its unfair PSNR. Never average its weights into that replacement.
        if (!old_ok && is_selector) continue;
        qualified = qualified && std::isfinite(old.psnr) &&
                    now.psnr + config_.psnr_regression_threshold >= old.psnr;
        const auto weight = std::max(views[v].outlier_score, 1e-6);
        quality_delta += weight * (now.psnr - old.psnr);
      }
    qualified = qualified && (recovered || quality_delta >= -1e-6);
    confirmations = qualified ? confirmations + 1 : 0;
    if (!qualified || (!recovered && confirmations < config_.non_regression_refresh_count)) {
      std::cout << "[HistoricalDistill KEEP] iter=" << mapping_iter
                << " module=" << (is_selector ? "SELECTOR" : "TF")
                << " confirmations=" << confirmations
                << " reason=" << (age >= static_cast<uint64_t>(config_.reference_horizon)
                      ? "horizon_waiting_quality_or_budget" : "quality_or_budget_confirmation") << std::endl;
      return;
    }
    auto candidate_s = selector;
    auto candidate_tf = tf;
    if (is_selector) {
      candidate_s = improvements::selector::GumbelNetwork(selector->cloneFrozen());
      if (old_budget_valid) blendReference(*old_selector, *candidate_s, config_.reference_ema_decay);
    } else {
      candidate_tf = TransformField(tf->cloneFrozen());
      blendReference(*old_tf, *candidate_tf, config_.reference_ema_decay);
    }
    // Validate the actual EMA network, with the same CURRENT other module,
    // against both current quality and the eligible old reference.
    bool candidate_ok = true;
    for (std::size_t v = 0; v < views.size(); ++v)
      for (std::size_t r = 0; r < n; ++r) {
        const auto q = evaluate_quality(views[v].keyframe_id, ratios[r], candidate_s, candidate_tf);
        const auto now = measure(v, r, false, is_selector);
        const auto old = measure(v, r, true, is_selector);
        candidate_ok = candidate_ok && std::isfinite(q.psnr) && budget_ok(q) &&
            q.psnr + config_.psnr_regression_threshold >= now.psnr &&
            (!(budget_ok(old) && std::isfinite(old.psnr)) ||
             q.psnr + config_.psnr_regression_threshold >= old.psnr);
      }
    if (candidate_ok) {
      if (is_selector) {
        reference_.selector_teacher = candidate_s;
        reference_.selector_reference_iter = mapping_iter;
        if (!old_budget_valid) reference_.selector_full_reset_iter = mapping_iter;
        selector_over_count_.assign(n, 0);
      } else {
        reference_.tf_teacher = candidate_tf;
        reference_.tf_reference_iter = mapping_iter;
        tf_over_count_.assign(n, 0);
      }
      confirmations = 0;
      report.reference_refreshed = true;
    }
    std::cout << "[HistoricalDistill "
              << (candidate_ok ? (old_budget_valid ? "EMA_UPDATE" : "REBASE_PENDING") : "KEEP")
              << "] iter=" << mapping_iter << " module=" << (is_selector ? "SELECTOR" : "TF")
              << " reason=" << (candidate_ok ? "budget_and_quality_validated"
                                            : "ema_candidate_quality_or_budget_failed") << std::endl;
  };
  update_reference(true, selector_age, report.selector_outcome);
  update_reference(false, tf_age, report.tf_outcome);
  if (any_trigger)
    std::cout << "[HistoricalDistill END] iter=" << mapping_iter
              << " gaussian_count=" << report.gaussian_count
              << " outcome(S/TF)=" << report.selector_outcome << "/" << report.tf_outcome << std::endl;
  return report;
}

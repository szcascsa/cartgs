#include "include/change_management.h"
#include "include/frozen_teacher_target_cache.h"
#include "include/transform_field_cuda_optimizer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace {
std::unique_ptr<torch::optim::Adam> independentAdam(
    torch::optim::Adam& source, const std::vector<double>& learning_rates) {
  TORCH_CHECK(!learning_rates.empty() &&
                  learning_rates.size() == source.param_groups().size(),
              "CD learning-rate count must match optimizer parameter groups");
  std::vector<torch::optim::OptimizerParamGroup> groups;
  for (std::size_t i = 0; i < source.param_groups().size(); ++i) {
    const auto& group = source.param_groups()[i];
    auto options = static_cast<const torch::optim::AdamOptions&>(group.options());
    options.weight_decay(0.0); // CD has only its explicit distillation objective.
    options.lr(learning_rates[i]);
    groups.emplace_back(group.params(), std::make_unique<torch::optim::AdamOptions>(options));
  }
  return std::make_unique<torch::optim::Adam>(groups);
}

torch::Tensor scopeIndices(DistillScope scope, const ChangeEvent& event,
                           const ChangeCapture& capture) {
  if (scope == DistillScope::ALL_SNAPSHOT_MATURE) return capture.mature_indices_before;
  if (scope == DistillScope::CORRECTED_MATURE)
    return torch::nonzero(capture.mature_mask_before & (event.correction_group_id >= 0)).reshape({-1});
  return capture.mature_indices_before.slice(0, 0, 0);
}

GaussianAttrState transportTarget(GaussianAttrState target, const torch::Tensor& indices,
                                  const ChangeEvent& event) {
  torch::NoGradGuard no_grad;
  auto ids = event.correction_group_id.index_select(0, indices);
  for (std::size_t g = 0; g < event.correction_groups.size(); ++g) {
    auto local = torch::nonzero(ids == static_cast<int64_t>(g)).reshape({-1});
    if (local.numel() == 0) continue;
    auto moved = applyGaussianTransform(target.select(local), event.correction_groups[g]);
    // index_copy is out-of-place: never modify the frozen snapshot via aliases.
    target = {target.xyz.index_copy(0, local, moved.xyz),
              target.log_scaling.index_copy(0, local, moved.log_scaling),
              target.rotation.index_copy(0, local, moved.rotation)};
  }
  return target;
}

std::array<torch::Tensor, 3> tfErrors(const GaussianAttrState& student,
    const GaussianAttrState& target, double diagonal) {
  auto qs = quaternion_utils::normalize(student.rotation);
  auto qt = quaternion_utils::normalize(target.rotation);
  return {(student.xyz - target.xyz).abs().mean() / (diagonal + 1e-8),
          (student.log_scaling - target.log_scaling).abs().mean(),
          (1.0 - (qs * qt).sum(1).abs().clamp(0.0, 1.0)).mean()};
}

torch::Tensor selectorLoss(improvements::selector::GumbelNetwork& student,
    const ChangeCapture& capture, const GaussianAttrState& current,
    const torch::Tensor& indices, float ratio, double temperature,
    FrozenTeacherTargetCache& targets) {
  auto after = current.select(indices);
  auto ratios = torch::full({indices.numel(), 1}, ratio, after.xyz.options());
  auto teacher = targets.get(indices, [&](const torch::Tensor& missing) {
    auto before = capture.old_state.select(missing);
    auto teacher_ratios = torch::full({missing.numel(), 1}, ratio, before.xyz.options());
    return capture.teacher_selector.ptr()->forwardProbability(
        before.xyz, before.rotation, before.log_scaling.exp(), teacher_ratios, temperature);
  });
  auto probability = student->forwardProbability(
      after.xyz, after.rotation, after.log_scaling.exp(), ratios, temperature);
  return (probability - teacher).abs().mean();
}

// Stratified batches preserve separate population means. Tiny strata are reused
// in full; otherwise every member is visited once, with population-size weights.
std::pair<torch::Tensor, double> stratumBatch(
    const torch::Tensor& indices, int64_t batch, int64_t batches) {
  const auto n = indices.numel();
  if (n < batches) return {indices, 1.0};
  const auto start = batch * n / batches, end = (batch + 1) * n / batches;
  return {indices.slice(0, start, end), static_cast<double>(batches * (end - start)) / n};
}

double elapsedMs(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
}
} // namespace

ChangeDistillManager::ChangeDistillManager(ChangeDistillConfig config)
    : config_(std::move(config)), rng_(config_.seed) {
  TORCH_CHECK(config_.batch_size > 0 && config_.epochs > 0, "Invalid CD batch size or epochs");
  TORCH_CHECK(config_.selectorEpochs() > 0 && config_.tfEpochs() > 0,
              "CD Selector and TF epochs must be positive");
  TORCH_CHECK(std::isfinite(config_.anchor_ratio_to_corrected) &&
                  config_.anchor_ratio_to_corrected >= 0 && config_.anchor_max_count >= 0,
              "Invalid CD anchor sampling parameters");
  for (float weight : {config_.position_weight, config_.scale_weight,
                       config_.rotation_weight, config_.selector_weight, config_.anchor_loss_weight})
    TORCH_CHECK(std::isfinite(weight) && weight >= 0, "CD weights must be finite and nonnegative");
  TORCH_CHECK(!config_.post_change_replay, "Post-change replay is not part of Change Distill");
  for (float learning_rate : {config_.selector_learning_rate,
                              config_.tf_grid_learning_rate,
                              config_.tf_mlp_learning_rate})
    TORCH_CHECK(std::isfinite(learning_rate) && learning_rate > 0.0f,
                "CD learning rates must be finite and positive");
}

torch::Tensor ChangeDistillManager::shuffle(const torch::Tensor& indices) {
  std::vector<int64_t> order(indices.numel());
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), rng_);
  auto permutation = torch::tensor(order, torch::TensorOptions().dtype(torch::kLong))
                         .to(indices.device());
  return indices.index_select(0, permutation);
}

torch::Tensor ChangeDistillManager::sampleAnchors(
    const torch::Tensor& unaffected, int64_t corrected_count) {
  if (!config_.enable_unaffected_anchors || corrected_count == 0)
    return unaffected.slice(0, 0, 0);
  const double wanted = std::floor(static_cast<double>(corrected_count) * config_.anchor_ratio_to_corrected);
  const auto count = static_cast<int64_t>(std::min(wanted, static_cast<double>(
      std::min(unaffected.numel(), config_.anchor_max_count))));
  if (count == 0) return unaffected.slice(0, 0, 0);
  return shuffle(unaffected).slice(0, 0, count);
}

ChangeDistillReport ChangeDistillManager::execute(
    const ChangeEvent& event, const ChangeCapture& capture, ChangeTrainingContext& context) {
  ChangeDistillReport report;
  if (!config_.enabled) return report;
  TORCH_CHECK(!context.ratios.empty() && std::isfinite(context.temperature) &&
                  context.temperature > 0, "CD requires ratios and a positive temperature");
  for (auto ratio : context.ratios)
    TORCH_CHECK(std::isfinite(ratio) && ratio >= 0 && ratio <= 1, "Invalid CD ratio");
  torch::AutoGradMode enable_grad(true);
  auto current = context.current.detach();
  const double ratio_count = static_cast<double>(context.ratios.size());
  const auto batch_size = config_.batch_size;
  auto selector_indices = scopeIndices(event.selector_scope, event, capture);
  auto tf_indices = scopeIndices(event.tf_scope, event, capture);
  const auto event_type = changeEventTypeName(event.flags);
  std::cout << "[CD BEGIN] event=" << event.id << " type=" << event_type
            << " mapping_iter=" << event.mapping_iter
            << " epochs=" << config_.epochs
            << " selector_epochs=" << config_.selectorEpochs()
            << " tf_epochs=" << config_.tfEpochs() << " batch_size=" << batch_size
            << " ratios=" << context.ratios.size()
            << " selector_scope=" << distillScopeName(event.selector_scope)
            << " tf_scope=" << distillScopeName(event.tf_scope) << std::endl;

  if (selector_indices.numel() > 0 && config_.selector_weight > 0) {
    TORCH_CHECK(context.selector && capture.teacher_selector && context.normal_selector_optimizer,
                "Selector CD requested without live model/teacher/optimizer");
    const auto start = std::chrono::steady_clock::now();
    report.selector_count = selector_indices.numel();
    std::vector<FrozenTeacherTargetCache> teacher_targets;
    for (std::size_t r = 0; r < context.ratios.size(); ++r)
      teacher_targets.emplace_back(capture.snapshot_count, current.xyz.options());
    const bool local = (event.flags & toBits(ChangeFlag::LOCAL_GEOMETRY)) != 0;
    auto unaffected = torch::nonzero(capture.mature_mask_before &
                                     (event.correction_group_id == -1)).reshape({-1});
    auto anchors = local ? sampleAnchors(unaffected, selector_indices.numel())
                         : unaffected.slice(0, 0, 0);
    // Keep exactly the same anchor audit set for pre/post metrics, even when
    // later epochs resample their training anchors.
    const auto audit_anchors = anchors;
    auto evaluate = [&](const torch::Tensor& indices) {
      torch::NoGradGuard no_grad;
      if (indices.numel() == 0) return 0.0;
      double total = 0.0;
      for (int64_t begin = 0; begin < indices.numel(); begin += batch_size) {
        auto batch = indices.slice(0, begin, std::min(begin + batch_size, indices.numel()));
        for (std::size_t r = 0; r < context.ratios.size(); ++r)
          total += selectorLoss(context.selector, capture, current, batch, context.ratios[r],
                                context.temperature, teacher_targets[r]).item<double>() * batch.numel();
      }
      return total / (indices.numel() * ratio_count);
    };
    report.selector_pre = evaluate(selector_indices);
    report.anchor_pre = evaluate(audit_anchors);
    const auto weighted_loss = [&](double primary, double anchor) {
      return config_.selector_weight * (primary + config_.anchor_loss_weight * anchor);
    };
    const auto log_epoch = [&](int64_t epoch, double primary, double anchor,
                               double delta, int64_t steps) {
      std::ostringstream line;
      line << std::fixed << std::setprecision(4)
           << "[SELECTOR CD EPOCH] event=" << event.id << " type=" << event_type
           << " epoch=" << epoch << "/" << config_.selectorEpochs()
           << " loss=" << weighted_loss(primary, anchor)
           << " primary_loss=" << primary << " anchor_loss=" << anchor
           << " delta_prev=" << delta
           << " trend=" << (epoch == 0 ? "BASELINE" :
                            delta < 0 ? "DOWN" : delta > 0 ? "UP" : "FLAT")
           << " steps=" << steps
           << " count=" << selector_indices.numel()
           << " audit_anchors=" << audit_anchors.numel();
      std::cout << line.str() << std::endl;
    };
    double previous_loss = weighted_loss(report.selector_pre, report.anchor_pre);
    log_epoch(0, report.selector_pre, report.anchor_pre, 0.0, 0);
    auto optimizer = independentAdam(
        *context.normal_selector_optimizer,
        {static_cast<double>(config_.selector_learning_rate)});
    for (int64_t epoch = 0; epoch < config_.selectorEpochs(); ++epoch) {
      const auto steps_before = report.selector_steps;
      if (epoch > 0 && local) anchors = sampleAnchors(unaffected, selector_indices.numel());
      report.anchor_count = anchors.numel();
      auto corrected = shuffle(selector_indices);
      const auto batches = (std::max(corrected.numel(), anchors.numel()) + batch_size - 1) / batch_size;
      for (int64_t b = 0; b < batches; ++b) {
        optimizer->zero_grad();
        auto main = stratumBatch(corrected, b, batches);
        auto anchor = stratumBatch(anchors, b, batches);
        auto total = torch::zeros({}, current.xyz.options());
        for (std::size_t r = 0; r < context.ratios.size(); ++r) {
          total = total + main.second * selectorLoss(context.selector, capture, current,
                          main.first, context.ratios[r], context.temperature, teacher_targets[r]);
          if (anchor.first.numel() > 0 && config_.anchor_loss_weight > 0)
            total = total + config_.anchor_loss_weight * anchor.second *
                selectorLoss(context.selector, capture, current, anchor.first,
                             context.ratios[r], context.temperature, teacher_targets[r]);
        }
        total = total * (config_.selector_weight / ratio_count);
        TORCH_CHECK(torch::isfinite(total).item<bool>(), "Nonfinite Selector CD loss");
        if (total.item<double>() > 0.0) {
          total.backward();
          if (context.selector->backend() == "cuda")
            tf_cuda::tfCudaAdamDirect(*optimizer, {0});
          else optimizer->step();
          ++report.selector_steps;
        }
      }
      report.selector_post = evaluate(selector_indices);
      report.anchor_post = evaluate(audit_anchors);
      const double loss = weighted_loss(report.selector_post, report.anchor_post);
      log_epoch(epoch + 1, report.selector_post, report.anchor_post,
                loss - previous_loss, report.selector_steps - steps_before);
      previous_loss = loss;
    }
    optimizer->zero_grad();
    std::ostringstream summary;
    summary << std::fixed << std::setprecision(4)
              << "[SELECTOR CD] event=" << event.id << " count=" << report.selector_count
              << " anchors=" << report.anchor_count << " pre=" << report.selector_pre
              << " post=" << report.selector_post << " anchor_pre=" << report.anchor_pre
              << " anchor_post=" << report.anchor_post << " steps=" << report.selector_steps
              << " epochs=" << config_.selectorEpochs()
              << " lr_micro=" << config_.selector_learning_rate * 1000000.0
              << " time_ms=" << elapsedMs(start);
    std::cout << summary.str() << std::endl;
  } else if (event.selector_scope != DistillScope::NONE) {
    std::cout << "[SELECTOR CD SKIP] event=" << event.id << " type=" << event_type
              << " reason=" << (selector_indices.numel() == 0 ? "empty_population" : "zero_weight")
              << std::endl;
  }

  if (tf_indices.numel() > 0 &&
      config_.position_weight + config_.scale_weight + config_.rotation_weight > 0) {
    TORCH_CHECK(context.tf && capture.teacher_tf && context.normal_tf_optimizer,
                "TF CD requested without live model/teacher/optimizer");
    const auto start = std::chrono::steady_clock::now();
    if (context.tf->config().outside_identity) {
      auto inside = context.tf->insideMask(current.xyz.index_select(0, tf_indices));
      const auto before = tf_indices.numel();
      tf_indices = tf_indices.index({inside});
      report.skipped_outside = before - tf_indices.numel();
    }
    report.tf_count = tf_indices.numel();
    const double diagonal = capture.old_aabb.diagonal().item<double>();
    const std::array<double, 3> weights = {config_.position_weight, config_.scale_weight, config_.rotation_weight};
    std::vector<FrozenTeacherTargetCache> teacher_targets;
    for (std::size_t r = 0; r < context.ratios.size(); ++r)
      teacher_targets.emplace_back(capture.snapshot_count, current.xyz.options());
    auto errors = [&](const torch::Tensor& batch, std::size_t r) {
      const auto ratio = context.ratios[r];
      auto packed = teacher_targets[r].get(batch, [&](const torch::Tensor& missing) {
        auto target = transportTarget(capture.teacher_tf.ptr()->applyLogResidual(
            capture.old_state.select(missing), ratio), missing, event);
        return torch::cat({target.xyz, target.log_scaling, target.rotation}, 1);
      });
      GaussianAttrState target{packed.slice(1, 0, 3), packed.slice(1, 3, 6),
                               packed.slice(1, 6, 10)};
      return tfErrors(context.tf->applyLogResidual(current.select(batch), ratio), target, diagonal);
    };
    auto evaluate = [&](const torch::Tensor& indices) {
      torch::NoGradGuard no_grad;
      std::array<double, 3> sums{};
      if (indices.numel() == 0) return sums;
      for (int64_t begin = 0; begin < indices.numel(); begin += batch_size) {
        auto batch = indices.slice(0, begin, std::min(begin + batch_size, indices.numel()));
        for (std::size_t r = 0; r < context.ratios.size(); ++r) {
          auto loss = errors(batch, r);
          for (int j = 0; j < 3; ++j) sums[j] += loss[j].item<double>() * batch.numel();
        }
      }
      for (auto& sum : sums) sum /= indices.numel() * ratio_count;
      return sums;
    };
    const auto pre = evaluate(tf_indices);
    const auto weighted_loss = [&](const std::array<double, 3>& loss) {
      return weights[0] * loss[0] + weights[1] * loss[1] + weights[2] * loss[2];
    };
    const auto log_epoch = [&](int64_t epoch, const std::array<double, 3>& loss,
                               double delta, int64_t steps) {
      std::ostringstream line;
      line << std::fixed << std::setprecision(4)
           << "[TF CD EPOCH] event=" << event.id << " type=" << event_type
           << " epoch=" << epoch << "/" << config_.tfEpochs()
           << " loss=" << weighted_loss(loss)
           << " position_loss=" << loss[0] << " log_scale_loss=" << loss[1]
           << " rotation_loss=" << loss[2] << " delta_prev=" << delta
           << " trend=" << (epoch == 0 ? "BASELINE" :
                            delta < 0 ? "DOWN" : delta > 0 ? "UP" : "FLAT")
           << " steps=" << steps << " count=" << tf_indices.numel()
           << " skipped_outside=" << report.skipped_outside;
      std::cout << line.str() << std::endl;
    };
    auto post = pre;
    double previous_loss = weighted_loss(pre);
    // Diagnostic only: no extra TF anchors are trained in this version.
    auto audit = torch::nonzero(capture.mature_mask_before &
                                (event.correction_group_id == -1)).reshape({-1});
    if (context.tf->config().outside_identity)
      audit = audit.index({context.tf->insideMask(current.xyz.index_select(0, audit))});
    audit = shuffle(audit).slice(0, 0, std::min(batch_size, audit.numel()));
    const auto audit_pre = evaluate(audit);
    if (tf_indices.numel() > 0) {
      log_epoch(0, pre, 0.0, 0);
      auto optimizer = independentAdam(
          *context.normal_tf_optimizer,
          {static_cast<double>(config_.tf_mlp_learning_rate),
           static_cast<double>(config_.tf_grid_learning_rate)});
      for (int64_t epoch = 0; epoch < config_.tfEpochs(); ++epoch) {
        const auto steps_before = report.tf_steps;
        auto indices = shuffle(tf_indices);
        for (int64_t begin = 0; begin < indices.numel(); begin += batch_size) {
          optimizer->zero_grad();
          auto batch = indices.slice(0, begin, std::min(begin + batch_size, indices.numel()));
          auto total = torch::zeros({}, current.xyz.options());
          for (std::size_t r = 0; r < context.ratios.size(); ++r) {
            auto loss = errors(batch, r);
            for (int j = 0; j < 3; ++j) total = total + weights[j] * loss[j];
          }
          total = total / ratio_count;
          TORCH_CHECK(torch::isfinite(total).item<bool>(), "Nonfinite TF CD loss");
          if (total.item<double>() > 0.0) {
            total.backward();
            if (context.tf->config().backend == "cuda")
              tf_cuda::tfCudaAdamDirect(*optimizer, {0, 1});
            else optimizer->step();
            ++report.tf_steps;
          }
        }
        post = evaluate(tf_indices);
        const double loss = weighted_loss(post);
        log_epoch(epoch + 1, post, loss - previous_loss, report.tf_steps - steps_before);
        previous_loss = loss;
      }
      optimizer->zero_grad();
    } else {
      std::cout << "[TF CD SKIP] event=" << event.id << " type=" << event_type
                << " reason=no_in_domain_gaussians skipped_outside=" << report.skipped_outside
                << std::endl;
    }
    const auto audit_post = evaluate(audit);
    report.tf_pre = weighted_loss(pre);
    report.tf_post = weighted_loss(post);
    std::ostringstream summary;
    summary << std::fixed << std::setprecision(4)
              << "[TF CD] event=" << event.id << " count=" << report.tf_count
              << " skipped_outside=" << report.skipped_outside
              << " pre_xyz/log_scale/rotation=" << pre[0] << "," << pre[1] << "," << pre[2]
              << " post_xyz/log_scale/rotation=" << post[0] << "," << post[1] << "," << post[2]
              << " untouched_pre=" << audit_pre[0] << "," << audit_pre[1] << "," << audit_pre[2]
              << " untouched_post=" << audit_post[0] << "," << audit_post[1] << "," << audit_post[2]
              << " steps=" << report.tf_steps
              << " mlp_lr_micro=" << config_.tf_mlp_learning_rate * 1000000.0
              << " grid_lr_micro=" << config_.tf_grid_learning_rate * 1000000.0
              << " time_ms=" << elapsedMs(start);
    std::cout << summary.str() << std::endl;
  } else if (event.tf_scope != DistillScope::NONE) {
    std::cout << "[TF CD SKIP] event=" << event.id << " type=" << event_type
              << " reason=" << (tf_indices.numel() == 0 ? "empty_population" : "zero_weight")
              << std::endl;
  }
  std::cout << "[CD END] event=" << event.id << " type=" << event_type
            << " selector_steps=" << report.selector_steps
            << " tf_steps=" << report.tf_steps << " skipped_outside=" << report.skipped_outside
            << std::endl;
  return report;
}

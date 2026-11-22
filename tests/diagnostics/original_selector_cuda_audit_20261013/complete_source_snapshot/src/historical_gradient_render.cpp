#include "include/historical_gradient_render.h"
#include "include/model_snapshot.h"

GaussianRenderInput GuardGaussianParameters::input(
    const AdamStepPreview& preview, const std::vector<torch::Tensor>* candidate) const {
  const auto value = [&](const torch::Tensor& p) {
    return candidate ? preview.value(p, *candidate) : p;
  };
  return {value(xyz), value(opacity).sigmoid(), value(log_scaling).exp(),
          torch::nn::functional::normalize(value(rotation)),
          value(features_dc), value(features_rest), active_sh_degree, max_sh_degree};
}

GaussianRenderInput historicalLoDInput(
    const GaussianRenderInput& full, const torch::Tensor& mature_mask,
    const torch::Tensor& mature_indices, const ElasticBudgetPlan& budget,
    const torch::Tensor& uniform,
    const std::shared_ptr<improvements::selector::GumbelNetworkImpl>& selector,
    const std::shared_ptr<TransformFieldImpl>& field,
    float ratio, double temperature) {
  const bool all = mature_indices.numel() == full.xyz.size(0);
  auto hard = torch::zeros({full.xyz.size(0)}, full.xyz.options().requires_grad(false));
  if (selector && (all || mature_indices.numel() > 0)) {
    auto sample = selector->forwardHardWithUniform(
        all ? full.xyz : full.xyz.index_select(0, mature_indices),
        all ? full.rotation : full.rotation.index_select(0, mature_indices),
        all ? full.scaling : full.scaling.index_select(0, mature_indices), ratio,
        all ? uniform : uniform.index_select(0, mature_indices), temperature);
    hard = all ? sample : hard.index_copy(0, mature_indices, sample);
  }
  const auto protected_f = budget.admitted.to(full.xyz.scalar_type());
  const auto mask = protected_f + (1.0f - protected_f) * hard;
  const auto map = torch::nonzero(mask.detach() > 0.0f).reshape({-1});
  auto compact = full.indexSelect(map);
  compact.opacity = compact.opacity * mask.index_select(0, map).unsqueeze(1);
  if (field && map.numel() > 0) {
    auto eligible = mask.detach().index_select(0, map) > 0.0f;
    if (field->config().mature_only)
      eligible = eligible & mature_mask.index_select(0, map);
    if (field->config().outside_identity) eligible = eligible & field->insideMask(compact.xyz);
    const auto local = torch::nonzero(eligible).reshape({-1});
    if (local.numel() > 0) {
      const auto transformed = field->applyResidual(compact.xyz.index_select(0, local),
          compact.scaling.index_select(0, local), compact.rotation.index_select(0, local), ratio);
      compact.xyz = compact.xyz.index_copy(0, local, transformed.xyz);
      compact.scaling = compact.scaling.index_copy(0, local, transformed.scaling);
      compact.rotation = compact.rotation.index_copy(0, local, transformed.rotation);
    }
  }
  return compact;
}

namespace {
void copyCandidate(const torch::nn::Module& source, torch::nn::Module& target,
                   const AdamStepPreview& preview, const std::vector<torch::Tensor>& candidate) {
  const auto target_parameters = target.named_parameters();
  for (const auto& item : source.named_parameters())
    target_parameters[item.key()].copy_(preview.value(item.value(), candidate));
  const auto target_buffers = target.named_buffers();
  for (const auto& item : source.named_buffers()) target_buffers[item.key()].copy_(item.value());
}
}
void GuardModuleCandidates::prepare(
    const AdamStepPreview& preview, const std::vector<torch::Tensor>& candidate,
    const std::shared_ptr<improvements::selector::GumbelNetworkImpl>& selector,
    const std::shared_ptr<TransformFieldImpl>& field) {
  torch::NoGradGuard no_grad;
  if (selector) {
    if (!selector_ || selector_source_.lock() != selector) {
      selector_ = selector->cloneFrozen(); selector_source_ = selector;
    }
    copyCandidate(*selector, *selector_, preview, candidate);
  } else { selector_.reset(); selector_source_.reset(); }
  if (field) {
    if (!field_ || field_source_.lock() != field) {
      field_ = field->cloneFrozen(); field_source_ = field;
    }
    // Includes AABB buffers: a reused scratch field cannot retain an old domain.
    copyCandidate(*field, *field_, preview, candidate);
  } else { field_.reset(); field_source_.reset(); }
}

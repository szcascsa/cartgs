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
    const GaussianRenderInput& full,
    const torch::Tensor& maturity,
    const torch::Tensor& uniform,
    const std::shared_ptr<improvements::selector::GumbelNetworkImpl>& selector,
    const std::shared_ptr<TransformFieldImpl>& field,
    float ratio, double temperature, const torch::Tensor& selection_maturity, float entry) {
  auto hard = torch::zeros({full.xyz.size(0)}, full.xyz.options().requires_grad(false));
  if (selector && full.xyz.size(0) > 0) {
    const auto indices = entry > 0 ? torch::nonzero(maturity >= entry).reshape({-1}) : torch::Tensor{};
    auto probability = elasticSelectorProbability(maturity,
        selection_maturity.defined() ? selection_maturity : maturity, entry, ratio, indices,
        [&](const torch::Tensor& rows) {
          return selector->forwardProbability(rows.defined() ? full.xyz.index_select(0, rows) : full.xyz,
              rows.defined() ? full.rotation.index_select(0, rows) : full.rotation,
              rows.defined() ? full.scaling.index_select(0, rows) : full.scaling, ratio, temperature, rows,
              rows.defined() ? full.features_dc.index_select(0, rows) : full.features_dc,
              rows.defined() ? full.opacity.index_select(0, rows) : full.opacity);
        });
    probability = selector->maskInactiveSlots(probability,full.opacity);
    hard = sampleSelectorST(probability, uniform);
  }
  const auto mask = hard;
  const auto map = torch::nonzero(mask.detach() > 0.0f).reshape({-1});
  auto compact = full.indexSelect(map);
  compact.opacity = compact.opacity * mask.index_select(0, map).unsqueeze(1);
  if (field && map.numel() > 0) {
    const auto compact_maturity = maturity.index_select(0, map).detach();
    const auto eligible = field->candidateMask(compact.xyz, compact_maturity);
    const auto local = torch::nonzero(eligible).reshape({-1});
    if (local.numel() > 0) {
      const auto transformed = field->applyResidual(compact.xyz.index_select(0, local),
          compact.scaling.index_select(0, local), compact.rotation.index_select(0, local), ratio,
          elasticMaturity(compact_maturity.index_select(0, local), field->config().elastic_entry_maturity));
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
    if (selector->anchorConditioned()) selector_->anchors()->copyLayoutFrom(*selector->anchors());
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

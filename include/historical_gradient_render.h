#pragma once

#include "constrained_adam.h"
#include "selector_handoff.h"
#include "gaussian_renderer.h"
#include "transform_field.h"
#include "improvements/selector/gumbel_network.h"

struct GuardGaussianParameters {
  torch::Tensor xyz, features_dc, features_rest, opacity, log_scaling, rotation;
  int active_sh_degree = 0, max_sh_degree = 0;
  GaussianRenderInput input(const AdamStepPreview& preview,
                           const std::vector<torch::Tensor>* candidate = nullptr) const;
};

// Uses exactly the normal compact LOD population/TF rules, with fixed noise.
GaussianRenderInput historicalLoDInput(
    const GaussianRenderInput& full,
    const torch::Tensor& maturity,
    const torch::Tensor& uniform,
    const std::shared_ptr<improvements::selector::GumbelNetworkImpl>& selector,
    const std::shared_ptr<TransformFieldImpl>& field,
    float ratio, double temperature, const torch::Tensor& selection_maturity = {}, float entry = 0);

// Reusable NoGrad scratch networks, NOT a teacher or an optimizer owner.
class GuardModuleCandidates {
 public:
  void prepare(const AdamStepPreview& preview, const std::vector<torch::Tensor>& candidate,
               const std::shared_ptr<improvements::selector::GumbelNetworkImpl>& selector,
               const std::shared_ptr<TransformFieldImpl>& field);
  std::shared_ptr<improvements::selector::GumbelNetworkImpl> selector() const { return selector_; }
  std::shared_ptr<TransformFieldImpl> field() const { return field_; }
 private:
  std::weak_ptr<improvements::selector::GumbelNetworkImpl> selector_source_;
  std::weak_ptr<TransformFieldImpl> field_source_;
  std::shared_ptr<improvements::selector::GumbelNetworkImpl> selector_;
  std::shared_ptr<TransformFieldImpl> field_;
};

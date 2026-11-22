#include "include/gaussian_residual_bank.h"

#include <array>
#include <cmath>

#include "include/quaternion_utils.h"
#include "include/model_snapshot.h"

GaussianResidualBankImpl::GaussianResidualBankImpl(
    int64_t gaussian_count, int64_t hidden_dim, int64_t sh_rest_columns,
    const torch::TensorOptions& options, bool attribute_blocks) {
  TORCH_CHECK(gaussian_count > 0 && hidden_dim > 0 && sh_rest_columns >= 2,
              "Invalid attribute bank dimensions");
  const int64_t output_dim = 14 + sh_rest_columns;
  embedding_ = register_parameter("embedding", torch::zeros({gaussian_count, 16}, options));
  head_ = register_module("head", torch::nn::Linear(hidden_dim + 16, output_dim));
  head_->to(options.device());
  projection_mask_ = register_buffer("projection_mask", torch::ones_like(head_->weight));
  attribute_units_ = register_buffer("attribute_units", torch::ones({output_dim}, options));
  torch::NoGradGuard no_grad;
  head_->weight.zero_();
  head_->bias.zero_();
  attribute_units_.slice(0, 3, 6).fill_(10.0);
  attribute_units_.slice(0, 6, 10).fill_(2.0);
  attribute_units_.select(0, 10).fill_(20.0);
  attribute_units_.slice(0, 11, 14).fill_(5.0);
  attribute_units_.slice(0, 14).fill_(0.25);
  if (attribute_blocks) {
    projection_mask_.slice(1, hidden_dim).zero_();
    const std::array<std::array<int64_t, 4>, 6> blocks = {{
        {{0, 3, 0, 3}}, {{3, 6, 3, 6}}, {{6, 10, 6, 10}},
        {{10, 11, 10, 11}}, {{11, 14, 11, 14}}, {{14, output_dim, 14, 16}}}};
    for (const auto& block : blocks) {
      auto weight = head_->weight.slice(0, block[0], block[1])
                        .slice(1, hidden_dim + block[2], hidden_dim + block[3]);
      projection_mask_.slice(0, block[0], block[1])
          .slice(1, hidden_dim + block[2], hidden_dim + block[3]).fill_(1.0);
      if (block[1] - block[0] == block[3] - block[2])
        weight.copy_(torch::eye(block[1] - block[0], options));
      else
        weight.normal_(0.0, std::pow(block[3] - block[2], -0.5));
    }
  } else {
    head_->weight.slice(1, hidden_dim).normal_(0.0, 0.25);
  }
}

torch::Tensor GaussianResidualBankImpl::residual(
    const torch::Tensor& hidden, const torch::Tensor& global_indices) {
  auto input = torch::cat({hidden, embedding_.index_select(0, global_indices)}, 1);
  return torch::linear(input, head_->weight * projection_mask_, head_->bias) * attribute_units_;
}

GaussianRenderInput GaussianResidualBankImpl::apply(
    const GaussianRenderInput& base, const torch::Tensor& opacity_logits,
    const torch::Tensor& global_indices, TransformField& field, float ratio) {
  TORCH_CHECK(base.xyz.size(0) == global_indices.numel() &&
              opacity_logits.sizes() == base.opacity.sizes(),
              "Attribute bank row alignment mismatch");
  auto prediction = field->forwardWithFeatures(base.xyz, ratio);
  const auto& original = prediction.first;
  auto extra = residual(prediction.second, global_indices);
  auto delta_rotation = original.delta_rotation_raw + extra.slice(1, 6, 10);
  auto identity = torch::zeros_like(delta_rotation);
  identity.select(1, 0).fill_(1.0);
  auto rotation = quaternion_utils::normalize(quaternion_utils::multiply(
      quaternion_utils::normalize(identity + delta_rotation),
      quaternion_utils::normalize(base.rotation)));
  return {base.xyz + original.delta_xyz + extra.slice(1, 0, 3),
          (opacity_logits + extra.slice(1, 10, 11)).sigmoid(),
          base.scaling * (original.delta_log_scale + extra.slice(1, 3, 6)).exp(),
          rotation,
          base.features_dc + extra.slice(1, 11, 14).reshape_as(base.features_dc),
          base.features_rest + extra.slice(1, 14).reshape_as(base.features_rest),
          base.active_sh_degree, base.max_sh_degree};
}

void GaussianResidualBankImpl::replaceEmbedding(
    const torch::Tensor& value, const torch::Tensor& survivors, int64_t appended,
    torch::optim::Adam& optimizer, std::size_t group) {
  torch::NoGradGuard no_grad;
  auto& parameter = optimizer.param_groups().at(group).params().at(0);
  TORCH_CHECK(parameter.is_same(embedding_), "Attribute optimizer row mismatch");
  const auto key = embedding_.unsafeGetTensorImpl();
  const auto found = optimizer.state().find(key);
  std::unique_ptr<torch::optim::AdamParamState> state;
  if (found != optimizer.state().end()) {
    const auto& old = static_cast<const torch::optim::AdamParamState&>(*found->second);
    state = std::make_unique<torch::optim::AdamParamState>();
    state->step(old.step());
    const auto remap = [&](const torch::Tensor& moment) {
      return survivors.defined() ? moment.index({survivors}).clone()
          : torch::cat({moment, torch::zeros({appended, 16}, moment.options())}, 0);
    };
    state->exp_avg(remap(old.exp_avg()));
    state->exp_avg_sq(remap(old.exp_avg_sq()));
    if (old.max_exp_avg_sq().defined()) state->max_exp_avg_sq(remap(old.max_exp_avg_sq()));
    optimizer.state().erase(found);
  }
  // A fresh leaf avoids stale autograd shape metadata after append/prune.
  embedding_ = value.detach().requires_grad_(true);
  parameters_["embedding"] = embedding_;
  parameter = embedding_;
  if (state) optimizer.state()[embedding_.unsafeGetTensorImpl()] = std::move(state);
}

void GaussianResidualBankImpl::append(
    int64_t count, torch::optim::Adam& optimizer, std::size_t group) {
  if (count == 0) return;
  replaceEmbedding(torch::cat({embedding_.detach(),
      torch::zeros({count, 16}, embedding_.options())}, 0), {}, count, optimizer, group);
}

void GaussianResidualBankImpl::prune(
    const torch::Tensor& survivors, torch::optim::Adam& optimizer, std::size_t group) {
  TORCH_CHECK(survivors.numel() == gaussianCount(), "Attribute prune mask mismatch");
  replaceEmbedding(embedding_.detach().index({survivors}).clone(), survivors, 0, optimizer, group);
}

std::shared_ptr<GaussianResidualBankImpl> GaussianResidualBankImpl::cloneFrozen() const {
  ScopedTorchRNG cpu_rng(torch::Device(torch::kCPU));
  ScopedTorchRNG device_rng(embedding_.device());
  auto copy = std::make_shared<GaussianResidualBankImpl>(gaussianCount(),
      head_->weight.size(1) - 16, head_->weight.size(0) - 14, embedding_.options());
  copyFrozenModule(*this, *copy);
  return copy;
}

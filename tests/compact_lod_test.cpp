#include "include/gaussian_renderer.h"
#include "include/transform_field.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

std::array<torch::Tensor, 6> attributes(const GaussianRenderInput& input) {
  return {input.xyz, input.opacity, input.scaling, input.rotation,
          input.features_dc, input.features_rest};
}

GaussianRenderInput makeInput(const torch::TensorOptions& options) {
  return {
      torch::tensor({{-0.4f, -0.3f, 2.0f}, {-0.2f, 0.1f, 2.1f},
                     {0.0f, -0.1f, 2.2f}, {0.2f, 0.2f, 2.3f},
                     {0.4f, -0.2f, 2.4f}, {0.1f, 0.3f, 2.5f},
                     {-0.3f, 0.0f, 2.6f}, {0.3f, 0.1f, 2.7f}}, options)
          .set_requires_grad(true),
      torch::full({8, 1}, 0.6f, options).set_requires_grad(true),
      torch::full({8, 3}, 0.07f, options).set_requires_grad(true),
      torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, options).repeat({8, 1})
          .set_requires_grad(true),
      torch::full({8, 1, 3}, 0.2f, options).set_requires_grad(true),
      torch::full({8, 3, 3}, 0.03f, options).set_requires_grad(true), 1, 1};
}

torch::Tensor matureMask(const torch::TensorOptions& options) {
  return torch::tensor({true, false, true, true, true, true, false, true},
                       options.dtype(torch::kBool));
}

torch::Tensor hardMask(const torch::TensorOptions& options) {
  return torch::tensor({0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f},
                       options);
}

void indicesAndTopology() {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  const auto ids = options.dtype(torch::kLong);
  const auto mature = matureMask(options);
  const auto effective = torch::logical_or(~mature, hardMask(options) > 0);
  const auto map = torch::nonzero(effective).reshape({-1});
  check(torch::equal(map, torch::tensor({1, 3, 5, 6}, ids)),
        "Protected/selected union must use unique, ordered master rows");
  const auto master = makeInput(options);
  const auto compact = master.indexSelect(map);
  const auto master_attrs = attributes(master), compact_attrs = attributes(compact);
  for (size_t i = 0; i < master_attrs.size(); ++i)
    check(torch::equal(compact_attrs[i], master_attrs[i].index_select(0, map)),
          "Render attributes used different row maps");
  check(compact.active_sh_degree == master.active_sh_degree &&
            compact.max_sh_degree == master.max_sh_degree, "SH degrees changed");

  // TF eligibility is projected into compact space, including outside identity.
  auto tf_mask = torch::logical_and(hardMask(options).index_select(0, map) > 0,
                                    mature.index_select(0, map));
  tf_mask = torch::logical_and(tf_mask, compact.xyz.select(1, 2) < 2.4f);
  const auto tf_local = torch::nonzero(tf_mask).reshape({-1});
  check(torch::equal(tf_local, torch::tensor({1}, ids)) &&
            torch::equal(map.index_select(0, tf_local), torch::tensor({3}, ids)),
        "TF local indices did not correspond to the original global mask");
  const auto elastic = compact.xyz.index_copy(
      0, tf_local, compact.xyz.index_select(0, tf_local) + 0.1f);
  check(elastic.size(0) == 4 && torch::equal(master.xyz, makeInput(options).xyz),
        "Compact TF writeback must not mutate or expand master rows");
  const auto global_visible = map.index_select(0, torch::tensor({0, 2}, ids));
  auto supervised_iter = torch::zeros({8}, ids);
  supervised_iter.index_put_({global_visible}, 42);
  check(torch::equal(torch::nonzero(supervised_iter).reshape({-1}),
                     torch::tensor({1, 5}, ids)), "Visibility was not mapped globally");

  // Empty/all-kept cases and a fresh map after append/prune.
  const auto empty = master.indexSelect(torch::empty({0}, ids));
  for (const auto& attr : attributes(empty))
    check(attr.size(0) == 0, "Empty compact attributes are not aligned");
  check(torch::equal(master.indexSelect(torch::arange(8, ids)).xyz, master.xyz),
        "All-kept gather changed input");
  const auto appended = torch::cat({effective, torch::ones({2}, effective.options())});
  check(torch::equal(torch::nonzero(appended).reshape({-1}),
                     torch::tensor({1, 3, 5, 6, 8, 9}, ids)), "Append map is stale");
  const auto survivors = torch::tensor({1, 2, 5, 6, 8}, ids);
  check(torch::equal(torch::nonzero(appended.index_select(0, survivors)).reshape({-1}),
                     torch::tensor({0, 2, 3, 4}, ids)), "Prune map is stale");
}

void gradientRouting(bool detach) {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  auto master = makeInput(options);
  auto input = master;
  if (detach) {
    input = {master.xyz.detach(), master.opacity.detach(), master.scaling.detach(),
             master.rotation.detach(), master.features_dc.detach(),
             master.features_rest.detach(), master.active_sh_degree,
             master.max_sh_degree};
  } else {
    // A full-render backward already contributed to every master row.
    master.xyz.sum().backward();
  }
  auto logits = torch::zeros({8}, options).set_requires_grad(true);
  const auto soft = logits.sigmoid();
  const auto hard = hardMask(options) - soft.detach() + soft;
  const auto protected_f = (~matureMask(options)).to(torch::kFloat32);
  const auto mask = protected_f + (1.0f - protected_f) * hard;
  const auto map = torch::nonzero(mask.detach() > 0).reshape({-1});
  auto compact = input.indexSelect(map);
  const auto tf_local = torch::tensor({1}, options.dtype(torch::kLong));
  auto residual = torch::full({1, 3}, 0.1f, options).set_requires_grad(true);
  compact.xyz = compact.xyz.index_copy(
      0, tf_local, compact.xyz.index_select(0, tf_local) + residual);
  compact.opacity = compact.opacity * mask.index_select(0, map).unsqueeze(1);
  // Tensor-level surrogate, not a claim about CUDA rasterizer equivalence.
  auto per_row = compact.xyz.sum(1, true) + compact.scaling.sum(1, true) +
                 compact.rotation.sum(1, true) +
                 compact.features_dc.flatten(1).sum(1, true) +
                 compact.features_rest.flatten(1).sum(1, true);
  (compact.opacity * per_row).sum().backward();
  check(logits.grad().index({3}).item<float>() > 0.0f &&
            logits.grad().index({5}).item<float>() > 0.0f,
        "Selected mature straight-through photometric gradient was lost");
  check(torch::equal(logits.grad().index_select(
      0, torch::tensor({0, 1, 2, 4, 6, 7}, options.dtype(torch::kLong))),
      torch::zeros({6}, options)), "Protected/unselected photo gradient changed");
  check(residual.grad().defined() && residual.grad().abs().sum().item<float>() > 0,
        "Compact TF index_copy lost its gradient");
  const auto master_attrs = attributes(master);
  for (size_t i = 0; i < master_attrs.size(); ++i) {
    const auto& attr = master_attrs[i];
    if (detach) {
      check(!attr.grad().defined(), "Detached Gaussian received LOD gradients");
    } else {
      auto expected = torch::zeros_like(attr);
      expected.index_fill_(0, map, 0.6f);
      if (i == 1)
        expected.index_copy_(0, map, per_row.detach());
      if (i == 0) expected += 1.0f;
      check(torch::allclose(attr.grad(), expected),
            "Gather backward lost/scattered/overwrote a master-row gradient");
    }
  }
}

struct RenderResult {
  torch::Tensor image, visible, radii, mask_grad;
  std::vector<torch::Tensor> gaussian_grads, field_grads;
};

RenderResult renderCase(bool compact, bool tf, bool detach, bool cov, bool sh,
                        bool separate_sh, int resolution) {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto master = makeInput(options);
  if (tf) {
    torch::NoGradGuard no_grad;
    // This selected point starts outside the camera and TF brings it inside.
    // Culling with the full-render visibility before TF would incorrectly drop it.
    master.xyz.index_put_({5, 0}, 2.5f);
  }
  auto input = master;
  if (detach)
    input = {master.xyz.detach(), master.opacity.detach(), master.scaling.detach(),
             master.rotation.detach(), master.features_dc.detach(),
             master.features_rest.detach(), 1, 1};
  auto logits = torch::zeros({8}, options).set_requires_grad(true);
  const auto soft = logits.sigmoid();
  const auto hard = hardMask(options) - soft.detach() + soft;
  const auto protected_f = (~matureMask(options)).to(torch::kFloat32);
  const auto mask = protected_f + (1.0f - protected_f) * hard;
  const auto map = compact ? torch::nonzero(mask.detach() > 0).reshape({-1})
                           : torch::arange(8, options.dtype(torch::kLong));
  input = input.indexSelect(map);
  input.opacity = input.opacity * mask.index_select(0, map).unsqueeze(1);

  TransformField field = nullptr;
  if (tf) {
    torch::manual_seed(42); // Identical frozen initialization in the A/B runs.
    TransformFieldConfig config;
    config.hexplane.spatial_resolution = 4;
    config.hexplane.ratio_resolution = 4;
    config.hexplane.feature_dim = 2;
    config.hexplane.multires = {1};
    config.hidden_dim = 8;
    field = TransformField(torch::tensor({-2.0f, -2.0f, 1.0f}),
                           torch::tensor({3.0f, 2.0f, 4.0f}), config);
    field->to(torch::kCUDA);
    {
      torch::NoGradGuard no_grad;
      field->named_parameters()["position_out.bias"].copy_(
          torch::tensor({-2.5f, 0.0f, 0.0f}, options));
    }
    const auto eligible = torch::logical_and(
        hard.detach().index_select(0, map) > 0, matureMask(options).index_select(0, map));
    const auto local = torch::nonzero(eligible).reshape({-1});
    const auto transformed = field->applyResidual(input.xyz.index_select(0, local),
        input.scaling.index_select(0, local), input.rotation.index_select(0, local), 0.1f);
    input.xyz = input.xyz.index_copy(0, local, transformed.xyz);
    input.scaling = input.scaling.index_copy(0, local, transformed.scaling);
    input.rotation = input.rotation.index_copy(0, local, transformed.rotation);
  }
  auto camera = std::make_shared<GaussianKeyframe>(0);
  camera->FoVx_ = 1.0f;
  camera->FoVy_ = 1.0f;
  camera->world_view_transform_ = torch::eye(4, options);
  camera->full_proj_transform_ = camera->getProjectionMatrix(
      0.01f, 100.0f, 1.0f, 1.0f, torch::kCUDA).transpose(0, 1).contiguous();
  camera->camera_center_ = torch::zeros({3}, options);
  GaussianPipelineParams pipe;
  pipe.compute_cov3D_ = cov;
  pipe.convert_SHs_ = sh;
  pipe.separate_sh_ = separate_sh;
  auto background = torch::zeros({3}, options);
  torch::Tensor override_color;
  auto rendered = GaussianRenderer::render(camera, resolution, resolution, input,
      pipe, background, override_color, 1.0f, false, false);
  check(rendered.radii.numel() == map.numel() &&
            rendered.viewspace_points.size(0) == map.numel(),
        "Rasterizer buffers were not compact");
  auto visible = rendered.visibility_indices;
  visible = visible.index({mask.detach().index_select(0, map).index_select(0, visible) > 0});
  visible = map.index_select(0, visible);
  if (tf) check((visible == 5).any().item<bool>(), "TF moved-in point was culled early");
  const auto target = torch::linspace(0.0, 0.8, resolution * resolution, options)
                          .reshape({1, resolution, resolution});
  (rendered.image - target).square().mean().backward();
  RenderResult result{rendered.image.detach(), visible,
      compact ? rendered.radii : rendered.radii.index_select(
          0, torch::nonzero(mask.detach() > 0).reshape({-1})), logits.grad().clone(), {}, {}};
  for (const auto& attr : attributes(master)) {
    if (detach) check(!attr.grad().defined(), "CUDA LOD leaked detached gradients");
    else result.gaussian_grads.push_back(attr.grad().clone());
  }
  if (field)
    for (const auto& parameter : field->parameters())
      result.field_grads.push_back(parameter.grad().defined()
          ? parameter.grad().clone() : torch::zeros_like(parameter));
  return result;
}

void cudaEquivalence() {
  for (const bool tf : {false, true})
    for (const bool detach : {false, true})
      for (const int resolution : {32, 64})
        for (const int mode : {0, 1, 2, 3}) {
          const bool cov = mode == 1 || mode == 2;
          const bool sh = mode == 2;
          const bool separate = mode == 3;
          const auto dense = renderCase(false, tf, detach, cov, sh, separate, resolution);
          const auto compact = renderCase(true, tf, detach, cov, sh, separate, resolution);
          check(torch::allclose(dense.image, compact.image, 1e-4, 1e-6),
                "Dense/compact CUDA images differ");
          check(torch::equal(dense.visible, compact.visible) &&
                    torch::equal(dense.radii, compact.radii), "CUDA visibility/radii differ");
          check(torch::allclose(dense.mask_grad, compact.mask_grad, 1e-3, 1e-6),
                "Dense/compact Selector straight-through gradients differ");
          for (size_t i = 0; i < dense.gaussian_grads.size(); ++i)
            check(torch::allclose(dense.gaussian_grads[i], compact.gaussian_grads[i], 1e-3, 1e-6),
                  "Dense/compact master gradients differ");
          for (size_t i = 0; i < dense.field_grads.size(); ++i)
            check(torch::allclose(dense.field_grads[i], compact.field_grads[i], 1e-3, 1e-6),
                  "Dense/compact TF gradients differ");
        }
}
} // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--cuda") {
    if (!torch::cuda::is_available()) {
      std::cout << "SKIP: CUDA is unavailable\n";
      return 77;
    }
    cudaEquivalence();
  } else {
    indicesAndTopology();
    gradientRouting(false);
    gradientRouting(true);
  }
  return 0;
}

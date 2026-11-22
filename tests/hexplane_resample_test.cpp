#include "include/transform_field.h"
#include "improvements/selector/gumbel_network.h"
#include "include/model_snapshot.h"

#include <stdexcept>

namespace {
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void close(const torch::Tensor& a, const torch::Tensor& b) {
  check(torch::allclose(a, b, 1e-5, 1e-6), "Tensor mismatch");
}
TransformFieldConfig smallConfig() {
  TransformFieldConfig config;
  config.hexplane.spatial_resolution = 3;
  config.hexplane.ratio_resolution = 3;
  config.hexplane.feature_dim = 2;
  config.hexplane.multires = {1, 2};
  config.hidden_dim = 8;
  return config;
}

void resampling() {
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  AABB3D old_box{torch::full({3}, -1.0f, opts), torch::ones({3}, opts)};
  auto new_box = old_box.clone();
  new_box.max_xyz[0] = 3.0f;
  PlaneChangeMask changed;
  changed.xy = changed.xz = changed.xe = true;
  TransformField field(old_box.min_xyz, old_box.max_xyz, smallConfig());
  auto planes = field->gridParameters();
  std::vector<torch::Tensor> before;
  std::vector<void*> identities;
  {
    torch::NoGradGuard no_grad;
    for (const auto& p : planes) {
      auto u = torch::linspace(-1, 1, p.size(3), opts).view({1, 1, 1, -1});
      auto v = torch::linspace(-1, 1, p.size(2), opts).view({1, 1, -1, 1});
      p.copy_(2.0 + 0.25 * u + 0.1 * v);
      before.push_back(p.clone());
      identities.push_back(p.unsafeGetTensorImpl());
    }
  }
  field->resampleDomain(old_box, new_box, changed);
  close(field->aabbMax(), old_box.max_xyz);
  for (std::size_t i = 0; i < planes.size(); ++i) {
    auto p = planes[i];
    check(p.unsafeGetTensorImpl() == identities[i], "Parameter identity changed");
    check(p.sizes() == before[i].sizes(), "Plane resolution changed");
    if (!changed.at(i % 6)) {
      check(torch::equal(p, before[i]), "Unaffected plane changed");
      continue;
    }
    for (int64_t column = 0; column < p.size(3); ++column) {
      const double u_new = 2.0 * column / (p.size(3) - 1) - 1.0;
      const double u_old = 2.0 * u_new + 1.0;
      auto actual = p.select(3, column);
      if (u_old <= 1.0) {
        auto v = torch::linspace(-1, 1, p.size(2), opts).view({1, 1, -1});
        close(actual, (2.0 + 0.25 * u_old + 0.1 * v).expand_as(actual));
      } else if (i % 6 == 2) {
        close(actual, torch::ones_like(actual));
      } else {
        check(((actual >= 0.1) & (actual <= 0.5)).all().item<bool>(),
              "Spatial fringe must use normal Uniform(0.1,0.5) initialization");
      }
    }
  }
  field->setAABB(new_box);
  close(field->aabbMax(), new_box.max_xyz);
  check(field->affectedParameters(changed).size() == 6, "Missing multires planes");
  bool rejected = false;
  try { field->resampleDomain(new_box, old_box, changed); }
  catch (const c10::Error&) { rejected = true; }
  check(rejected, "Shrinking domain was accepted");
}

void frozenSnapshotsAndIdentity() {
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  TransformField field(torch::full({3}, -1.0f, opts), torch::ones({3}, opts), smallConfig());
  improvements::selector::GumbelNetwork selector;
  auto generator = at::globalContext().defaultGenerator(torch::Device(torch::kCPU));
  auto rng_before = generator.get_state();
  TransformField teacher(field->cloneFrozen());
  improvements::selector::GumbelNetwork selector_teacher(selector->cloneFrozen());
  check(torch::equal(rng_before, generator.get_state()), "Clone changed normal RNG");
  for (const auto& parameter : teacher->parameters())
    check(!parameter.requires_grad(), "TF teacher is not frozen");
  for (const auto& parameter : selector_teacher->parameters())
    check(!parameter.requires_grad(), "Selector teacher is not frozen");
  check(!teacher->is_training() && !selector_teacher->is_training(), "Teacher is in train mode");
  auto frozen = teacher->parameters().front().clone();
  auto frozen_selector = selector_teacher->parameters().front().clone();
  {
    torch::NoGradGuard no_grad;
    field->parameters().front().add_(1.0);
    selector->parameters().front().add_(1.0);
  }
  close(teacher->parameters().front(), frozen);
  close(selector_teacher->parameters().front(), frozen_selector);
  AABB3D expanded{torch::full({3}, -2.0f, opts), torch::full({3}, 2.0f, opts)};
  field->setAABB(expanded);
  close(teacher->aabbMax(), torch::ones({3}, opts));
  GaussianAttrState state{torch::tensor({{1.5f, 0.0f, 0.0f}}, opts),
                          torch::zeros({1, 3}, opts),
                          torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, opts)};
  auto identity = teacher->applyLogResidual(state, 0.3f);
  close(identity.xyz, state.xyz);
  close(identity.log_scaling, state.log_scaling);
  close(identity.rotation, state.rotation);
  check(field->insideMask(state.xyz).item<bool>(), "Old-outside/new-inside coverage failed");
  {
    ScopedTorchRNG isolated(torch::Device(torch::kCPU), 123);
    torch::rand({10});
  }
  check(torch::equal(rng_before, generator.get_state()), "Scoped initialization changed normal RNG");
}
} // namespace

int main() {
  torch::manual_seed(7);
  resampling();
  frozenSnapshotsAndIdentity();
}

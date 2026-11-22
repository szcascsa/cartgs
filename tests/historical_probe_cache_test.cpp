#include "include/historical_probe_cache.h"

#include <stdexcept>

namespace {
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
}

int main() {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  const auto longs = options.dtype(torch::kLong);
  GaussianAttrState state{torch::zeros({6, 3}, options).requires_grad_(true),
      torch::zeros({6, 3}, options).requires_grad_(true), torch::zeros({6, 4}, options)};
  state.rotation.select(1, 0).fill_(1.0f);
  state.rotation.requires_grad_(true);
  improvements::selector::GumbelNetwork selector;
  TransformFieldConfig config;
  config.hidden_dim = 8;
  config.hexplane.feature_dim = 2;
  config.hexplane.spatial_resolution = 4;
  config.hexplane.ratio_resolution = 4;
  config.hexplane.multires = {1};
  TransformField field(torch::full({3}, -2.0f, options),
                       torch::full({3}, 2.0f, options), config);
  {
    torch::NoGradGuard no_grad;
    for (const auto& p : selector->parameters()) p.zero_();
    for (const auto& p : field->parameters()) p.zero_();
    state.xyz.index_put_({5}, 3.0f); // Outside-domain TF must still be identity.
  }
  auto teacher_s = improvements::selector::GumbelNetwork(selector->cloneFrozen());
  auto teacher_tf = TransformField(field->cloneFrozen());
  HistoricalProbeCache cache(state, 1.0);
  const auto a = torch::tensor({0, 2}, longs);
  const auto b = torch::tensor({2, 4}, longs);
  const auto full = torch::arange(6, longs);
  auto probability = cache.probability(selector, 0.1f, a);
  cache.probability(selector, 0.1f, b);
  check(cache.selector_rows == 3, "Overlapping historical views recomputed Selector rows");
  cache.probability(selector, 0.1f, full);
  const auto forwards = cache.selector_forwards;
  cache.probability(selector, 0.1f, a);
  check(cache.selector_rows == 6 && cache.selector_forwards == forwards,
        "QUALITY must fill only the remaining Selector rows");
  cache.probability(selector, 0.15f, a);
  cache.probability(teacher_s, 0.1f, full);
  check(cache.selector_rows == 14, "Network/ratio caches were mixed");
  auto transformed = cache.transformed(field, 0.1f, a);
  cache.transformed(field, 0.1f, b);
  cache.transformed(field, 0.1f, full);
  check(cache.tf_rows == 6 && torch::equal(cache.transformed(field, 0.1f,
      torch::tensor({5}, longs)).xyz, state.xyz.detach().index_select(0, torch::tensor({5}, longs))),
        "TF row reuse or outside identity changed");
  check(!probability.requires_grad() && !transformed.xyz.requires_grad() &&
            !state.xyz.grad().defined(), "Probe cache leaked a training graph");
  const auto empty = torch::empty({0}, longs);
  check(cache.probability(selector, 0.1f, empty).numel() == 0 &&
            cache.transformed(field, 0.1f, empty).xyz.size(0) == 0,
        "Empty probe queries must not require an initialized target cache");

  int renders = 0;
  auto render = [&] { ++renders; return HistoricalRenderQuality{20.0, 2, 3}; };
  cache.quality(7, 0.1f, selector, field, render);
  cache.quality(7, 0.1f, selector, field, render);
  cache.quality(8, 0.1f, selector, field, render);
  cache.quality(7, 0.15f, selector, field, render);
  cache.quality(7, 0.1f, teacher_s, teacher_tf, render);
  check(renders == 4 && cache.quality_hits == 1,
        "Quality key must distinguish view, ratio and both modules");
  const auto old_field_forwards = cache.tf_forwards;
  {
    torch::NoGradGuard no_grad;
    selector->named_parameters()["soft_net.bias"][1].fill_(2.0f);
  }
  cache.invalidateStudent(selector, field, true, false);
  check(cache.revision() == 1 && cache.selectorRevision() == 1 && cache.tfRevision() == 0 &&
            (cache.probability(selector, 0.1f, a) > 0.8f).all().item<bool>() &&
            torch::equal(cache.probability(teacher_s, 0.1f, a), torch::full({2}, 0.5f, options)),
        "Live Student mutation reused stale probabilities or erased Teacher identity");
  cache.transformed(field, 0.1f, a);
  check(cache.tf_forwards == old_field_forwards, "Selector update invalidated unchanged TF targets");
  cache.quality(7, 0.1f, teacher_s, teacher_tf, render);
  cache.quality(7, 0.1f, selector, field, render);
  check(renders == 5, "Student update must rerender Student but preserve Teacher PSNR");
  cache.transformed(teacher_tf, 0.1f, full);
  {
    torch::NoGradGuard no_grad;
    field->named_parameters()["position_out.bias"].fill_(0.5f);
  }
  cache.invalidateStudent(selector, field, false, true);
  check(cache.selectorRevision() == 1 && cache.tfRevision() == 1,
        "Module revisions must invalidate prepared inputs independently");
  check(torch::allclose(cache.transformed(field, 0.1f, a).xyz,
                        torch::full({2, 3}, 0.5f, options)) &&
            torch::equal(cache.transformed(teacher_tf, 0.1f, a).xyz, torch::zeros({2, 3}, options)),
        "TF optimizer update reused stale output or changed frozen Teacher targets");

  // Temporary EMA candidates cannot be destroyed and have their addresses reused
  // while their cached outputs/quality still exist.
  std::weak_ptr<improvements::selector::GumbelNetworkImpl> candidate_ptr;
  {
    HistoricalProbeCache local(state, 1.0);
    auto candidate = improvements::selector::GumbelNetwork(selector->cloneFrozen());
    candidate_ptr = candidate.ptr();
    local.quality(7, 0.1f, candidate, field, render);
    candidate = nullptr;
    check(!candidate_ptr.expired(), "Cached candidate identity lost model ownership");
  }
  check(candidate_ptr.expired(), "Event-local cache kept a candidate beyond probe lifetime");
  HistoricalProbeCache next_probe(state, 1.0);
  next_probe.probability(selector, 0.1f, a);
  check(next_probe.selector_rows == 2 && next_probe.quality_renders == 0,
        "Probe cache escaped into a later snapshot");
  // QUALITY uses one shared Bernoulli draw, not soft opacity or Top-K.
  const auto draw = torch::tensor({0.1f, 0.6f, 0.3f, 0.9f, 0.2f, 0.8f}, options);
  const auto p_old = cache.probability(teacher_s, 0.1f, full);
  const auto p_current = cache.probability(selector, 0.1f, full);
  check((draw < p_old).equal(torch::tensor({true, false, true, false, true, false})),
        "Shared-uniform QUALITY changed the frozen teacher selection");
  check(((draw < p_old) & ~(draw < p_current)).sum().item<int64_t>() == 0,
        "A higher keep probability lost points with common random numbers");
  const auto generator = at::globalContext().defaultGenerator(torch::Device(torch::kCPU));
  const auto saved_rng = generator.get_state();
  torch::Tensor random_a, random_b;
  { ScopedTorchRNG rng(torch::Device(torch::kCPU), 123); random_a = torch::rand({6}); }
  { ScopedTorchRNG rng(torch::Device(torch::kCPU), 123); random_b = torch::rand({6}); }
  check(random_a.equal(random_b) && generator.get_state().equal(saved_rng),
        "Probe selection advanced the normal training RNG or was not reproducible");
  return 0;
}

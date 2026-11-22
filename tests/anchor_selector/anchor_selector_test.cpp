#include "improvements/selector/gumbel_network.h"
#include "include/selector_handoff.h"
#include "include/gi_teacher.h"
#include <chrono>
#include <iostream>
#include <sstream>

using improvements::selector::GumbelNetwork;
static void check(bool value, const char* reason) { TORCH_CHECK(value, reason); }

int main(int argc, char** argv) {
  const auto device = argc > 1 && std::string(argv[1]) == "cuda" ? torch::kCUDA : torch::kCPU;
  torch::manual_seed(2026);
  auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
  auto xyz = torch::tensor({{.01f,.02f,.03f},{.08f,.04f,.07f},{.21f,.1f,.12f},{-.03f,.02f,.01f}}, options);
  auto rotation = torch::randn({4, 4}, options), scale = torch::ones({4, 3}, options);
  auto model = GumbelNetwork(.1);
  model->to(device);
  model->anchors()->reset(xyz);
  check(model->anchors()->size() == 3, "Voxel initialization");
  check(model->anchors()->rows()[0].item<int64_t>() == model->anchors()->rows()[1].item<int64_t>(), "Feature sharing");
  auto input = model->anchors()->inputs(xyz);
  check(torch::equal(input.first[0], input.first[1]), "Shared feature lookup");
  auto probability = model->forwardProbability(xyz, rotation, scale, .35f);
  check(!torch::equal(probability[0], probability[1]), "Offsets must distinguish same-anchor Gaussians");
  check(torch::equal(probability, model->forwardProbability(xyz, rotation*9, scale*8, .35f)), "View/attribute independence");
  check(!torch::equal(probability, model->forwardProbability(xyz, rotation, scale, .9f)), "Ratio conditioning");
  auto compact = torch::tensor({3, 1}, torch::TensorOptions().device(device).dtype(torch::kLong));
  check(torch::equal(probability.index_select(0, compact), model->forwardProbability(
      xyz.index_select(0, compact), rotation.index_select(0, compact), scale.index_select(0, compact), .35f, 1, compact)), "Compact row identity");

  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(.001));
  probability.sum().backward();
  check(model->anchors()->features().grad().abs().sum().item<double>() > 0, "Anchor feature gradient");
  optimizer.step(); optimizer.zero_grad(true);
  // Optional SI supervision uses the existing detached teacher and all-row ST
  // mask; the network inputs and cached Bernoulli export stay unchanged.
  const auto si_teacher = GITeacherBuilder().build(torch::tensor({4.f,1.f,3.f,2.f}, options),
      torch::ones({4}, options.dtype(torch::kBool)), torch::ones({4}, options.dtype(torch::kBool)),
      torch::ones({4}, options), .5f, 1);
  check(si_teacher.valid() && si_teacher.supervised_k == 4 && !si_teacher.label.requires_grad(), "Detached SI teacher");
  auto si_mask = sampleSelectorST(model->forwardProbability(xyz, {}, {}, .5f), torch::zeros({4}, options));
  si_teacher.loss(si_mask, false).backward();
  check(model->anchors()->features().grad().isfinite().all().item<bool>() &&
        model->anchors()->features().grad().abs().sum().item<double>() > 0, "SI must reach Anchor features");
  check(model->named_parameters()["soft_net.weight"].grad().abs().sum().item<double>() > 0,
        "SI must reach Selector MLP");
  optimizer.zero_grad(true);
  auto feature_before = model->anchors()->features().detach().clone();
  auto state_key = model->anchors()->features().unsafeGetTensorImpl();
  auto& state = static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(state_key));
  auto moment_before = state.exp_avg().clone();
  model->anchors()->append(torch::tensor({{.31f,.01f,.02f}}, options), {&optimizer});
  check(!optimizer.state().count(state_key) && optimizer.state().count(model->anchors()->features().unsafeGetTensorImpl()), "Registered leaf/Adam state migration");
  bool optimizer_alias = false;
  for (const auto& parameter : optimizer.parameters()) optimizer_alias |= parameter.is_same(model->anchors()->features());
  check(optimizer_alias, "Optimizer must own the new leaf");
  check(torch::equal(feature_before, model->anchors()->features().slice(0,0,3)), "Feature growth preservation");
  check(torch::equal(moment_before, state.exp_avg().slice(0,0,3)) && state.exp_avg()[3].abs().sum().item<double>() == 0, "Adam growth preservation");
  xyz = torch::cat({xyz, torch::tensor({{.31f,.01f,.02f}}, options)});
  auto survivors = torch::tensor({4, 1, 0}, compact.options());
  model->anchors()->selectRows(survivors, {&optimizer});
  xyz = xyz.index_select(0, survivors);
  check(model->anchors()->size() == 2 && model->anchors()->rows().numel() == 3, "Prune/reorder association");
  auto teacher = model->cloneFrozen();
  auto before = teacher->forwardProbability(xyz, {}, {}, .35f).detach();
  auto rng_before = at::globalContext().defaultGenerator(torch::kCPU).get_state();
  auto clone = model->cloneFrozen();
  check(torch::equal(rng_before, at::globalContext().defaultGenerator(torch::kCPU).get_state()), "Snapshot RNG isolation");
  check(torch::equal(before, clone->forwardProbability(xyz, {}, {}, .35f)), "Frozen snapshot");
  auto moved = xyz + torch::tensor({.4f,0.f,0.f}, options);
  auto old_inputs = model->anchors()->inputs(xyz).first.detach().clone();
  model->anchors()->reassociate(moved, {&optimizer});
  check(torch::allclose(old_inputs, model->anchors()->inputs(moved).first, 1e-6, 1e-7), "Geometry feature transport");
  check(torch::equal(before, teacher->forwardProbability(xyz, {}, {}, .35f)), "Teacher uses old association");
  auto loss = (model->forwardProbability(moved, {}, {}, .35f) - before).square().mean();
  loss.backward(); optimizer.step(); optimizer.zero_grad(true);
  check(torch::isfinite(model->anchors()->features()).all().item<bool>(), "Post-event Adam state");

  std::stringstream stream;
  torch::serialize::OutputArchive saved;
  model->save(saved); saved.save_to(stream);
  auto restored = GumbelNetwork(.1); restored->to(device);
  torch::serialize::InputArchive archive; archive.load_from(stream); restored->load(archive);
  check(torch::equal(model->forwardProbability(moved, {}, {}, .35f), restored->forwardProbability(moved, {}, {}, .35f)), "Archive roundtrip");
  // Full stage must not change Selector or Anchor momentum when gradients are undefined.
  const auto frozen = model->anchors()->features().detach().clone();
  optimizer.step();
  check(torch::equal(frozen, model->anchors()->features()), "Full-stage undefined-gradient skip");
  torch::serialize::OutputArchive exported;
  model->save(exported); exported.save_to("selector_test.pt");
  torch::serialize::OutputArchive probe;
  probe.write("xyz", moved.detach());
  auto maturity = torch::ones({3}, options);
  auto uniform = torch::tensor({.1f,.5f,.9f}, options);
  probe.write("maturity", maturity); probe.write("uniform", uniform);
  const std::vector<float> ratios{.01f,.05f,.1f,.15f};
  for (int r=0;r<4;++r) {
    auto p = model->forwardProbability(moved, {}, {}, ratios[r]);
    probe.write("probability_" + std::to_string(r), p.detach());
    probe.write("mask_" + std::to_string(r), (uniform < mixSelectorProbability(p, maturity, ratios[r])).detach());
  }
  probe.save_to("selector_probe.pt");

  // Small forward/backward benchmark: not a production speedup claim.
  auto large_xyz = torch::rand({20000,3}, options) * 2;
  auto bench = GumbelNetwork(.1); bench->to(device);
  if (device == torch::kCUDA) torch::cuda::synchronize();
  const auto init_start = std::chrono::steady_clock::now();
  bench->anchors()->reset(large_xyz);
  if (device == torch::kCUDA) torch::cuda::synchronize();
  const auto init_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-init_start).count();
  for (int i=0;i<5;++i) { bench->zero_grad(true); bench->forwardProbability(large_xyz, {}, {}, .1f).mean().backward(); }
  if (device == torch::kCUDA) torch::cuda::synchronize();
  const auto start = std::chrono::steady_clock::now();
  for (int i=0;i<20;++i) { bench->zero_grad(true); bench->forwardProbability(large_xyz, {}, {}, .1f).mean().backward(); }
  if (device == torch::kCUDA) torch::cuda::synchronize();
  const auto ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/20;
  auto legacy = GumbelNetwork(); legacy->to(device);
  const auto large_rotation = torch::randn({20000,4}, options);
  const auto large_scale = torch::ones({20000,3}, options);
  for (int i=0;i<5;++i) { legacy->zero_grad(true); legacy->forwardProbability(large_xyz, large_rotation, large_scale, .1f).mean().backward(); }
  if (device == torch::kCUDA) torch::cuda::synchronize();
  const auto legacy_start = std::chrono::steady_clock::now();
  for (int i=0;i<20;++i) { legacy->zero_grad(true); legacy->forwardProbability(large_xyz, large_rotation, large_scale, .1f).mean().backward(); }
  if (device == torch::kCUDA) torch::cuda::synchronize();
  const auto legacy_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-legacy_start).count()/20;
  std::cout << "{\"passed\":true,\"device\":\"" << (device == torch::kCUDA ? "cuda" : "cpu")
            << "\",\"population\":20000,\"anchors\":" << bench->anchors()->size()
            << ",\"initialization_ms\":" << init_ms << ",\"forward_backward_ms\":" << ms
            << ",\"legacy_libtorch_forward_backward_ms\":" << legacy_ms
            << ",\"feature_bytes\":" << bench->anchors()->features().numel()*4 << "}\n";
}

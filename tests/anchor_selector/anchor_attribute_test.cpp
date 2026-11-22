#include "improvements/selector/gumbel_network.h"
#include "include/selector_handoff.h"
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
  const auto device = argc > 1 && std::string(argv[1]) == "cuda" ? torch::kCUDA : torch::kCPU;
  torch::manual_seed(2026);
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto xyz = torch::tensor({{.01f,.02f,.03f},{.08f,.04f,.07f},{.21f,.1f,.12f},{-.03f,.02f,.01f}}, options);
  auto scale = torch::full({4,3}, .02f, options).requires_grad_(true);
  auto dc = torch::randn({4,1,3}, options).requires_grad_(true);
  auto opacity = torch::full({4,1}, .5f, options).requires_grad_(true);
  auto rotation = torch::randn({4,4}, options);
  const bool global_xyz=argc>2 && std::string(argv[2])=="global_xyz";
  auto model = improvements::selector::GumbelNetwork(.1, true, global_xyz); model->to(device);
  model->anchors()->reset(xyz);
  auto p = model->forwardProbability(xyz, rotation, scale, .05f, 1, {}, dc, opacity);
  TORCH_CHECK(model->named_parameters()["local_emd.0.weight"].size(1) == (global_xyz?30:27), "Input width");
  TORCH_CHECK(torch::equal(p, model->forwardProbability(xyz, rotation*10, scale, .05f, 1, {}, dc, opacity)), "Rotation/view independence");
  for (int attr=0; attr<3; ++attr) {
    auto varied = model->forwardProbability(xyz, rotation, attr==0 ? scale+1 : scale,
        .05f, 1, {}, attr==1 ? dc+1 : dc, attr==2 ? opacity+.2 : opacity);
    TORCH_CHECK(!torch::equal(p, varied), "New Gaussian attribute did not affect scores");
  }
  auto indices = torch::tensor({3,1}, options.dtype(torch::kLong));
  auto compact = model->forwardAnchorProbability(xyz.index_select(0, indices), .05f, 1, indices,
      scale.index_select(0, indices), dc.index_select(0, indices), opacity.index_select(0, indices));
  TORCH_CHECK(torch::allclose(compact, p.index_select(0, indices), 1e-6, 1e-7), "Compact attributes/source rows mismatch");
  auto many = model->forwardProbabilities(xyz, rotation, scale, {.01f,.05f,.1f,.15f}, 1, {}, dc, opacity);
  TORCH_CHECK(torch::allclose(many[1], p, 1e-6, 1e-7), "Change Distill multi-ratio forward mismatch");
  sampleSelectorST(p, torch::zeros({4}, options)).sum().backward();
  for (const auto& input : {scale, dc, opacity, model->anchors()->features()})
    TORCH_CHECK(input.grad().defined() && input.grad().isfinite().all().item<bool>() &&
                input.grad().abs().sum().item<double>() > 0, "Missing Gaussian/Anchor input gradient");
  auto teacher = model->cloneFrozen();
  TORCH_CHECK(teacher->anchorGaussianAttributes() && torch::equal(p,
      teacher->forwardProbability(xyz, rotation, scale, .05f, 1, {}, dc, opacity)), "Frozen teacher layout");
  std::stringstream bytes;
  torch::serialize::OutputArchive saved; model->save(saved); saved.save_to(bytes);
  auto restored = improvements::selector::GumbelNetwork(.1, true, global_xyz); restored->to(device);
  torch::serialize::InputArchive loaded; loaded.load_from(bytes); restored->load(loaded);
  TORCH_CHECK(torch::equal(p, restored->forwardProbability(xyz, rotation, scale, .05f, 1, {}, dc, opacity)), "Archive roundtrip");
  torch::serialize::OutputArchive checkpoint; model->save(checkpoint); checkpoint.save_to("attribute_selector_test.pt");
  torch::serialize::OutputArchive probe;
  probe.write("xyz", xyz); probe.write("scale", scale.detach()); probe.write("sh_dc", dc.detach());
  probe.write("opacity", opacity.detach()); probe.write("uniform", torch::tensor({.1f,.3f,.6f,.9f}, options));
  for (int r=0; r<4; ++r) probe.write("probability_"+std::to_string(r), many[r].detach());
  probe.save_to("attribute_selector_probe.pt");
  std::cout << "PASS: 27 inputs, attribute dependence/gradients, view independence, compact rows, multi-ratio, frozen clone, checkpoint\n";
}

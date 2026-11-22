#include "improvements/selector/gumbel_network.h"
#include "include/selector_handoff.h"
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
  try {
    const auto device = argc > 1 && std::string(argv[1]) == "cuda" ? torch::kCUDA : torch::kCPU;
    const auto output = std::filesystem::path(argc > 2 ? argv[2] : "flexgs_sampling_test");
    std::filesystem::create_directories(output);
    torch::set_num_threads(2); torch::manual_seed(20261028);
    auto o = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto model = improvements::selector::GumbelNetwork(.1, true, false, true, std::vector<double>{}, 0);
    model->to(device); model->anchors()->enableVariableMembers(); model->enableSpatialCompensation();
    auto observed = torch::tensor({{.01f,.02f,2.01f},{.03f,.02f,2.01f},{.21f,.02f,2.01f}}, o);
    auto seeds = model->anchors()->observeBlocks(observed, torch::full_like(observed, .4));
    auto xyz = seeds.xyz.detach().clone().requires_grad_(true);
    const auto n = xyz.size(0);
    auto scale = torch::full_like(xyz, .02).requires_grad_(true);
    auto dc = torch::randn({n,1,3}, o).requires_grad_(true);
    auto opacity = torch::full({n,1}, .5f, o).requires_grad_(true);
    { torch::NoGradGuard guard; model->named_parameters()["soft_net.weight"].normal_(0, .1); }
    auto selected = model->forwardSelectionFeatures(xyz, .05f, 1, {}, scale, dc, opacity);
    auto raw = model->forwardLogits(xyz, {}, scale, .05f, {}, dc, opacity);
    TORCH_CHECK(torch::equal(selected.logits, raw), "Raw logits changed the current MLP");
    TORCH_CHECK(torch::equal(selected.probability, model->forwardAnchorProbability(xyz,.05f,1,{},scale,dc,opacity)),
                "Deterministic probability regression");
    auto noise = -torch::empty_like(raw).exponential_().log();
    auto logits = raw.detach().clone().requires_grad_(true);
    auto mask = sampleSelectorGumbelST(logits, 1, noise);
    auto vjp = torch::randn({n}, o);
    auto gradient = torch::autograd::grad({mask}, {logits}, {vjp})[0];
    auto normalized_mask = sampleSelectorGumbelST(logits, 1, noise);
    auto normalized_budget = selectorGumbelRatioLoss(normalized_mask, .05f, true);
    auto normalized_gradient = torch::autograd::grad({normalized_budget}, {logits})[0];
    TORCH_CHECK(((mask == 0) | (mask == 1)).all().item<bool>(), "Forward must be binary");
    TORCH_CHECK(torch::equal(mask, sampleSelectorGumbelST(logits, .3, noise)) &&
                torch::equal(mask, sampleSelectorGumbelST(logits, 2, noise)), "Hard mask temperature invariance");
    auto train_mask = sampleSelectorGumbelST(selected.logits, 1, noise);
    auto hard_budget = selectorGumbelRatioLoss(train_mask, .05f);
    ((train_mask*vjp).sum() + .01*hard_budget).backward();
    TORCH_CHECK(xyz.grad().abs().sum().item<double>() > 0 && scale.grad().abs().sum().item<double>() > 0 &&
                dc.grad().abs().sum().item<double>() > 0 && opacity.grad().abs().sum().item<double>() > 0,
                "Joint Gaussian input gradients missing");
    torch::save(model, (output/"selector.pt").string());
    torch::serialize::OutputArchive fixture;
    fixture.write("xyz", xyz.detach(), true); fixture.write("scale", scale.detach(), true);
    fixture.write("dc", dc.detach(), true); fixture.write("opacity", opacity.detach(), true);
    fixture.write("probability", selected.probability.detach(), true); fixture.write("logits", raw.detach(), true);
    fixture.write("noise", noise, true); fixture.write("mask", mask.detach(), true);
    fixture.write("gradient", gradient, true); fixture.write("vjp", vjp, true);
    fixture.write("normalized_budget", normalized_budget.detach(), true);
    fixture.write("normalized_gradient", normalized_gradient, true);
    fixture.write("budget", hard_budget.detach(), true);
    for (const auto& value : std::vector<std::pair<std::string,torch::Tensor>>{
             {"xyz_gradient",xyz.grad()}, {"scale_gradient",scale.grad()},
             {"dc_gradient",dc.grad()}, {"opacity_gradient",opacity.grad()}})
      fixture.write(value.first,value.second,true);
    fixture.save_to((output/"fixture.pt").string());
    auto empty = sampleSelectorGumbelST(torch::empty({0,2},o),1);
    TORCH_CHECK(empty.numel() == 0, "Empty candidate set");
    auto many = torch::zeros({4096,2},o);
    TORCH_CHECK(!torch::equal(sampleSelectorGumbelST(many,1),sampleSelectorGumbelST(many,1)),
                "Training reused a random mask");
    auto uniform = torch::rand({n},o);
    auto legacy = sampleSelectorST(selected.probability.detach(),uniform);
    TORCH_CHECK(torch::equal(legacy,(uniform<selected.probability.detach()).to(torch::kFloat32)),"Legacy sampling regression");
    std::cout << "PASS flexgs_sampling device=" << (device==torch::kCUDA?"cuda":"cpu")
              << " gaussian_rows=" << n << " selected=" << mask.sum().item<int64_t>() << '\n';
    return 0;
  } catch (const c10::Error& error) { std::cerr << error.what() << '\n'; return 1; }
}

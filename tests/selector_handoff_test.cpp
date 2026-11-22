#include "include/selector_handoff.h"
#include "include/protection_config.h"
#include <iostream>
#include <stdexcept>

void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
int main(int argc, char** argv) {
  const bool cuda = argc > 1 && std::string(argv[1]) == "--cuda";
  if (cuda && !torch::cuda::is_available()) return 77;
  const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(cuda ? torch::kCUDA : torch::kCPU);
  auto p = torch::tensor({.8f, .8f, .8f}, opts.requires_grad(true));
  auto m = torch::tensor({0.f, .5f, 1.f}, opts.requires_grad(true));
  const auto mix = mixSelectorProbability(p, m, .1f);
  check(torch::allclose(mix, torch::tensor({.1f, .45f, .8f}, opts)), "G: continuous handoff failed");
  const auto mask = sampleSelectorST(mix, torch::tensor({.05f, .5f, .5f}, opts));
  check(mask.equal(torch::tensor({1.f, 0.f, 1.f}, opts)), "I: hard ST forward is not binary");
  mask.sum().backward();
  check(p.grad().equal(m.detach()) && !m.grad().defined(), "H: Selector gradient is not maturity-scaled/detached");
  const auto probability = torch::tensor({.6f, .8f}, opts);
  check(std::abs(selectorRatioLoss(probability, .1f).item<float>() - 1.2f) < 1e-6,
        "J: ratio loss does not use all-row expected probability");
  for (const float ratio : {.01f, .05f, .10f, .15f}) {
    const auto neutral = mixSelectorProbability(torch::rand({1000}, opts), torch::zeros({1000}, opts), ratio);
    check(std::abs(neutral.sum().item<double>() - ratio * 1000) < 1e-4 &&
          selectorRatioLoss(neutral, ratio).item<float>() < 1e-5, "J: neutral expected budget changed");
  }
  ProtectionConfig config; config.validate();
  config.beta_slow = config.beta_fast;
  bool rejected = false;
  try { config.validate(); } catch (const std::invalid_argument&) { rejected = true; }
  check(rejected, "Invalid asymmetric EMA accepted");
  std::cout << "PASS G/H/I/J handoff, detached gradient, hard ST, all-row expected budget\n";
}

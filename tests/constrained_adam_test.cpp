#include "include/constrained_adam.h"
#include <stdexcept>

namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void parity(bool amsgrad, double decay) {
  const auto opts = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
  auto a = torch::tensor({0.3f, -0.7f}, opts), b = a.detach().clone().set_requires_grad(true);
  auto c = torch::tensor({1.0f}, opts), d = c.detach().clone().set_requires_grad(true);
  auto absent = torch::ones({1}, opts);
  const auto options = torch::optim::AdamOptions(0.003).eps(1e-15).amsgrad(amsgrad).weight_decay(decay);
  torch::optim::Adam original({a}, options), guarded({b}, options);
  original.add_param_group(std::vector<torch::Tensor>{c});
  guarded.add_param_group(std::vector<torch::Tensor>{d, absent});
  original.param_groups()[1].options().set_lr(0.0001);
  guarded.param_groups()[1].options().set_lr(0.0001);
  for (int step = 0; step < 8; ++step) {
    a.mutable_grad() = torch::full_like(a, step == 2 ? 0.0 : 0.02 * (step + 1));
    b.mutable_grad() = a.grad().clone();
    c.mutable_grad() = step == 3 ? torch::Tensor() : torch::full_like(c, -0.3);
    d.mutable_grad() = c.grad().defined() ? c.grad().clone() : torch::Tensor();
    const auto before = b.detach().clone(), normal_grad = b.grad().clone();
    const auto state_count = guarded.state().size();
    AdamStepPreview preview(guarded);
    check(b.equal(before) && b.grad().equal(normal_grad) && guarded.state().size() == state_count,
          "Preview mutated live parameters, gradients or state");
    original.step();
    preview.commit(preview.candidate(preview.delta(), 1.0));
    check(torch::allclose(a, b, 1e-7, 1e-8) && torch::allclose(c, d, 1e-7, 1e-8),
          "Preview/commit differs from LibTorch Adam");
    const auto& s1 = static_cast<const torch::optim::AdamParamState&>(*original.state().at(a.unsafeGetTensorImpl()));
    const auto& s2 = static_cast<const torch::optim::AdamParamState&>(*guarded.state().at(b.unsafeGetTensorImpl()));
    check(s1.step() == s2.step() && s1.exp_avg().equal(s2.exp_avg()) && s1.exp_avg_sq().equal(s2.exp_avg_sq()),
          "Preview advanced Adam moments/counter incorrectly");
    check(guarded.state().count(absent.unsafeGetTensorImpl()) == 0, "grad=None parameter acquired Adam state");
  }
  const auto before = b.detach().clone();
  const auto& state = static_cast<const torch::optim::AdamParamState&>(*guarded.state().at(b.unsafeGetTensorImpl()));
  const auto moment = state.exp_avg().clone(); const auto step = state.step();
  { AdamStepPreview rejected(guarded); rejected.candidate(rejected.delta(), 0.5); }
  check(b.equal(before) && state.exp_avg().equal(moment) && state.step() == step,
        "Discarded candidate changed parameters or Adam state");
  // Topology replacement creates a new population/key, not a stale cached block.
  auto replacement = b.detach().slice(0, 0, 1).clone().set_requires_grad(true);
  guarded.param_groups()[0].params()[0] = replacement;
  AdamStepPreview fresh(guarded);
  check(fresh.value(replacement, fresh.candidate(fresh.delta(), 1.0)).equal(replacement),
        "grad=None topology replacement must not receive a historical update");
}
}
int main() { parity(false, 0.0); parity(false, 0.1); parity(true, 0.1); }

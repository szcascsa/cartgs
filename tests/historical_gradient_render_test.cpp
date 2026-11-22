#include "include/historical_gradient_render.h"
#include "include/gradient_constraint.h"
#include <torch/csrc/autograd/autograd.h>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <string>

namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void run(torch::Device device, bool render) {
  const auto opts = torch::TensorOptions().device(device).dtype(torch::kFloat32);
  GuardGaussianParameters raw{
      torch::tensor({{-0.1f, 0.0f, 2.0f}, {0.1f, 0.0f, 2.0f}, {0.0f, 0.1f, 2.1f}}, opts).set_requires_grad(true),
      torch::full({3, 1, 3}, 0.2f, opts).set_requires_grad(true),
      torch::full({3, 3, 3}, 0.01f, opts).set_requires_grad(true),
      torch::zeros({3, 1}, opts).set_requires_grad(true),
      torch::full({3, 3}, -2.7f, opts).set_requires_grad(true),
      torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, opts).repeat({3, 1}).set_requires_grad(true), 1, 1};
  auto selector = std::make_shared<improvements::selector::GumbelNetworkImpl>(); selector->to(device);
#ifdef CARTGS_TEST_SELECTOR_CUDA
  selector->setBackend("cuda");
#endif
  TransformFieldConfig config; config.hexplane.spatial_resolution = 4; config.hexplane.ratio_resolution = 4;
  config.hexplane.feature_dim = 2; config.hexplane.multires = {1}; config.hidden_dim = 4;
  auto field = std::make_shared<TransformFieldImpl>(torch::full({3}, -3.0f, opts), torch::full({3}, 3.0f, opts), config);
  field->to(device);
  std::vector<torch::Tensor> params = {raw.xyz, raw.features_dc, raw.features_rest, raw.opacity, raw.log_scaling, raw.rotation};
  const auto sparams = selector->parameters(), fparams = field->parameters();
  params.insert(params.end(), sparams.begin(), sparams.end()); params.insert(params.end(), fparams.begin(), fparams.end());
  torch::optim::Adam optimizer(params, torch::optim::AdamOptions(1e-4));
  for (auto& p : params) p.mutable_grad() = torch::ones_like(p);
  AdamStepPreview preview(optimizer);
  const auto original_grad = raw.xyz.grad().clone();
  const auto mature = torch::tensor({true, false, true}, opts.dtype(torch::kBool));
  const auto mature_indices = torch::nonzero(mature).reshape({-1});
  const auto maturity = mature.to(torch::kFloat32);
  const auto uniform = torch::zeros({3}, opts);
  const auto generator = at::globalContext().defaultGenerator(device);
  const auto rng_before = generator.get_state().clone();
  const auto full = raw.input(preview);
  {
    ScopedTorchRNG restore(device);
    torch::Tensor probability_a, probability_b;
    selector->forwardHard(full.xyz, full.rotation, full.scaling, 0.1f, 0.5, &probability_a);
    selector->forwardHard(full.xyz, full.rotation, full.scaling, 0.1f, 2.0, &probability_b);
    check(probability_a.requires_grad() && torch::allclose(probability_a, probability_b) &&
          torch::allclose(probability_a, selector->forwardProbability(full.xyz, full.rotation, full.scaling, 0.1f)),
          "Reused expected count changed with Gumbel noise/temperature or lost gradients");
  }
  const auto a = selector->forwardHardWithUniform(full.xyz, full.rotation, full.scaling, 0.1f, uniform, 0.5);
  const auto b = selector->forwardHardWithUniform(full.xyz, full.rotation, full.scaling, 0.1f, uniform, 2.0);
  check(a.equal(b) && a.equal((uniform < selector->forwardProbability(full.xyz, full.rotation, full.scaling, 0.1f)).to(torch::kFloat32)),
        "Fixed uniform forward is not the temperature-independent Gumbel hard law");
  check(generator.get_state().equal(rng_before), "Fixed-uniform forward consumed training RNG");
  const auto lod = historicalLoDInput(full, maturity, uniform, selector, field, 0.1f, 1.0);
  check(lod.xyz[1].equal(full.xyz[1]) && lod.opacity[1].equal(full.opacity[1]), "TF compatibility identity rule changed");
  const auto surrogate = (lod.opacity * lod.xyz.sum(1, true)).sum();
  const auto grads = torch::autograd::grad({surrogate}, {raw.xyz, raw.log_scaling, raw.rotation, sparams.back(), fparams.back()}, {}, false, false, true);
  check(grads[0].defined() && grads[0].abs().sum().item<float>() > 0.0f && grads[3].defined(),
        "Historical LOD detached Gaussian or Selector input gradients");
  check(raw.xyz.grad().equal(original_grad), "Historical autograd changed normal Gaussian gradients");
  GuardModuleCandidates scratch;
  {
    torch::NoGradGuard no_grad;
    auto zero = preview.delta(); for (auto& d : zero) d = torch::zeros_like(d);
    const auto candidate = preview.candidate(zero, 1.0);
    scratch.prepare(preview, candidate, selector, field);
    const auto cfull = raw.input(preview, &candidate);
    const auto clod = historicalLoDInput(cfull, maturity, uniform, scratch.selector(), scratch.field(), 0.1f, 1.0);
    check(torch::allclose(lod.xyz, clod.xyz) && lod.opacity.equal(clod.opacity), "Zero candidate changed render inputs");
    check(!clod.xyz.requires_grad() && generator.get_state().equal(rng_before), "Candidate retained graph or consumed RNG");
    const auto scratch_ptr = scratch.field();
    field->setAABB({torch::full({3}, -4.0f, opts), torch::full({3}, 4.0f, opts)});
    scratch.prepare(preview, candidate, selector, field);
    check(scratch.field() == scratch_ptr && scratch.field()->aabbMin().equal(field->aabbMin()), "Scratch reuse kept a stale AABB");
    if (render) {
      auto camera = std::make_shared<GaussianKeyframe>(0);
      camera->FoVx_ = camera->FoVy_ = 1.0f; camera->world_view_transform_ = torch::eye(4, opts);
      camera->full_proj_transform_ = camera->getProjectionMatrix(0.01f, 100.0f, 1.0f, 1.0f, torch::kCUDA).transpose(0, 1).contiguous();
      camera->camera_center_ = torch::zeros({3}, opts);
      GaussianPipelineParams pipe; auto background = torch::zeros({3}, opts); torch::Tensor color;
      for (const auto& input : {full, lod}) {
        const auto training = GaussianRenderer::render(camera, 32, 40, input, pipe, background, color);
        const auto inference = GaussianRenderer::render(camera, 32, 40, input, pipe, background, color, 1.0f, false, false, RasterizeMode::Inference);
        check(torch::allclose(training.image, inference.image, 1e-5, 1e-6), "Historical training/inference pixels differ");
      }
      const auto target = GaussianRenderer::render(camera, 32, 40, cfull, pipe, background, color,
          1.0f, false, false, RasterizeMode::Inference).image * 0.9f;
      GuardEvaluation evaluation;
      evaluation.objectives = {{"full"}, {"lod"}};
      GaussianRenderInput candidate_input;
      evaluation.prepare_candidate = [&](const AdamStepPreview& p, const std::vector<torch::Tensor>& values) {
        scratch.prepare(p, values, selector, field); candidate_input = raw.input(p, &values);
      };
      evaluation.evaluate = [&](std::size_t objective, const AdamStepPreview& p, const std::vector<torch::Tensor>* values) {
        const auto input = values ? candidate_input : raw.input(p);
        const auto snet = values ? scratch.selector() : selector;
        const auto tf = values ? scratch.field() : field;
        const auto render_input = objective == 0 ? input : historicalLoDInput(
            input, maturity, uniform, snet, tf, 0.1f, 1.0);
        return (GaussianRenderer::render(camera, 32, 40, render_input, pipe, background, color,
            1.0f, false, false, torch::GradMode::is_enabled()
                ? RasterizeMode::Training : RasterizeMode::Inference).image - target).square().mean();
      };
      std::vector<torch::Tensor> old;
      for (const auto& p : params) old.push_back(p.detach().clone());
      const auto result = GradientConstraintManager().step(optimizer, evaluation);
      std::cout << std::fixed << std::setprecision(4) << "[GradientCUDA] status=" << result.status
                << " reason=" << result.reason << " alpha=" << result.alpha << " ms=" << result.milliseconds << '\n';
      check(raw.xyz.grad().equal(original_grad), "CUDA historical guard polluted normal Gaussian gradients");
      if (!result.committed) {
        check(optimizer.state().empty(), "Cancelled CUDA guard committed Adam moments");
        for (std::size_t i = 0; i < params.size(); ++i) check(params[i].equal(old[i]), "Cancelled CUDA guard changed parameters");
      } else {
        check(result.after[0] <= result.before[0] + 1e-7 + result.before[0] * 1e-5 &&
              result.after[1] <= result.before[1] + 1e-7 + result.before[1] * 1e-5,
              "Committed CUDA joint update increased actual Full/LoD MSE");
        for (const auto& p : params) {
          const auto& state = static_cast<const torch::optim::AdamParamState&>(*optimizer.state().at(p.unsafeGetTensorImpl()));
          check(state.step() == 1, "CUDA joint update did not commit state exactly once");
        }
      }
      check(result.status == "ACCEPT" || result.status == "SHRINK" || result.status == "CANCEL" || result.status == "NOOP",
            "Unknown CUDA guard status");
    }
  }
}
}
int main(int argc, char** argv) {
  const bool cuda = argc > 1 && std::string(argv[1]) == "--cuda";
  if (cuda && !torch::cuda::is_available()) return 77;
  run(torch::Device(cuda ? torch::kCUDA : torch::kCPU), cuda);
}

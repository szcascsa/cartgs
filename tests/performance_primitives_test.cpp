#include "improvements/selector/gumbel_network.h"
#include "include/historical_population.h"
#include "include/resized_image_cache.h"

#include <stdexcept>

namespace {
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void selectorBroadcast() {
  torch::manual_seed(31);
  improvements::selector::GumbelNetwork network;
  const float ratio = 0.15f;
  auto x = torch::randn({7, 3}).requires_grad_(true);
  auto q = torch::randn({7, 4}).requires_grad_(true);
  auto s = torch::rand({7, 3}).requires_grad_(true);
  const auto ids = torch::tensor({0, 2, 3, 6}, torch::kLong);
  auto dense = network->forwardProbability(x, q, s, torch::full({7, 1}, ratio));
  auto compact = network->forwardProbability(x.index_select(0, ids),
      q.index_select(0, ids), s.index_select(0, ids), ratio);
  check(torch::allclose(dense.index_select(0, ids), compact, 1e-5, 1e-6),
        "Mature-only scalar ratio changed deterministic probabilities");
  dense.index_select(0, ids).sum().backward();
  auto dx = x.grad().clone(), dq = q.grad().clone(), ds = s.grad().clone();
  std::vector<torch::Tensor> gradients;
  for (const auto& p : network->parameters()) gradients.push_back(p.grad().clone());
  network->zero_grad();
  x.mutable_grad() = torch::Tensor(); q.mutable_grad() = torch::Tensor();
  s.mutable_grad() = torch::Tensor();
  compact.sum().backward();
  check(torch::allclose(dx, x.grad(), 1e-4, 1e-6) &&
        torch::allclose(dq, q.grad(), 1e-4, 1e-6) &&
        torch::allclose(ds, s.grad(), 1e-4, 1e-6), "Mature gather changed Gaussian gradients");
  size_t i = 0;
  for (const auto& p : network->parameters())
    check(torch::allclose(gradients[i++], p.grad(), 1e-4, 1e-6),
          "Broadcast ratio changed Selector parameter gradients");
  const auto protected_ids = torch::tensor({1, 4, 5}, torch::kLong);
  check(x.grad().index_select(0, protected_ids).equal(torch::zeros({3, 3})),
        "Protected input received Selector gradients");

  // Same seed/population: forwardHard equals the original first sample.
  torch::manual_seed(77);
  auto hard = std::get<1>(network->forward(x, q, s, torch::full({7, 1}, ratio)));
  torch::manual_seed(77);
  auto fast = network->forwardHard(x, q, s, ratio);
  check(hard.equal(fast), "Hard-only forward changed binary sample");
  network->zero_grad();
  hard.sum().backward();
  gradients.clear();
  for (const auto& p : network->parameters()) gradients.push_back(p.grad().clone());
  network->zero_grad();
  fast.sum().backward();
  i = 0;
  for (const auto& p : network->parameters())
    check(torch::allclose(gradients[i++], p.grad(), 1e-4, 1e-6),
          "Hard-only forward lost straight-through gradients");
}
void unionCases() {
  const auto a = torch::tensor({0, 2, 4, 6}, torch::kLong);
  const auto b = torch::tensor({2, 3, 4, 7}, torch::kLong);
  const auto expected = torch::tensor({0, 2, 3, 4, 6, 7}, torch::kLong);
  check(historicalPopulationUnion({a, b}, 8).equal(expected), "Dense mask union differs");
  check(historicalPopulationUnion({a, b}, 100).equal(expected), "Sparse sort union differs");
  check(historicalPopulationUnion({a, a, b}, 8).equal(expected), "Duplicate view union differs");
  auto empty = torch::empty({0}, torch::kLong);
  check(historicalPopulationUnion({empty}, 0).numel() == 0 &&
        historicalPopulationUnion({empty}, 8).numel() == 0, "Empty union differs");
}
void imageCache() {
  ResizedImageCache cache;
  auto source = torch::rand({3, 24, 32}).requires_grad_(true);
  auto snapshot = source.detach().clone();
  auto first = cache.get(source, 12, 16);
  check(first.unsafeGetTensorImpl() == cache.get(source, 12, 16).unsafeGetTensorImpl(),
        "Resized GT cache did not reuse Tensor");
  const auto expected = torch::nn::functional::interpolate(source.unsqueeze(0),
      torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{12, 16})
          .mode(torch::kBilinear).align_corners(false)).squeeze(0);
  check(first.equal(expected) && !first.requires_grad() && source.detach().equal(snapshot),
        "Resized GT cache changed interpolation or source/grad semantics");
  check(cache.get(source, 24, 32).unsafeGetTensorImpl() == source.unsafeGetTensorImpl(),
        "Native GT must not be copied");
  auto replacement = source.detach().clone();
  check(cache.get(replacement, 12, 16).unsafeGetTensorImpl() != first.unsafeGetTensorImpl(),
        "Replacement GT reused an unrelated cached image");
}
}
int main() { selectorBroadcast(); unionCases(); imageCache(); }

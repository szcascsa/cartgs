#include "include/online_gi.h"

#include <torch/torch.h>

#include <stdexcept>

namespace {

void expectClose(const torch::Tensor& actual,
                 const torch::Tensor& expected,
                 const char* message) {
  if (!torch::allclose(actual, expected)) throw std::runtime_error(message);
}

template <typename Function>
void expectInvalidArgument(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(message);
}

template <typename Function>
void expectRuntimeError(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error(message);
}

}  // namespace

int main() {
  OnlineGIConfig config;
  OnlineGIManager manager(torch::kCPU, config);
  const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
  const auto long_options = torch::TensorOptions().dtype(torch::kInt64);
  const auto bool_options = torch::TensorOptions().dtype(torch::kBool);

  manager.resetForGaussianCount(3);
  if (manager.size() != 3 || manager.topologyVersion() != 0)
    throw std::runtime_error("OnlineGI reset lifecycle state is incorrect");
  const auto gi_version_after_reset = manager.version();
  expectClose(manager.getCombinedGI(), torch::zeros({3}, float_options),
              "OnlineGI reset must initialize zero scores");

  FrameGIScore zero_frame = manager.normalizeFrame(
      torch::zeros({3}, float_options), torch::ones({3}, bool_options));
  if (zero_frame.num_observed != 0 || zero_frame.observed_indices.numel() != 0)
    throw std::runtime_error(
        "Zero-importance frames must be skipped by OnlineGI");

  FrameGIScore frame;
  frame.score = torch::tensor({1.0f, 2.0f, 3.0f}, float_options);
  frame.observed_mask = torch::ones({3}, bool_options);
  frame.observed_indices = torch::arange(3, long_options);
  frame.num_observed = 3;
  manager.updateFast(frame, 7);
  manager.updateSlow(frame);
  expectClose(manager.getCombinedGI(), frame.score,
              "OnlineGI update must preserve distinguishable state");

  manager.appendGaussians(2);
  if (manager.size() != 5 || manager.topologyVersion() != 1)
    throw std::runtime_error("OnlineGI append lifecycle state is incorrect");
  if (manager.version() != gi_version_after_reset + 1)
    throw std::runtime_error("OnlineGI append must invalidate GI teachers");
  expectClose(manager.getCombinedGI(),
              torch::tensor({1.0f, 2.0f, 3.0f, 0.0f, 0.0f}, float_options),
              "OnlineGI append must preserve old state and zero new state");
  manager.assertAligned(torch::zeros({5, 3}, float_options));

  FrameGIScore appended_frame;
  appended_frame.score =
      torch::tensor({0.0f, 0.0f, 0.0f, 4.0f, 5.0f}, float_options);
  appended_frame.observed_mask =
      torch::tensor({0, 0, 0, 1, 1}, bool_options);
  appended_frame.observed_indices = torch::tensor({3, 4}, long_options);
  appended_frame.num_observed = 2;
  manager.updateFast(appended_frame, 8);
  manager.updateSlow(appended_frame);
  expectClose(manager.getCombinedGI(),
              torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, float_options),
              "Fresh OnlineGI entries must learn from their first observation");

  const auto topology_after_append = manager.topologyVersion();
  manager.appendGaussians(0);
  if (manager.topologyVersion() != topology_after_append)
    throw std::runtime_error("Empty OnlineGI append must be a no-op");
  expectInvalidArgument(
      [&manager]() { manager.appendGaussians(-1); },
      "OnlineGI append must reject a negative count");

  expectRuntimeError(
      [&manager, &float_options]() {
        manager.assertAligned(torch::zeros({4, 3}, float_options));
      },
      "OnlineGI alignment must reject a mismatched Gaussian count");
  expectInvalidArgument(
      [&manager, &float_options]() {
        manager.pruneGaussians(torch::ones({5}, float_options));
      },
      "OnlineGI pruning must reject a non-boolean survivor mask");
  expectInvalidArgument(
      [&manager, &bool_options]() {
        manager.pruneGaussians(torch::ones({4}, bool_options));
      },
      "OnlineGI pruning must reject a mismatched survivor mask");

  auto survivor_mask = torch::tensor({1, 0, 1, 0, 1}, bool_options);
  manager.pruneGaussians(survivor_mask);
  if (manager.size() != 3 || manager.topologyVersion() != 2)
    throw std::runtime_error("OnlineGI prune lifecycle state is incorrect");
  if (manager.version() != gi_version_after_reset + 2)
    throw std::runtime_error("OnlineGI prune must invalidate GI teachers");
  expectClose(manager.getCombinedGI(),
              torch::tensor({1.0f, 3.0f, 5.0f}, float_options),
              "OnlineGI prune must preserve survivor ordering and state");

  return 0;
}

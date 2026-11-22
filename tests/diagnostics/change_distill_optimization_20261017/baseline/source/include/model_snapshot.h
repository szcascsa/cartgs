#pragma once

#include <ATen/Context.h>
#include <torch/torch.h>
#include <optional>
#include <mutex>

// Mapping is serial and callers hold mutex_render_. Snapshot construction and
// domain initialization must not advance the normal training RNG stream.
class ScopedTorchRNG {
 public:
  explicit ScopedTorchRNG(torch::Device device,
                          std::optional<uint64_t> seed = std::nullopt)
      : generator_(at::globalContext().defaultGenerator(device)) {
    std::lock_guard<std::mutex> lock(generator_.mutex());
    state_ = generator_.get_state();
    if (seed) generator_.set_current_seed(*seed);
  }
  ~ScopedTorchRNG() {
    std::lock_guard<std::mutex> lock(generator_.mutex());
    generator_.set_state(state_);
  }
  ScopedTorchRNG(const ScopedTorchRNG&) = delete;
  ScopedTorchRNG& operator=(const ScopedTorchRNG&) = delete;
 private:
  at::Generator generator_;
  torch::Tensor state_;
};

inline void copyFrozenModule(const torch::nn::Module& source,
                             torch::nn::Module& target) {
  torch::NoGradGuard no_grad;
  auto target_parameters = target.named_parameters();
  for (const auto& item : source.named_parameters()) {
    target_parameters[item.key()].copy_(item.value());
    target_parameters[item.key()].requires_grad_(false);
  }
  auto target_buffers = target.named_buffers();
  for (const auto& item : source.named_buffers())
    target_buffers[item.key()].copy_(item.value());
  target.eval();
}

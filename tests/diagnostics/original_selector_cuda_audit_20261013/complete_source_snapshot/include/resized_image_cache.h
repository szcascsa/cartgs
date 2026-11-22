#pragma once

#include <cstdint>
#include <map>
#include <tuple>
#include <torch/torch.h>
#include <torch/nn/functional/upsampling.h>

// Owned by the KF: immutable GT images may be reused across mapping/probes,
// while culled KFs release both the source and resized GPU tensors.
class ResizedImageCache {
 public:
  torch::Tensor get(const torch::Tensor& source, int height, int width) {
    if (source.size(1) == height && source.size(2) == width) return source;
    const Key key{reinterpret_cast<uintptr_t>(source.unsafeGetTensorImpl()), height, width};
    const auto found = images_.find(key);
    if (found != images_.end()) return found->second.resized;
    torch::NoGradGuard no_grad;
    auto resized = torch::nn::functional::interpolate(source.unsqueeze(0),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{height, width})
            .mode(torch::kBilinear).align_corners(false)).squeeze(0);
    // Retaining source ownership prevents TensorImpl address reuse.
    return images_.emplace(key, Entry{source, resized}).first->second.resized;
  }
 private:
  using Key = std::tuple<uintptr_t, int, int>;
  struct Entry { torch::Tensor source, resized; };
  std::map<Key, Entry> images_;
};

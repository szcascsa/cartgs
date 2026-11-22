#pragma once
#include <torch/torch.h>
#include <cstdlib>
#include <string>

// Only compiled into the separate correctness executable, never the native runner.
namespace cdverify {
inline torch::serialize::OutputArchive archive;
inline int64_t targets = 0, selector_steps = 0, tf_steps = 0;
inline void target(const torch::Tensor& value) {
  archive.write("target_" + std::to_string(targets++), value.detach().clone());
}
inline void step(const torch::Tensor& loss, const std::vector<torch::Tensor>& parameters,
                 bool tf) {
  auto& count = tf ? tf_steps : selector_steps;
  const auto prefix = tf ? "tf_" : "selector_";
  archive.write(prefix + std::string("loss_") + std::to_string(count), loss.detach().clone());
  if (count++ == 0)
    for (size_t i = 0; i < parameters.size(); ++i)
      if (parameters[i].grad().defined())
        archive.write(prefix + std::string("gradient_") + std::to_string(i),
                      parameters[i].grad().detach().clone());
}
inline void updated(const std::vector<torch::Tensor>& parameters, bool tf) {
  if ((tf ? tf_steps : selector_steps) != 1) return;
  for (size_t i = 0; i < parameters.size(); ++i)
    archive.write(std::string(tf ? "tf_" : "selector_") + "updated_" + std::to_string(i),
                  parameters[i].detach().clone());
}
}

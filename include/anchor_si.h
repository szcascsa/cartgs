#pragma once
#include <torch/torch.h>
#include <cstdint>
#include <filesystem>
#include <limits>
#include "improvements/selector/gumbel_network.h"

namespace anchor_si {
// Mean, eligible count and slot bitset per Anchor; only values have a VJP.
std::vector<torch::Tensor> reduce(const torch::Tensor& values,
    const torch::Tensor& owner, const torch::Tensor& members,
    const torch::Tensor& counts, const torch::Tensor& opacity,
    const torch::Tensor& inactive);
std::vector<torch::Tensor> reduceForward(const std::vector<torch::Tensor>& input);
torch::Tensor reduceBackward(const std::vector<torch::Tensor>& input,
    const torch::Tensor& count, const torch::Tensor& gradient);
torch::Tensor targets(const torch::Tensor& ema, const torch::Tensor& visits,
    const torch::Tensor& last, const torch::Tensor& active,
    const torch::Tensor& ratios, int64_t full_clock, int64_t population,
    int64_t minimum_visits, int64_t maximum_age, float temperature);

struct Config {
  bool enabled = false;
  float weight = .01f, decay = .9f, temperature = 1.f;
  int collect_every_full = 4, minimum_visits = 4, maximum_age_full = 128;
};

class State {
 public:
  explicit State(Config config = {}) : config_(config) {}
  void configure(Config config) { config_ = config; }
  bool captureNextFull() { return config_.enabled && (++full_clock_ % config_.collect_every_full == 0); }
  void invalidate() { visits_=torch::Tensor(); version_=std::numeric_limits<uint64_t>::max(); }
  void sync(const improvements::selector::GumbelNetwork& network, uint64_t version);
  void observe(const improvements::selector::GumbelNetwork& network,
      const torch::Tensor& contribution, const torch::Tensor& opacity,
      int64_t pixels, uint64_t version, const std::vector<float>& ratios);
  torch::Tensor loss(const improvements::selector::GumbelNetwork& network,
      const torch::Tensor& probability, const torch::Tensor& opacity,
      int ratio_index, uint64_t version);
  void accumulateHeadGradients(const torch::Tensor& loss,
      const improvements::selector::GumbelNetwork& network) const;
  void save(const std::filesystem::path& file) const;
  const torch::Tensor& ema() const { return ema_; }
  const torch::Tensor& visits() const { return visits_; }
  const torch::Tensor& labels() const { return labels_; }
 private:
  Config config_;
  int64_t full_clock_ = 0;
  uint64_t version_ = std::numeric_limits<uint64_t>::max();
  torch::Tensor uids_, member_uids_, ema_, visits_, last_, active_, activity_, labels_, members_;
};
}

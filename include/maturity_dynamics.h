#pragma once
#include <torch/torch.h>
#include <filesystem>
#include <vector>

inline torch::Tensor temporalMaturity(const torch::Tensor& maturity,
    const torch::Tensor& stability, float rise, float fall) {
  // Rates 1/1 reproduce the archived instantaneous-stability path exactly.
  if (rise == 1 && fall == 1) return stability;
  const auto rate = torch::where(stability >= maturity, rise, fall);
  return (maturity + rate * (stability - maturity)).clamp(0, 1);
}

// Diagnostic state never controls optimization/admission. All row state and
// retired cohorts stay on device; only reduced distributions are exported.
class MaturityDynamics {
 public:
  enum Column { Birth, Snapshot, FirstValid, Observations, First03, First05, First08, First095,
    Obs03, Obs05, Obs08, Obs095, HighThenLow, RecoveryStart, RecoveryObservation,
    RecoveryEvents, Drop10, Drop20, Inherited, Columns };
  torch::Tensor state, values;
  int64_t full_cycle_index = 0;
  bool enabled = true;
  void reset(int64_t count, const torch::TensorOptions& options, int64_t iteration);
  void append(int64_t count, int64_t iteration, const torch::Tensor& initial_maturity, bool inherited);
  void prune(const torch::Tensor& survivors, const torch::Tensor& maturity);
  void observe(int64_t iteration, const torch::Tensor& valid, const torch::Tensor& touched,
      const torch::Tensor& snapshot, const torch::Tensor& initialized,
      const torch::Tensor& drift, const torch::Tensor& drift_ema, const torch::Tensor& stability,
      const torch::Tensor& previous, const torch::Tensor& maturity);
  void save(const std::filesystem::path& folder, const torch::Tensor& maturity,
      float rise, float fall) const;
  double memoryMB() const;
 private:
  std::vector<torch::Tensor> retired_state_, retired_values_, curve_;
  torch::Tensor recovery_histogram_;
};

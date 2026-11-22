#pragma once
#include <torch/torch.h>
#include <map>
#include <ostream>
#include "protection_profiler.h"
#include "kernel_fusion.h"

// Accumulate detached scalars on device; read back only for reports/checkpoints.
struct ContinuousTFStatistics {
  struct Ratio {
    int64_t steps = 0, executed = 0;
    torch::Tensor sums;
    ProtectionProfiler selector_profiler;
  };
  std::map<float, Ratio> ratios;
  int64_t tf_start_mapping_iteration = -1;
  float entry_threshold = 0;
  int64_t first_ready_population_iteration = -1;
  int64_t ready_count_when_initialized = -1, tf_initialization_iteration = -1;
  int64_t initial_aabb_population = 0;
  void record(float ratio, int64_t selected, int64_t candidates,
      const torch::Tensor& inside, const torch::Tensor& candidate_maturity,
      int64_t gi_count, const torch::Tensor& gi_weight,
      int64_t selector_candidates = 0, int64_t population = 0,
      const torch::Tensor& before_epsilon = {},bool known_zero_tf = false) {
    torch::NoGradGuard guard;
    auto& row = ratios[ratio];
    ++row.steps;
    row.executed += candidates > 0;
#ifdef CARTGS_FUSED_MAPPING_CUDA
    if(known_zero_tf&&inside.is_cuda()&&candidates==0&&candidate_maturity.numel()==0&&!gi_weight.defined()) {
      mapping_cuda::addZeroTfStatistics(row.sums,inside.options(),selected,candidates,gi_count,selector_candidates,population);
      return;
    }
#endif
    const auto zero = torch::zeros({}, inside.options().dtype(torch::kDouble));
    const auto values = torch::stack({zero + selected, inside.sum().to(torch::kDouble),
        zero + candidates, candidate_maturity.sum().to(torch::kDouble), zero + gi_count,
        gi_weight.defined() ? gi_weight.sum().to(torch::kDouble) : zero,
        zero + selector_candidates, zero + population,
        before_epsilon.defined() ? before_epsilon.sum().to(torch::kDouble) : inside.sum().to(torch::kDouble)});
    if (!row.sums.defined()) row.sums = torch::zeros_like(values);
    row.sums.add_(values);
  }
  void report(std::ostream& out, int64_t first_cycle) {
    out << "[Continuous TF START] tf_start_mapping_iteration=" << tf_start_mapping_iteration
        << " first_valid_maturity_cycle_iteration=" << first_cycle << '\n';
    out << "[Elastic TF READINESS] first_valid_maturity_cycle=" << first_cycle
        << " first_ready_population_iteration=" << first_ready_population_iteration
        << " ready_count_when_initialized=" << ready_count_when_initialized
        << " tf_initialization_iteration=" << tf_initialization_iteration
        << " initial_aabb_population=" << initial_aabb_population << '\n';
    for (auto& entry : ratios) {
      auto& row = entry.second;
      const auto host = row.sums.cpu();
      const auto* sums = host.data_ptr<double>();
      out << "[Continuous TF GI] ratio=" << entry.first << " lod_steps=" << row.steps
          << " executed_tf_step_count=" << row.executed
          << " skipped_tf_step_count=" << row.steps - row.executed
          << " selected_count=" << sums[0] << " inside_aabb_count=" << sums[1]
          << " tf_candidate_before_epsilon=" << sums[8]
          << " tf_candidate_after_epsilon=" << sums[2]
          << " tf_skipped_by_epsilon=" << sums[8] - sums[2]
          << " tf_candidate_count=" << sums[2]
          << " mean_tf_maturity=" << sums[3] / std::max(1.0, sums[2])
          << " mean_elastic_maturity=" << sums[3] / std::max(1.0, sums[2])
          << " effective_tf_mass=" << sums[3]
          << " gi_supervised_count=" << sums[4]
          << " gi_effective_weight_sum=" << sums[5]
          << " mean_gi_maturity=" << sums[5] / std::max(1.0, sums[4]) << '\n';
      out << "[Elastic ENTRY] ratio=" << entry.first << " elastic_entry_threshold=" << entry_threshold
          << " elastic_eligible_count=" << sums[6] << " elastic_population_count=" << sums[7]
          << " elastic_eligible_fraction=" << sums[6] / std::max(1.0, sums[7])
          << " selector_candidate_count=" << sums[6]
          << " selector_skipped_low_maturity_count=" << sums[7] - sums[6] << '\n';
      row.selector_profiler.reportSelector(out, entry.first);
    }
  }
};

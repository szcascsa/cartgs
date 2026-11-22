#pragma once
#include "protection_config.h"
#include "protection_profiler.h"
#include "maturity_dynamics.h"
#include "gaussian_transform_ops.h"
#include "reliability_maturity_cuda.h"
#include <torch/torch.h>
#include <map>

// Online convergence state. Committed mode-1/2 Full boundaries observe drift.
class GaussianProtection {
 public:
  ProtectionConfig config;
  int64_t mapping_iteration = 0;  // Logging/supervision clock, not maturity evidence.
  int64_t first_valid_maturity_cycle_iteration = -1;
  torch::Tensor maturity, drift_ema;
  torch::Tensor prev_full_xyz, prev_full_log_scale, prev_full_rotation;
  torch::Tensor snapshot_valid, touched_in_cycle, cached_uniform;
  // First-drift validity is independent of EMA=0 and exp underflow to maturity=0.
  torch::Tensor drift_initialized;
  torch::Tensor optimization_support, view_support, view_bits;
  torch::Tensor contribution_ema, contribution_maturity;
  int64_t contribution_full_calls=0;
  bool captureContribution() {
    return config.rendering_contribution && contribution_full_calls++%config.contribution_every_full==0;
  }
  torch::Tensor selectorReliability() const {
    return (config.rendering_contribution?contribution_maturity:maturity).detach();
  }
  std::map<int64_t,int64_t> reliability_views;
  MaturityDynamics dynamics;
  std::shared_ptr<ProtectionProfiler> profiler = std::make_shared<ProtectionProfiler>();
  ProtectionProfiler* timing() const { return config.performance_logging ? profiler.get() : nullptr; }
  // GPU row indices are reused between maturity/topology changes. Tensor
  // identity/version also catches in-place changes made by checkpoint/tests.
  const torch::Tensor& indicesAbove(float threshold) const {
    torch::NoGradGuard guard;
    if (!entry_cache_source_.defined() || !entry_cache_source_.is_same(maturity) ||
        entry_cache_version_ != maturity._version()) {
      entry_indices_.clear(); entry_cache_source_ = maturity;
      entry_cache_version_ = maturity._version();
    }
    auto found = entry_indices_.find(threshold);
    if (found == entry_indices_.end())
      found = entry_indices_.emplace(threshold,
          torch::nonzero(maturity >= threshold).reshape({-1})).first;
    return found->second;
  }

  void reset(int64_t count, const torch::TensorOptions& options) {
    torch::NoGradGuard guard;
    config.validate();
    first_valid_maturity_cycle_iteration = config.fixed_maturity_one && count ? mapping_iteration : -1;
    entry_indices_.clear(); entry_cache_source_ = torch::Tensor{};
    const auto floats = options.dtype(torch::kFloat32).requires_grad(false);
    const auto bools = floats.dtype(torch::kBool);
    maturity = config.fixed_maturity_one ? torch::ones({count}, floats) : torch::zeros({count}, floats);
    drift_ema = torch::zeros({count}, floats);
    prev_full_xyz = torch::zeros({count, 3}, floats);
    prev_full_log_scale = torch::zeros({count, 3}, floats);
    prev_full_rotation = torch::zeros({count, 4}, floats);
    snapshot_valid = torch::zeros({count}, bools);
    touched_in_cycle = torch::zeros({count}, bools);
    drift_initialized = torch::zeros({count}, bools);
    cached_uniform = torch::rand({count}, floats);
    reliability_views.clear();contribution_full_calls=0;
    contribution_ema=torch::zeros({config.rendering_contribution?count:0},floats);
    contribution_maturity=torch::zeros_like(contribution_ema);
    optimization_support=torch::zeros({config.reliability_v2?count:0},floats);
    view_support=torch::zeros_like(optimization_support);
    view_bits=torch::zeros({config.reliability_v2?count:0,1},floats.dtype(torch::kInt64));
    dynamics.enabled = config.maturity_diagnostics && !config.fixed_maturity_one;
    dynamics.reset(count, floats, mapping_iteration);
  }
  void assertAligned(int64_t count) const {
    for (const auto& t : {maturity, drift_ema, cached_uniform, snapshot_valid, touched_in_cycle, drift_initialized})
      TORCH_CHECK(t.dim() == 1 && t.numel() == count && t.device() == maturity.device() &&
          !t.requires_grad(), "Maturity state is not Gaussian-topology aligned");
    for (const auto& t : {maturity, drift_ema, cached_uniform, prev_full_xyz, prev_full_log_scale, prev_full_rotation})
      TORCH_CHECK(t.scalar_type() == torch::kFloat32, "Maturity geometry must be FP32");
    for (const auto& t : {snapshot_valid, touched_in_cycle, drift_initialized})
      TORCH_CHECK(t.scalar_type() == torch::kBool, "Maturity validity must be bool");
    for (const auto& t : {prev_full_xyz, prev_full_log_scale, prev_full_rotation})
      TORCH_CHECK(t.dim() == 2 && t.size(0) == count && t.device() == maturity.device() &&
          !t.requires_grad(), "Full snapshot is not Gaussian-topology aligned");
    TORCH_CHECK(prev_full_xyz.size(1) == 3 && prev_full_log_scale.size(1) == 3 &&
        prev_full_rotation.size(1) == 4, "Invalid Full snapshot shape");
    if (dynamics.enabled)
      TORCH_CHECK(dynamics.state.size(0) == count && dynamics.values.size(0) == count &&
          dynamics.state.device() == maturity.device(), "Maturity diagnostics topology mismatch");
    if(config.rendering_contribution)for(const auto& t:{contribution_ema,contribution_maturity})
      TORCH_CHECK(t.numel()==count&&t.device()==maturity.device()&&!t.requires_grad(),"Contribution identity mismatch");
    if(config.reliability_v2)TORCH_CHECK(optimization_support.numel()==count&&view_support.numel()==count&&view_bits.size(0)==count,
        "Reliability support identity mismatch");
  }
  void append(int64_t count, const torch::Tensor& parents = {}) {
    torch::NoGradGuard guard;
    const auto inherit = [&](const torch::Tensor& value) {
      return parents.defined() ? value.index_select(0, parents) : torch::zeros({count}, value.options());
    };
    if (parents.defined()) TORCH_CHECK(parents.numel() == count, "Maturity parent mapping size mismatch");
    maturity = torch::cat({maturity, config.fixed_maturity_one ? torch::ones({count}, maturity.options()) : inherit(maturity)});
    if (config.fixed_maturity_one && first_valid_maturity_cycle_iteration < 0 && maturity.numel())
      first_valid_maturity_cycle_iteration = mapping_iteration;
    drift_ema = torch::cat({drift_ema, inherit(drift_ema)});
    drift_initialized = torch::cat({drift_initialized, inherit(drift_initialized)});
    if(config.reliability_v2) {
      optimization_support=torch::cat({optimization_support,inherit(optimization_support)*config.reliability_inherit});
      view_support=torch::cat({view_support,inherit(view_support)*config.reliability_inherit});
      auto bits=parents.defined()?view_bits.index_select(0,parents):torch::zeros({count,view_bits.size(1)},view_bits.options());
      view_bits=torch::cat({view_bits,bits});
      // Child geometry must produce its own valid drift observation.
      maturity.slice(0,maturity.numel()-count).zero_();
      drift_initialized.slice(0,drift_initialized.numel()-count).zero_();
    }
    if(config.rendering_contribution) {
      // 新生、Clone和Split都重新观测真实贡献；不把父贡献复制给多个子点。
      contribution_ema=torch::cat({contribution_ema,torch::zeros({count},contribution_ema.options())});
      contribution_maturity=torch::cat({contribution_maturity,torch::zeros({count},contribution_maturity.options())});
    }
    prev_full_xyz = torch::cat({prev_full_xyz, torch::zeros({count, 3}, prev_full_xyz.options())});
    prev_full_log_scale = torch::cat({prev_full_log_scale, torch::zeros({count, 3}, prev_full_log_scale.options())});
    prev_full_rotation = torch::cat({prev_full_rotation, torch::zeros({count, 4}, prev_full_rotation.options())});
    snapshot_valid = torch::cat({snapshot_valid, torch::zeros({count}, snapshot_valid.options())});
    touched_in_cycle = torch::cat({touched_in_cycle, torch::zeros({count}, touched_in_cycle.options())});
    cached_uniform = torch::cat({cached_uniform, torch::rand({count}, cached_uniform.options())});
    dynamics.append(count, mapping_iteration, maturity.slice(0, maturity.numel()-count), parents.defined());
    assertAligned(maturity.numel());
  }
  void prune(const torch::Tensor& survivors) {
    torch::NoGradGuard guard;
    assertAligned(survivors.numel());
    dynamics.prune(survivors, maturity);
    for (auto* t : {&maturity, &drift_ema, &prev_full_xyz, &prev_full_log_scale, &prev_full_rotation,
                   &snapshot_valid, &touched_in_cycle, &drift_initialized, &cached_uniform})
      *t = t->index({survivors});
    if(config.reliability_v2)for(auto* t:{&optimization_support,&view_support,&view_bits})*t=t->index({survivors});
    if(config.rendering_contribution)for(auto* t:{&contribution_ema,&contribution_maturity})*t=t->index({survivors});
    assertAligned(maturity.numel());
  }
  torch::Tensor selectionMaturity(bool enabled) const {
    return enabled ? maturity.detach() : torch::ones_like(maturity);
  }
  torch::Tensor matureMask() const { return maturity >= config.compatibility_mature_threshold; }
  void markTouched(const torch::Tensor& indices) {
    if (config.fixed_maturity_one) return;
    torch::NoGradGuard guard;
    if (indices.defined() && indices.numel()) touched_in_cycle.index_fill_(0, indices, true);
  }
  void observeFull(const torch::Tensor& xyz, const torch::Tensor& log_scale,
                   const torch::Tensor& normalized_rotation) {
    torch::NoGradGuard guard;
    assertAligned(xyz.size(0));
    if (config.fixed_maturity_one) return;
    if (!xyz.size(0)) return;
    const auto previous = dynamics.enabled ? maturity.clone() : torch::Tensor{};
    torch::Tensor valid, stability, drift;
    {
      ProtectionProfiler::Scope timing_scope(timing(), ProtectionProfiler::MaturityUpdate, xyz.is_cuda());
      // exp(mean(log(s))) == cbrt(sx*sy*sz), avoiding intermediate overflow.
      const auto size = prev_full_log_scale.mean(1).exp().clamp_min(1e-8);
      const auto d_xyz = (xyz - prev_full_xyz).norm(2, 1) / size;
      const auto d_scale = (log_scale - prev_full_log_scale).square().mean(1).sqrt();
      const auto q0 = torch::nn::functional::normalize(prev_full_rotation);
      const auto dot = (q0 * normalized_rotation).sum(1).abs().clamp(0, 1);
      const auto d_rot = (1 - dot.square()).clamp_min(0).sqrt();
      drift = torch::maximum(torch::maximum(d_xyz, d_scale), d_rot);
      valid = touched_in_cycle & snapshot_valid & torch::isfinite(drift);
      // One startup observation, never a per-LoD synchronization for TF cutoff.
      if (first_valid_maturity_cycle_iteration < 0 && valid.any().item<bool>())
        first_valid_maturity_cycle_iteration = mapping_iteration;
      const auto beta = torch::where(drift > drift_ema, config.beta_fast, config.beta_slow);
      const auto next = torch::where(drift_initialized, drift_ema + beta * (drift - drift_ema), drift);
      drift_ema.copy_(torch::where(valid, next, drift_ema));
      stability = (-drift_ema / config.drift_temperature).exp().clamp(0, 1);
      maturity.copy_(torch::where(valid, temporalMaturity(maturity, stability,
          config.maturity_rise_rate, config.maturity_fall_rate), maturity));
      drift_initialized.logical_or_(valid);
    }
    {
      ProtectionProfiler::Scope timing_scope(timing(), ProtectionProfiler::MaturitySnapshot, xyz.is_cuda());
      prev_full_xyz.copy_(xyz.detach());
      prev_full_log_scale.copy_(log_scale.detach());
      prev_full_rotation.copy_(normalized_rotation.detach());
      snapshot_valid.copy_(torch::isfinite(xyz).all(1) & torch::isfinite(log_scale).all(1) &
                           torch::isfinite(normalized_rotation).all(1));
    }
    {
      ProtectionProfiler::Scope diagnostic_scope(timing(), ProtectionProfiler::MaturityDiagnostics, xyz.is_cuda());
      dynamics.observe(mapping_iteration, valid, touched_in_cycle, snapshot_valid,
          drift_initialized, drift, drift_ema, stability, previous, maturity);
    }
    touched_in_cycle.zero_();
  }
  void observeReliabilityFull(const torch::Tensor& xyz,const torch::Tensor& log_scale,
      const torch::Tensor& rotation,const torch::Tensor& radii,
      const std::vector<torch::Tensor>& gradients,int64_t keyframe_id,
      const torch::Tensor& contribution={},int64_t sample_pixels=0) {
    torch::NoGradGuard guard;
    TORCH_CHECK(config.reliability_v2&&!config.fixed_maturity_one,"Reliability mode mismatch");
    assertAligned(xyz.size(0));
    const auto ordinal=reliability_views.emplace(keyframe_id,reliability_views.size()).first->second;
    if(ordinal/64>=view_bits.size(1)) {
      auto expanded=torch::zeros({xyz.size(0),view_bits.size(1)*2},view_bits.options());
      expanded.slice(1,0,view_bits.size(1)).copy_(view_bits);view_bits=expanded;
    }
    reliabilityMaturityUpdate({maturity,drift_ema,prev_full_xyz,prev_full_log_scale,prev_full_rotation,
        snapshot_valid,drift_initialized,touched_in_cycle,optimization_support,view_support,view_bits,
        contribution_ema,contribution_maturity},
        xyz,log_scale,rotation,radii,gradients,ordinal,config,contribution,sample_pixels);
    entry_indices_.clear();entry_cache_source_=torch::Tensor{};
    ++dynamics.full_cycle_index;
  }
  void saveReliability(const std::string& path) const {
    if(!config.reliability_v2)return;
    torch::serialize::OutputArchive out;
    std::vector<torch::Tensor> tensors{maturity,drift_ema,prev_full_xyz,prev_full_log_scale,prev_full_rotation,
        snapshot_valid,drift_initialized,touched_in_cycle,optimization_support,view_support,view_bits};
    for(size_t j=0;j<tensors.size();++j)out.write("state"+std::to_string(j),tensors[j],true);
    std::vector<int64_t> ids(reliability_views.size());
    for(const auto& p:reliability_views)ids[p.second]=p.first;
    out.write("views",torch::tensor(ids,torch::kInt64),true);
    if(config.rendering_contribution) {
      out.write("contribution_ema",contribution_ema,true);
      out.write("contribution_maturity",contribution_maturity,true);
      out.write("contribution_full_calls",torch::tensor(contribution_full_calls,torch::kInt64),true);
    }
    out.write("mapping",torch::tensor(mapping_iteration,torch::kInt64),true);
    out.save_to(path);
  }
  void loadReliability(const std::string& path) {
    if(!config.reliability_v2)return;
    torch::NoGradGuard guard;torch::serialize::InputArchive in;in.load_from(path,maturity.device());
    std::vector<torch::Tensor*> tensors{&maturity,&drift_ema,&prev_full_xyz,&prev_full_log_scale,&prev_full_rotation,
        &snapshot_valid,&drift_initialized,&touched_in_cycle,&optimization_support,&view_support,&view_bits};
    for(size_t j=0;j<tensors.size();++j)in.read("state"+std::to_string(j),*tensors[j],true);
    if(config.rendering_contribution) {
      contribution_ema=torch::zeros_like(maturity);
      contribution_maturity=maturity*(1-config.contribution_weight);
      in.try_read("contribution_ema",contribution_ema,true);
      in.try_read("contribution_maturity",contribution_maturity,true);
      torch::Tensor calls;
      if(in.try_read("contribution_full_calls",calls,true))contribution_full_calls=calls.item<int64_t>();
    }
    torch::Tensor ids,step;in.read("views",ids,true);in.read("mapping",step,true);
    ids=ids.cpu().contiguous();reliability_views.clear();auto data=ids.accessor<int64_t,1>();
    for(int64_t j=0;j<ids.numel();++j)reliability_views[data[j]]=j;
    mapping_iteration=step.item<int64_t>();assertAligned(maturity.numel());
  }
  void transportSnapshot(const GaussianTransformResult& correction) {
    if (config.fixed_maturity_one) return;
    torch::NoGradGuard guard;
    if (!correction.changed_attributes) return;
    assertAligned(correction.population_count);
    GaussianAttrState state{prev_full_xyz, prev_full_log_scale, prev_full_rotation};
    if (!correction.affects_all) state = state.select(correction.indices);
    const auto transformed = applyGaussianTransform(state, correction);
    if (correction.affects_all) {
      prev_full_xyz.copy_(transformed.xyz);
      prev_full_log_scale.copy_(transformed.log_scaling);
      prev_full_rotation.copy_(transformed.rotation);
    } else {
      prev_full_xyz.index_copy_(0, correction.indices, transformed.xyz);
      prev_full_log_scale.index_copy_(0, correction.indices, transformed.log_scaling);
      prev_full_rotation.index_copy_(0, correction.indices, transformed.rotation);
    }
  }
  double metadataMB() const {
    int64_t bytes = 0;
    for (const auto& t : {maturity, drift_ema, prev_full_xyz, prev_full_log_scale, prev_full_rotation,
                         snapshot_valid, touched_in_cycle, drift_initialized, cached_uniform})
      bytes += t.numel() * t.element_size();
    for(const auto& t:{optimization_support,view_support,view_bits,contribution_ema,contribution_maturity})if(t.defined())bytes+=t.numel()*t.element_size();
    return bytes / 1048576.0 + dynamics.memoryMB();
  }
  std::map<std::string, double> statistics(bool enabled = true) const {
    torch::NoGradGuard guard;
    const auto m = selectionMaturity(enabled);
    std::map<std::string, double> result{{"total_gaussians", double(m.numel())}};
    const auto distribution = [&](const torch::Tensor& t, const std::string& prefix,
                                  const std::vector<double>& percentiles) {
      result[prefix + "_mean"] = t.numel() ? t.mean().item<double>() : 0;
      for (const auto q : percentiles)
        result[prefix + "_p" + std::to_string(int(q * 100))] = t.numel() ? torch::quantile(t, q).item<double>() : 0;
      result[prefix + "_median"] = result[prefix + "_p50"];
    };
    distribution(m, "maturity", {.1, .5, .9});
    distribution(drift_ema, "drift", {.5, .9, .95});
    result["snapshot_valid_count"] = snapshot_valid.sum().item<double>();
    result["touched_this_cycle_count"] = touched_in_cycle.sum().item<double>();
    result["compatibility_mature_count"] = (m >= config.compatibility_mature_threshold).sum().item<double>();
    result["maturity_below_0_3_fraction"] = maturity.numel() ? (maturity < .3f).to(torch::kFloat32).mean().item<double>() : 0;
    result["maturity_above_equal_0_3_fraction"] = maturity.numel() ? (maturity >= .3f).to(torch::kFloat32).mean().item<double>() : 0;
    for (int i = 0; i < 5; ++i)
      result["maturity_bucket_" + std::to_string(i)] =
          ((m >= i * .2f) & (i == 4 ? m <= 1 : m < (i + 1) * .2f)).sum().item<double>();
    return result;
  }
  void report(std::ostream& out, bool enabled = true) const {
    out << "[Protection MATURITY] schema=continuous_cycle_geometric mapping_iter=" << mapping_iteration
        << " compatibility_threshold=" << config.compatibility_mature_threshold;
    for (const auto& entry : statistics(enabled)) out << ' ' << entry.first << '=' << entry.second;
    out << '\n';
    if (config.performance_logging) profiler->report(out, metadataMB());
  }
 private:
  mutable torch::Tensor entry_cache_source_;
  mutable int64_t entry_cache_version_ = -1;
  mutable std::map<float, torch::Tensor> entry_indices_;
};

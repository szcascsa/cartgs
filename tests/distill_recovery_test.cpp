#include "include/drift_stability.h"
#include "include/selector_handoff.h"
#include "include/gaussian_renderer.h"
#include "include/graphics_utils.h"
#include "include/historical_population.h"
#include "include/loss_utils.h"

#include <json/json.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

using Selector = improvements::selector::GumbelNetwork;
namespace fs = std::filesystem;

// This is the EXACT DSD loss, target cache, ratio ordering and temporary Adam.
// Only trigger/acceptance gates are bypassed in the optimizer-only experiment.
struct DistillRecoveryTestAccess {
  static std::vector<int64_t> optimize(DriftStabilityManager& manager, bool selector,
      const std::vector<torch::Tensor>& populations, const GaussianAttrState& state,
      const Selector& student, const TransformField& field,
      const std::vector<float>& ratios, double temperature, double diagonal) {
    return manager.distillModule(selector, populations, state, student, field,
                                 ratios, temperature, diagonal, 1, nullptr);
  }
};

namespace {
void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
Json::Value readJson(const fs::path& path) {
  std::ifstream stream(path);
  check(stream.good(), "Cannot open " + path.string());
  Json::Value value;
  stream >> value;
  return value;
}
template <typename T>
void setting(const cv::FileStorage& file, const std::string& key, T& value) {
  if (!file[key].empty()) file[key] >> value;
}
DriftStabilityConfig readConfig(const cv::FileStorage& file) {
  DriftStabilityConfig config;
  // This standalone benchmark deliberately exercises opt-in recovery.
  config.enabled = true;
  // OpenCV does not deserialize int64_t; convert these fields explicitly.
#define READ_INT(name) { int value = static_cast<int>(config.name); \
  setting(file, "DriftStability." #name, value); config.name = value; }
  READ_INT(probe_interval); READ_INT(min_bad_views); READ_INT(max_gaussians);
  READ_INT(batch_size); READ_INT(consecutive_trigger_count); READ_INT(reference_horizon);
  READ_INT(reference_min_age); READ_INT(reference_max_age);
  READ_INT(non_regression_refresh_count); READ_INT(epochs);
#undef READ_INT
#define READ_FLOAT(name) setting(file, "DriftStability." #name, config.name)
  READ_FLOAT(reference_ema_decay); READ_FLOAT(function_threshold);
  READ_FLOAT(psnr_regression_threshold); READ_FLOAT(budget_relative_tolerance);
  READ_FLOAT(position_weight); READ_FLOAT(scale_weight); READ_FLOAT(rotation_weight);
  READ_FLOAT(selector_learning_rate); READ_FLOAT(tf_grid_learning_rate);
  READ_FLOAT(tf_mlp_learning_rate);
#undef READ_FLOAT
  int seed = 0;
  setting(file, "DriftStability.seed", seed);
  config.seed = seed;
  return config;
}
template <typename Model>
Model trainableClone(const Model& source) {
  Model result(source->cloneFrozen());
  for (auto parameter : result->parameters()) parameter.requires_grad_(true);
  result->train();
  return result;
}
template <typename Model>
void restore(const Model& source, const Model& destination) {
  torch::NoGradGuard guard;
  const auto original = source->named_parameters();
  for (auto& parameter : destination->named_parameters())
    parameter.value().copy_(original[parameter.key()]);
}
struct View {
  std::shared_ptr<GaussianKeyframe> camera;
  torch::Tensor gt;
  HistoricalViewObservation observation;
};
struct Metric {
  double train_psnr = 0, holdout_psnr = 0, function_error = 0;
  double xyz_error = 0, scale_error = 0, rotation_error = 0;
  double expected = 0, target = 0;
  int64_t actual = 0;
  std::vector<double> view_psnr;
};

class Benchmark {
 public:
  GaussianAttrState state;
  GaussianRenderInput full;
  torch::Tensor mature, mature_indices, training_indices, uniform, background;
  AABB3D aabb;
  GaussianPipelineParams pipe;
  std::vector<View> views;
  std::vector<float> ratios;
  torch::Tensor maturity;
  DriftStabilityConfig config;
  double temperature = 1, render_scale = 1;

  HistoricalRenderQuality quality(std::size_t view, float ratio,
      const Selector& selector, const TransformField& field, HistoricalProbeCache& cache) {
    torch::NoGradGuard guard;
    return cache.quality(view, ratio, selector, field, [&] {
      const auto selector_probability = selector.ptr()->forwardProbability(
          state.xyz, state.rotation, state.log_scaling.exp(), ratio, temperature);
      const auto render_probability = mixSelectorProbability(selector_probability, maturity, ratio);
      const auto keep = uniform < render_probability;
      auto input = full;
      const auto mask = keep.to(torch::kFloat32);
      input.opacity = full.opacity * mask.unsqueeze(1);
      const auto ids = torch::nonzero(mature & field.ptr()->insideMask(state.xyz)).reshape({-1});
      const auto elastic = cache.transformed(field, ratio, ids);
      input.xyz = state.xyz.index_copy(0, ids, elastic.xyz);
      input.scaling = full.scaling.index_copy(0, ids, elastic.log_scaling.exp());
      input.rotation = state.rotation.index_copy(0, ids, elastic.rotation);
      torch::Tensor color;
      auto& v = views.at(view);
      auto rendered = GaussianRenderer::render(v.camera, v.gt.size(1), v.gt.size(2),
          input, pipe, background, color, 1.0f, false, false, RasterizeMode::Inference);
      const double psnr = loss_utils::psnr(rendered.image, v.gt).item<double>();
      check(std::isfinite(psnr), "Non-finite rendered PSNR");
      const auto active = keep.sum().item<int64_t>();
      const auto expected = render_probability.sum().item<double>();
      return HistoricalRenderQuality(psnr, 0,
          active, double(ratio) * state.xyz.size(0), expected,
          double(ratio) * state.xyz.size(0), expected);
    });
  }

  std::vector<Metric> measure(bool selector_case, const Selector& selector,
      const TransformField& field, const Selector& reference_selector,
      const TransformField& reference_field) {
    torch::NoGradGuard guard;
    HistoricalProbeCache cache(state, temperature);
    std::vector<Metric> result;
    for (const auto ratio : ratios) {
      Metric metric;
      for (std::size_t v = 0; v < views.size(); ++v) {
        const auto q = quality(v, ratio, selector, field, cache);
        metric.view_psnr.push_back(q.psnr);
        if (views[v].observation.guard_only) metric.holdout_psnr += q.psnr / 2.0;
        else metric.train_psnr += q.psnr / 4.0;
        metric.actual = q.active_mature_count;
        metric.expected = q.mature_expected;
        metric.target = q.mature_target;
      }
      if (selector_case) {
        metric.function_error = (cache.probability(selector, ratio, training_indices) -
            cache.probability(reference_selector, ratio, training_indices))
            .abs().mean().item<double>();
      } else {
        const auto a = cache.transformed(field, ratio, training_indices);
        const auto b = cache.transformed(reference_field, ratio, training_indices);
        metric.xyz_error = (a.xyz - b.xyz).abs().mean().item<double>() / aabb.diagonal().item<double>();
        metric.scale_error = (a.log_scaling - b.log_scaling).abs().mean().item<double>();
        metric.rotation_error = (1 - (a.rotation * b.rotation).sum(1).abs().clamp(0, 1))
                                   .mean().item<double>();
        metric.function_error = config.position_weight * metric.xyz_error +
            config.scale_weight * metric.scale_error + config.rotation_weight * metric.rotation_error;
      }
      result.push_back(metric);
    }
    return result;
  }
};

Json::Value metricJson(const Metric& metric) {
  Json::Value value;
  value["train_psnr"] = metric.train_psnr;
  value["holdout_psnr"] = metric.holdout_psnr;
  value["function_error"] = metric.function_error;
  value["xyz_error"] = metric.xyz_error;
  value["scale_error"] = metric.scale_error;
  value["rotation_error"] = metric.rotation_error;
  value["mature_expected"] = metric.expected;
  value["mature_actual"] = Json::Int64(metric.actual);
  value["mature_target"] = metric.target;
  for (auto psnr : metric.view_psnr) value["view_psnr"].append(psnr);
  return value;
}

Json::Value runCase(Benchmark& b, const std::string& name, const Selector& reference_selector,
    const TransformField& reference_field, uint64_t seed) {
  const bool selector_case = name != "TF";
  const bool ranking_case = name == "SELECTOR_RANK";
  ScopedTorchRNG perturb_rng(b.state.xyz.device(), seed);
  auto selector = trainableClone(reference_selector);
  auto field = trainableClone(reference_field);
  const auto baseline = b.measure(selector_case, selector, field, reference_selector, reference_field);
  auto duplicate = b.measure(selector_case, reference_selector, reference_field,
                             reference_selector, reference_field);
  for (std::size_t r = 0; r < baseline.size(); ++r) {
    check(baseline[r].function_error < 1e-7, "Unperturbed clone is not function-identical");
    check(std::abs(baseline[r].train_psnr - duplicate[r].train_psnr) < 1e-5 &&
          baseline[r].actual == duplicate[r].actual, "Teacher/student render parity failed");
  }
  std::vector<Metric> damaged;
  double amplitude = 0;
  bool harmful = false;
  Json::Value result;
  torch::Tensor ranking_noise;
  if (ranking_case) {
    const auto weight = reference_selector->named_parameters()["soft_net.weight"];
    // Perturb spatial/attribute ranking, not the ratio embedding weights.
    ranking_noise = torch::randn_like(weight);
    ranking_noise.slice(1, 96, 128).zero_();
    ranking_noise.mul_(weight.std().item<double>());
  }
  const std::vector<double> amplitudes = selector_case ?
      (ranking_case ? std::vector<double>{0.25, 0.5, 1.0, 2.0, 4.0} :
                     std::vector<double>{0.1, 0.3, 0.6, 1.0, 2.0, 3.0}) :
      std::vector<double>{0.005, 0.01, 0.02, 0.05};
  for (auto candidate : amplitudes) {
    restore(reference_selector, selector);
    restore(reference_field, field);
    {
      torch::NoGradGuard guard;
      if (ranking_case) {
        selector->named_parameters()["soft_net.weight"].add_(ranking_noise, candidate);
        const auto attrs = b.state.select(b.mature_indices);
        std::vector<torch::Tensor> logits;
        double target = 0;
        for (std::size_t r = 0; r < b.ratios.size(); ++r) {
          logits.push_back(selector->forwardKeepLogit(attrs.xyz, attrs.rotation,
              attrs.log_scaling.exp(), b.ratios[r], 1.0));
          target += baseline[r].expected;
        }
        auto packed = torch::cat(logits);
        target /= packed.numel();
        double low = -40, high = 40;
        for (int step = 0; step < 24; ++step) {
          const double middle = (low + high) / 2;
          if ((packed + middle).sigmoid().mean().item<double>() < target) low = middle;
          else high = middle;
        }
        // Match reference mean expected cardinality across ratios. This is a
        // TEST FAULT calibration, not a new selector/production rendering rule.
        selector->named_parameters()["soft_net.bias"][1].add_((low + high) / 2);
      } else if (selector_case) {
        // Controlled logit drift, not a different Gaussian population or GT.
        selector->named_parameters()["soft_net.bias"][1].sub_(candidate);
      } else {
        // World-space XYZ drift; Selector and its exact admission stay fixed.
        field->named_parameters()["position_out.bias"].add_(candidate);
      }
    }
    damaged = b.measure(selector_case, selector, field, reference_selector, reference_field);
    double train_drop = 0, holdout_drop = 0;
    for (std::size_t r = 0; r < damaged.size(); ++r) {
      train_drop += baseline[r].train_psnr - damaged[r].train_psnr;
      holdout_drop += baseline[r].holdout_psnr - damaged[r].holdout_psnr;
    }
    amplitude = candidate;
    harmful = train_drop / damaged.size() > 0.1 && holdout_drop / damaged.size() > 0.1;
    Json::Value trial;
    trial["amplitude"] = candidate;
    trial["train_psnr_drop"] = train_drop / damaged.size();
    trial["holdout_psnr_drop"] = holdout_drop / damaged.size();
    result["perturbation_trials"].append(trial);
    if (harmful) break;
  }
  result["module"] = name;
  result["amplitude"] = amplitude;
  result["harmful_perturbation"] = harmful;
  if (!harmful) {
    for (std::size_t r = 0; r < b.ratios.size(); ++r) {
      auto& row = result["ratios"][static_cast<Json::ArrayIndex>(r)];
      row["ratio"] = b.ratios[r];
      row["baseline"] = metricJson(baseline[r]);
      row["damaged"] = metricJson(damaged[r]);
    }
    result["outcome"] = "INCONCLUSIVE_NO_RENDER_DEGRADATION";
    std::cout << "[RecoveryTest] module=" << name << " outcome=INCONCLUSIVE_NO_RENDER_DEGRADATION\n";
    return result;
  }
  auto damaged_selector = Selector(selector->cloneFrozen());
  auto damaged_field = TransformField(field->cloneFrozen());
  std::vector<HistoricalViewObservation> observations;
  for (const auto& view : b.views) observations.push_back(view.observation);

  // Strict production path: original thresholds, quality guards and rollback.
  DriftStabilityManager strict(b.config);
  strict.refreshReference(reference_selector, reference_field, 0, 0, 0, true, true, "recovery_test");
  DriftProbeReport report;
  const auto strict_start = std::chrono::steady_clock::now();
  for (int64_t hit = 1; hit <= b.config.consecutive_trigger_count; ++hit) {
    HistoricalProbeCache cache(b.state, b.temperature);
    report = strict.maybeProbeAndDistill(b.state, selector, field, b.ratios,
        b.temperature, b.aabb, observations, hit * b.config.probe_interval, 0,
        [&](std::size_t view, float ratio, const Selector& s, const TransformField& f) {
          const auto quality = b.quality(view, ratio, s, f, cache);
          Json::Value trace;
          trace["probe"] = Json::Int64(hit);
          trace["view"] = Json::UInt64(view);
          trace["ratio"] = ratio;
          trace["student_selector"] = s.ptr() == selector.ptr();
          trace["student_tf"] = f.ptr() == field.ptr();
          // Revision 0: baseline; 1: tentative update; 2: rollback (if any).
          trace["selector_revision"] = Json::UInt64(cache.selectorRevision());
          trace["tf_revision"] = Json::UInt64(cache.tfRevision());
          trace["psnr"] = quality.psnr;
          trace["mature_expected"] = quality.mature_expected;
          trace["mature_target"] = quality.mature_target;
          result["strict_quality_trace"].append(trace);
          return quality;
        }, &cache);
  }
  result["strict_seconds"] = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - strict_start).count();
  result["strict_outcome"] = selector_case ? report.selector_outcome : report.tf_outcome;
  int64_t strict_steps = 0;
  for (const auto& ratio : report.ratios)
    strict_steps += selector_case ? ratio.selector_steps : ratio.tf_steps;
  result["strict_steps"] = Json::Int64(strict_steps);
  const auto strict_post = b.measure(selector_case, selector, field, reference_selector, reference_field);

  // Restore EXACTLY the same degraded weights; no easier fault for this pass.
  restore(damaged_selector, selector);
  restore(damaged_field, field);
  auto direct_config = b.config;
  direct_config.seed = seed;
  DriftStabilityManager direct(direct_config);
  direct.refreshReference(reference_selector, reference_field, 0, 0, 0, true, true, "optimizer_test");
  std::vector<torch::Tensor> populations;
  for (std::size_t r = 0; r < b.ratios.size(); ++r) {
    std::vector<torch::Tensor> bad_sets;
    for (std::size_t v = 0; v < b.views.size(); ++v)
      if (!b.views[v].observation.guard_only &&
          baseline[r].view_psnr[v] - damaged[r].view_psnr[v] > b.config.psnr_regression_threshold)
        bad_sets.push_back(b.views[v].observation.visible_mature_indices);
    populations.push_back(bad_sets.empty() ? torch::Tensor() :
        historicalPopulationUnion(bad_sets, b.state.xyz.size(0)));
  }
  const auto direct_start = std::chrono::steady_clock::now();
  const auto steps = DistillRecoveryTestAccess::optimize(direct, selector_case, populations,
      b.state, selector, field, b.ratios, b.temperature, b.aabb.diagonal().item<double>());
  result["optimizer_seconds"] = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - direct_start).count();
  result["optimizer_steps"] = Json::Int64(std::accumulate(steps.begin(), steps.end(), int64_t{0}));
  const auto post = b.measure(selector_case, selector, field, reference_selector, reference_field);
  double gap_before = 0, gap_after = 0, error_before = 0, error_after = 0;
  double holdout_gain = 0, worst_new_drop = 0, worst_remaining_gap = 0;
  for (std::size_t r = 0; r < b.ratios.size(); ++r) {
    auto& row = result["ratios"][static_cast<Json::ArrayIndex>(r)];
    row["ratio"] = b.ratios[r];
    row["baseline"] = metricJson(baseline[r]);
    row["damaged"] = metricJson(damaged[r]);
    row["strict_post"] = metricJson(strict_post[r]);
    row["optimizer_post"] = metricJson(post[r]);
    row["optimizer_population"] = Json::Int64(populations[r].defined() ? populations[r].numel() : 0);
    row["union_function_over_threshold"] = damaged[r].function_error > b.config.function_threshold;
    row["strict_selector_drift"] = report.ratios[r].selector_drift;
    row["strict_tf_drift"] = report.ratios[r].tf_drift;
    row["strict_quality_evaluated"] = report.ratios[r].quality_evaluated;
    row["strict_selector_budget_qualified"] = report.ratios[r].selector_budget_qualified;
    row["teacher_budget_relative_error"] = std::abs(baseline[r].expected - baseline[r].target) /
                                            std::max(baseline[r].target, 1.0);
    row["student_budget_relative_error"] = std::abs(damaged[r].expected - damaged[r].target) /
                                            std::max(damaged[r].target, 1.0);
    row["strict_selector_triggered"] = report.ratios[r].selector_triggered;
    row["strict_tf_triggered"] = report.ratios[r].tf_triggered;
    gap_before += baseline[r].train_psnr - damaged[r].train_psnr;
    gap_after += baseline[r].train_psnr - post[r].train_psnr;
    holdout_gain += post[r].holdout_psnr - damaged[r].holdout_psnr;
    error_before += damaged[r].function_error;
    error_after += post[r].function_error;
    for (std::size_t v = 0; v < b.views.size(); ++v) {
      worst_new_drop = std::max(worst_new_drop, damaged[r].view_psnr[v] - post[r].view_psnr[v]);
      worst_remaining_gap = std::max(worst_remaining_gap, baseline[r].view_psnr[v] - post[r].view_psnr[v]);
    }
    std::cout << "[RecoveryTest] module=" << name << " ratio=" << b.ratios[r]
              << " train_psnr(base/bad/post)=" << baseline[r].train_psnr << "/"
              << damaged[r].train_psnr << "/" << post[r].train_psnr
              << " holdout_psnr=" << baseline[r].holdout_psnr << "/"
              << damaged[r].holdout_psnr << "/" << post[r].holdout_psnr
              << " drift=" << damaged[r].function_error << "->" << post[r].function_error
              << " mature_actual=" << baseline[r].actual << "/" << damaged[r].actual
              << "/" << post[r].actual << std::endl;
  }
  const double recovery = 1 - gap_after / gap_before;
  result["quality_recovery_fraction"] = recovery;
  result["function_error_reduction_fraction"] = 1 - error_after / error_before;
  result["holdout_psnr_gain"] = holdout_gain / b.ratios.size();
  result["worst_new_drop"] = worst_new_drop;
  result["worst_remaining_gap"] = worst_remaining_gap;
  const bool improved = error_after < error_before && gap_after < gap_before && holdout_gain > 0;
  const bool recovered = improved && worst_remaining_gap <= b.config.psnr_regression_threshold &&
                         worst_new_drop <= b.config.psnr_regression_threshold;
  result["outcome"] = recovered ? "RECOVERED" : improved ? "PARTIAL_RECOVERY" :
      error_after < error_before ? "FUNCTION_ONLY_NO_QUALITY_RECOVERY" :
      gap_after < gap_before && holdout_gain > 0 ? "QUALITY_ONLY_FUNCTION_REGRESSION" : "OPTIMIZER_FAILED";
  // Optimizer-only results are evidence, NOT a production acceptance decision.
  result["production_accepted"] = result["strict_outcome"] == "IMPROVED" ||
                                    result["strict_outcome"] == "RECOVERED";
  check(error_before > 0 && std::isfinite(error_after), "Invalid function recovery measurement");
  check(b.state.xyz.equal(b.full.xyz) && b.state.log_scaling.exp().equal(b.full.scaling),
        "Recovery changed the frozen Gaussian inputs");
  std::cout << "[RecoveryTest RESULT] module=" << name
            << " strict=" << result["strict_outcome"].asString() << " steps=" << strict_steps
            << " optimizer=" << result["outcome"].asString()
            << " quality_recovered=" << recovery * 100 << "%"
            << " function_reduced=" << (1 - error_after / error_before) * 100 << "%"
            << " seconds=" << result["optimizer_seconds"].asDouble() << std::endl;
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    check(argc >= 5 && argc <= 8,
        "Usage: distill_recovery_test CHECKPOINT DATASET SELECTOR_CONFIG OUTPUT_JSON [EPOCHS] [SEED] [RENDER_SCALE]");
    if (!torch::cuda::is_available()) return 77;
    std::cout << std::fixed << std::setprecision(4);
    const fs::path checkpoint(argv[1]), dataset(argv[2]), output(argv[4]);
    const uint64_t seed = argc > 6 ? std::stoull(argv[6]) : 2026;
    torch::manual_seed(seed);
    cv::FileStorage file(argv[3], cv::FileStorage::READ);
    check(file.isOpened(), "Cannot read Selector config");
    Benchmark b;
    b.config = readConfig(file);
    if (argc > 5) b.config.epochs = std::stoi(argv[5]);
    b.render_scale = argc > 7 ? std::stod(argv[7]) : 1.0;
    check(b.render_scale > 0 && b.render_scale <= 1, "Render scale must be in (0,1]");
    setting(file, "Selector.temperature", b.temperature);
    file["Selector.target_ratios"] >> b.ratios;
    check(!b.ratios.empty(), "No elastic ratios configured");
    const auto saved = readJson(checkpoint / "transform_state.json");
    const auto cameras = readJson(checkpoint / "cameras.json");
    check(cameras.size() >= 6, "Need at least six checkpoint views");
    const int64_t iteration = saved["mapping_iter"].asInt64();
    GaussianModel model(3);
    model.loadPly(checkpoint / "point_cloud" / ("iteration_" + std::to_string(iteration)) / "point_cloud.ply");
    model.loadSelectorMetadataPly(checkpoint / "point_cloud" / ("iteration_" + std::to_string(iteration)) / "selector_metadata.ply");
    torch::NoGradGuard no_grad;
    b.state = {model.xyz_.detach().clone(), model.scaling_.detach().clone(),
               model.getRotationActivation().detach().clone()};
    b.full = {b.state.xyz, model.getOpacityActivation().detach().clone(),
              b.state.log_scaling.exp(), b.state.rotation, model.features_dc_.detach().clone(),
              model.features_rest_.detach().clone(), model.active_sh_degree_, model.max_sh_degree_};
    b.background = torch::zeros({3}, b.state.xyz.options());
    b.mature = model.getSelectorMatureMask().clone();
    b.mature_indices = torch::nonzero(b.mature).reshape({-1});
    check(b.mature_indices.numel() > 0, "No saved mature population");
    b.maturity = model.protection_.maturity.clone();
    b.uniform = model.protection_.cached_uniform.clone();
    TransformFieldConfig tf_config;
    tf_config.hexplane.feature_dim = saved["feature_dim"].asInt();
    tf_config.hexplane.spatial_resolution = saved["spatial_resolution"].asInt();
    tf_config.hexplane.ratio_resolution = saved["ratio_resolution"].asInt();
    tf_config.hidden_dim = saved["hidden_dim"].asInt();
    tf_config.hexplane.multires.clear();
    for (const auto& value : saved["multires"]) tf_config.hexplane.multires.push_back(value.asInt());
    tf_config.outside_identity = saved["outside_identity"].asBool();
    tf_config.maturity_compute_epsilon = saved.get("maturity_compute_epsilon", 0).asFloat();
    std::vector<float> low, high;
    for (const auto& value : saved["aabb_min"]) low.push_back(value.asFloat());
    for (const auto& value : saved["aabb_max"]) high.push_back(value.asFloat());
    b.aabb = {torch::tensor(low, b.state.xyz.options()), torch::tensor(high, b.state.xyz.options())};
    Selector reference_selector;
    reference_selector->to(torch::kCUDA);
    TransformField reference_field(b.aabb.min_xyz, b.aabb.max_xyz, tf_config);
    reference_field->to(torch::kCUDA);
    torch::serialize::InputArchive selector_archive, field_archive;
    selector_archive.load_from((checkpoint / "selector.pt").string(), torch::kCUDA);
    field_archive.load_from((checkpoint / "transform_field.pt").string(), torch::kCUDA);
    reference_selector->load(selector_archive); reference_field->load(field_archive);
    reference_selector->eval(); reference_field->eval();
    for (auto parameter : reference_selector->parameters()) parameter.requires_grad_(false);
    for (auto parameter : reference_field->parameters()) parameter.requires_grad_(false);
    std::vector<torch::Tensor> training_sets;
    Json::Value report;
    for (std::size_t v = 0; v < 6; ++v) {
      // Interleave holdout cameras along the trajectory, rather than use the tail.
      const auto& saved_camera = cameras[static_cast<Json::ArrayIndex>((v + 1) * (cameras.size() - 1) / 7)];
      View view;
      view.camera = std::make_shared<GaussianKeyframe>(v);
      view.camera->image_width_ = saved_camera["width"].asInt();
      view.camera->image_height_ = saved_camera["height"].asInt();
      view.camera->FoVx_ = graphics_utils::focal2fov(saved_camera["fx"].asFloat(), view.camera->image_width_);
      view.camera->FoVy_ = graphics_utils::focal2fov(saved_camera["fy"].asFloat(), view.camera->image_height_);
      auto twc = torch::eye(4);
      for (int row = 0; row < 3; ++row) {
        twc.index_put_({row, 3}, saved_camera["position"][row].asFloat());
        for (int col = 0; col < 3; ++col)
          twc.index_put_({row, col}, saved_camera["rotation"][row][col].asFloat());
      }
      twc = twc.to(torch::kCUDA);
      view.camera->world_view_transform_ = twc.inverse().transpose(0, 1).contiguous();
      view.camera->camera_center_ = twc.slice(0, 0, 3).select(1, 3).contiguous();
      view.camera->full_proj_transform_ = view.camera->world_view_transform_.matmul(
          view.camera->getProjectionMatrix(0.01f, 100.0f, view.camera->FoVx_, view.camera->FoVy_,
                                           torch::kCUDA).transpose(0, 1)).contiguous();
      const auto image_path = dataset / saved_camera["img_name"].asString();
      auto image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      check(!image.empty(), "Missing ground truth " + image_path.string());
      check(image.cols == view.camera->image_width_ && image.rows == view.camera->image_height_,
            "Image size does not match the saved camera");
      // tum_rgbd/Frame store the original RGB image, not a remapped image.
      cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
      cv::resize(image, image, cv::Size(std::max(1, int(image.cols * b.render_scale)),
                                       std::max(1, int(image.rows * b.render_scale))), 0, 0, cv::INTER_LINEAR);
      image.convertTo(image, CV_32FC3, 1.0 / 255.0);
      view.gt = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kFloat32)
                    .clone().permute({2, 0, 1}).contiguous().to(torch::kCUDA);
      torch::Tensor color;
      const auto visible = GaussianRenderer::render(view.camera, view.camera->image_height_,
          view.camera->image_width_, b.full, b.pipe, b.background, color, 1.0f,
          false, false, RasterizeMode::Visibility).visibility_indices;
      view.observation = {v, 1.0, visible.index({b.mature.index_select(0, visible)}), v == 1 || v == 4};
      check(view.observation.visible_mature_indices.numel() > 0, "Empty test view visibility");
      if (!view.observation.guard_only) training_sets.push_back(view.observation.visible_mature_indices);
      report["views"][static_cast<Json::ArrayIndex>(v)]["checkpoint_id"] = saved_camera["id"];
      report["views"][static_cast<Json::ArrayIndex>(v)]["holdout"] = view.observation.guard_only;
      report["views"][static_cast<Json::ArrayIndex>(v)]["image"] = image_path.string();
      b.views.push_back(std::move(view));
    }
    b.training_indices = historicalPopulationUnion(training_sets, b.state.xyz.size(0));
    report["checkpoint"] = checkpoint.string(); report["seed"] = Json::UInt64(seed);
    report["render_scale"] = b.render_scale; report["epochs"] = Json::Int64(b.config.epochs);
    report["function_threshold"] = b.config.function_threshold;
    report["psnr_regression_threshold"] = b.config.psnr_regression_threshold;
    report["budget_relative_tolerance"] = b.config.budget_relative_tolerance;
    report["selector_lr"] = b.config.selector_learning_rate;
    report["tf_grid_lr"] = b.config.tf_grid_learning_rate;
    report["tf_mlp_lr"] = b.config.tf_mlp_learning_rate;
    report["total_gaussians"] = Json::Int64(b.state.xyz.size(0));
    report["mature_gaussians"] = Json::Int64(b.mature_indices.numel());
    report["training_visible_mature"] = Json::Int64(b.training_indices.numel());
    std::cout << "[RecoveryTest BEGIN] total=" << b.state.xyz.size(0)
              << " mature=" << b.mature_indices.numel() << " training_visible=" << b.training_indices.numel()
              << " views(train/holdout)=4/2 epochs=" << b.config.epochs << " seed=" << seed << std::endl;
    // Leave no-grad scope explicitly: only Student networks may train.
    {
      torch::AutoGradMode enable_grad(true);
      report["cases"].append(runCase(b, "SELECTOR_BIAS", reference_selector, reference_field, seed));
      report["cases"].append(runCase(b, "SELECTOR_RANK", reference_selector, reference_field, seed));
      report["cases"].append(runCase(b, "TF", reference_selector, reference_field, seed));
    }
    check(model.xyz_.equal(b.state.xyz) && model.scaling_.equal(b.state.log_scaling) &&
          model.getRotationActivation().equal(b.state.rotation) &&
          model.getOpacityActivation().equal(b.full.opacity) &&
          model.features_dc_.equal(b.full.features_dc) && model.features_rest_.equal(b.full.features_rest),
          "Test changed master Gaussian attributes");
    if (!output.parent_path().empty()) fs::create_directories(output.parent_path());
    std::ofstream stream(output);
    check(stream.good(), "Cannot create result file");
    stream << report << std::endl;
    check(stream.good(), "Cannot write result file");
    std::cout << "[RecoveryTest END] report=" << output << std::endl;
    return 0;  // Successful experiment != successful recovery; see per-case outcome.
  } catch (const std::exception& error) {
    std::cerr << "[RecoveryTest ERROR] " << error.what() << std::endl;
    return 1;
  }
}

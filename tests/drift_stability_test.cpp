#include "include/drift_stability.h"
#include "tf_backend_test_options.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
using Selector = improvements::selector::GumbelNetwork;
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
class CaptureLog {
 public:
  CaptureLog() : previous_(std::cout.rdbuf(stream_.rdbuf())) {}
  ~CaptureLog() { std::cout.rdbuf(previous_); }
  std::string str() const { return stream_.str(); }
 private:
  std::ostringstream stream_;
  std::streambuf* previous_;
};
DriftStabilityConfig smallConfig() {
  DriftStabilityConfig config;
  config.enabled = true;
  config.probe_interval = 1;
  config.min_bad_views = 1;
  config.batch_size = 8;
  config.max_gaussians = 8;
  config.reference_min_age = 100;
  config.reference_horizon = 200;
  config.reference_max_age = 1000;
  config.selector_learning_rate = 0.01f;
  config.tf_mlp_learning_rate = 0.001f;
  config.tf_grid_learning_rate = 0.001f;
  return config;
}
HistoricalRenderQuality quality(double psnr, double expected = 4.0, double target = 4.0) {
  return {psnr, 0, 4, target, expected, target, expected};
}
struct Fixture {
  Selector selector;
  TransformField tf = nullptr;
  GaussianAttrState state;
  AABB3D aabb;
  std::vector<float> ratios{0.1f, 0.15f};
  Fixture() {
    torch::manual_seed(17);
    const auto options = torch::TensorOptions().device(tf_test_device).dtype(torch::kFloat32);
    aabb = {torch::full({3}, -2.0f, options), torch::full({3}, 2.0f, options)};
    TransformFieldConfig config;
    config.backend = tf_test_backend;
    config.hidden_dim = 8;
    config.hexplane.spatial_resolution = 4;
    config.hexplane.ratio_resolution = 4;
    config.hexplane.feature_dim = 2;
    config.hexplane.multires = {1};
    tf = TransformField(aabb.min_xyz, aabb.max_xyz, config);
    tf->to(tf_test_device);selector->to(tf_test_device);
#ifdef CARTGS_TEST_SELECTOR_CUDA
    selector->setBackend("cuda");
#endif
    torch::NoGradGuard no_grad;
    for (auto& p : selector->parameters()) p.zero_();
    for (auto& p : tf->parameters()) p.zero_();
    state.xyz = torch::zeros({8, 3}, options).requires_grad_(true);
    state.log_scaling = torch::zeros({8, 3}, options).requires_grad_(true);
    state.rotation = torch::zeros({8, 4}, options);
    state.rotation.select(1, 0).fill_(1.0f);
    state.rotation.requires_grad_(true);
  }
  void bias(float value) {
    torch::NoGradGuard no_grad;
    selector->named_parameters()["soft_net.bias"][1].fill_(value);
  }
  void tfBias(float value) {
    torch::NoGradGuard no_grad;
    tf->named_parameters()["position_out.bias"].fill_(value);
  }
  double logit(const Selector& s) const {
    const auto b = s->named_parameters()["soft_net.bias"];
    return (b[1] - b[0]).item<double>();
  }
  double tfValue(const TransformField& field) const {
    return field->named_parameters()["position_out.bias"][0].item<double>();
  }
  double selectorQuality(const Selector& s) const {
    return 20.0 - 10.0 * std::abs(1.0 / (1.0 + std::exp(-logit(s))) - 0.5);
  }
  std::vector<HistoricalViewObservation> views(bool guard = false) const {
    HistoricalViewObservation view;
    view.keyframe_id = 7;
    view.outlier_score = 1.0;
    view.visible_mature_indices = torch::arange(8, state.xyz.options().dtype(torch::kLong));
    std::vector<HistoricalViewObservation> result{view};
    if (guard) {
      HistoricalViewObservation recent;
      recent.keyframe_id = 8;
      recent.outlier_score = 1.0;
      recent.guard_only = true;
      result.push_back(recent);
    }
    return result;
  }
  DriftProbeReport probe(DriftStabilityManager& manager, uint64_t iter,
      const DriftStabilityManager::QualityEvaluator& evaluator, bool guard = false,
      HistoricalProbeCache* cache = nullptr) {
    return manager.maybeProbeAndDistill(state, selector, tf, ratios, 1.0,
                                       aabb, views(guard), iter, 0, evaluator, cache);
  }
  void noMasterGradient() const {
    check(!state.xyz.grad().defined() && !state.log_scaling.grad().defined() &&
          !state.rotation.grad().defined(), "DSD leaked gradients to master Gaussians");
  }
};
void recoveryDisabledByDefault() {
  check(!DriftStabilityConfig{}.enabled, "Historical recovery must be opt-in");
  Fixture f;
  auto config = smallConfig();
  config.enabled = false;
  DriftStabilityManager manager(config);
  std::vector<torch::Tensor> before;
  auto parameters = f.selector->parameters();
  const auto tf_parameters = f.tf->parameters();
  parameters.insert(parameters.end(), tf_parameters.begin(), tf_parameters.end());
  for (const auto& parameter : parameters) before.push_back(parameter.detach().clone());
  int calls = 0;
  const auto evaluator = [&](std::size_t, float, const Selector&, const TransformField&) {
    ++calls;
    return quality(20.0);
  };
  CaptureLog log;
  manager.refreshReference(f.selector, f.tf, 1, 1, 1);
  check(!manager.shouldProbe(1) &&
        !manager.prepareProbe(f.selector, f.tf, 1, 1),
        "Disabled recovery scheduled a historical probe");
  const auto report = f.probe(manager, 1, evaluator);
  check(!report.ran && report.ratios.empty() && calls == 0,
        "Disabled recovery evaluated historical quality or distilled");
  check(!manager.reference().selector_teacher && !manager.reference().tf_teacher &&
        manager.reference().change_version == 0 && log.str().empty(),
        "Disabled recovery maintained a teacher or emitted recovery events");
  for (std::size_t i = 0; i < parameters.size(); ++i)
    check(torch::equal(parameters[i], before[i]), "Disabled recovery changed network parameters");
  f.noMasterGradient();
}
void lazyQualityAndIndependentInitialization() {
  Fixture f;
  DriftStabilityManager manager(smallConfig());
  manager.refreshReference(f.selector, nullptr, 0, 0);
  f.bias(0.4f);
  int calls = 0;
  const auto evaluator = [&](std::size_t, float, const Selector& s, const TransformField&) {
    ++calls;
    return quality(f.selectorQuality(s));
  };
  auto first = f.probe(manager, 1, evaluator);
  check(calls == 0 && !first.ratios[0].selector_triggered,
        "Early drift probe must not render QUALITY");
  // The TF initialized during probe 1 must not clear Selector's one prior hit.
  auto second = f.probe(manager, 2, evaluator);
  check(calls > 0 && second.ratios[0].selector_triggered &&
        second.ratios[0].selector_steps == 1 && second.ratios[0].tf_steps == 0,
        "TF initialization erased independent Selector trigger history");
  check(second.selector_outcome == "IMPROVED" && !second.selector_recovery_passed &&
        manager.reference().selector_reference_iter == 0,
        "Partial improvement replaced its unfinished target");
  f.noMasterGradient();
}
void budgetQualificationAndAttribution() {
  for (int mode = 0; mode < 3; ++mode) {
    Fixture f;
    auto config = smallConfig();
    config.consecutive_trigger_count = 1;
    DriftStabilityManager manager(config);
    manager.refreshReference(f.selector, f.tf, 0, 0);
    const auto old_s = manager.reference().selector_teacher.ptr();
    f.bias(0.4f);
    f.tfBias(0.5f);
    const auto evaluator = [&](std::size_t, float, const Selector& s, const TransformField& tf) {
      // mode 0: old Selector exceeds the budget, TF genuinely helps.
      // mode 1: only Selector affects quality despite both modules drifting.
      // mode 2: missing budget evidence must fail closed.
      const double psnr = f.selectorQuality(s) - (mode == 1 ? 0.0 : 10.0 * std::abs(f.tfValue(tf)));
      if (mode == 2) return HistoricalRenderQuality{psnr};
      return quality(psnr, mode == 0 && s.ptr() == old_s ? 40.0 : 4.0);
    };
    const auto report = f.probe(manager, 1, evaluator);
    if (mode == 0)
      check(report.ratios[0].selector_steps == 0 && report.ratios[0].tf_steps > 0,
            "Invalid Selector teacher blocked an independently useful TF teacher");
    else if (mode == 1)
      check(report.ratios[0].selector_steps > 0 && report.ratios[0].tf_steps == 0,
            "Selector quality advantage was misattributed to TF");
    else
      check(report.ratios[0].selector_steps == 0 && report.ratios[0].tf_steps == 0,
            "Unknown budget was silently treated as budget compliant");
    f.noMasterGradient();
  }
}
void recoveryAndInterleaving() {
  Fixture f;
  auto config = smallConfig();
  config.consecutive_trigger_count = 1;
  config.epochs = 80;
  DriftStabilityManager manager(config);
  manager.refreshReference(f.selector, f.tf, 0, 0);
  f.bias(0.4f);
  const auto evaluator = [&](std::size_t, float, const Selector& s, const TransformField&) {
    return quality(f.selectorQuality(s));
  };
  CaptureLog log;
  const auto report = f.probe(manager, 1, evaluator);
  check(report.selector_recovery_passed && report.selector_outcome == "RECOVERED" &&
        report.ratios[0].selector_steps == 80 && report.ratios[1].selector_steps == 80,
        "Bounded multi-ratio BCE distillation did not recover the synthetic function");
  check(manager.reference().selector_reference_iter == 1 &&
        std::abs(f.logit(manager.reference().selector_teacher) - 0.5 * f.logit(f.selector)) < 1e-5,
        "Recovered module did not publish validated frozen EMA");
  const auto text = log.str();
  check(text.find("ratio=0.1000 epoch=1/") < text.find("ratio=0.1500 epoch=1/") &&
        text.find("ratio=0.1500 epoch=1/") < text.find("ratio=0.1000 epoch=2/"),
        "Ratios were trained to completion sequentially instead of interleaved");
  check(text.find("loss_type=soft_BCE") != std::string::npos &&
        text.find("probability_l1=") != std::string::npos,
        "BCE optimization loss was confused with the L1 drift metric");
  f.noMasterGradient();
}
void rollbackAndAllRatioGuards() {
  for (int mode = 0; mode < 4; ++mode) {
    Fixture f;
    auto config = smallConfig();
    config.consecutive_trigger_count = 1;
    DriftStabilityManager manager(config);
    manager.refreshReference(f.selector, f.tf, 0, 0);
    const auto old_s = manager.reference().selector_teacher.ptr();
    f.bias(0.4f);
    std::vector<torch::Tensor> saved;
    for (const auto& p : f.selector->parameters()) saved.push_back(p.detach().clone());
    HistoricalProbeCache cache(f.state, 1.0);
    bool untriggered_baseline = false;
    const auto evaluator = [&](std::size_t view, float ratio, const Selector& s, const TransformField& tf) {
      return cache.quality(view, ratio, s, tf, [&] {
        const bool live = s.ptr() == f.selector.ptr();
        const double movement = std::abs(f.logit(s) - 0.4);
        if (mode == 0) return quality(s.ptr() == old_s ? 20.0 : 18.0);
        if (mode == 1 && view == 8) return quality(20.0 - 100.0 * movement);
        if (mode == 2 && ratio > 0.12f) {
          if (live && !untriggered_baseline) {
            check(movement < 1e-6, "Untriggered ratio baseline was captured after mutation");
            untriggered_baseline = true;
          }
          return quality(20.0 - 100.0 * movement);
        }
        if (mode == 3 && live && movement > 1e-5)
          return quality(f.selectorQuality(s), 40.0);
        return quality(f.selectorQuality(s));
      });
    };
    const auto report = f.probe(manager, 1, evaluator, mode == 1, &cache);
    check(report.selector_outcome == (mode == 0 ? "NO_PROGRESS" : "REJECTED"),
          "Failed quality/budget guard did not reject the candidate");
    const auto parameters = f.selector->parameters();
    for (std::size_t i = 0; i < parameters.size(); ++i)
      check(parameters[i].equal(saved[i]), "Rejected DSD did not restore parameters exactly");
    check(cache.selectorRevision() >= 2, "Rollback did not invalidate Student caches again");
    if (mode == 2)
      check(untriggered_baseline && report.ratios[1].selector_steps == 0,
            "Untriggered ratio guard became a training target");
    check(!report.reference_refreshed, "Rejected DSD was promoted to a teacher");
    f.noMasterGradient();
  }
}
void tfOnlyAndOutsideIdentity() {
  Fixture f;
  auto config = smallConfig();
  config.consecutive_trigger_count = 1;
  DriftStabilityManager manager(config);
  manager.refreshReference(f.selector, f.tf, 0, 0);
  f.tfBias(0.5f);
  const auto evaluator = [&](std::size_t, float, const Selector&, const TransformField& tf) {
    return quality(20.0 - 10.0 * std::abs(f.tfValue(tf)));
  };
  auto report = f.probe(manager, 1, evaluator);
  check(report.tf_outcome == "IMPROVED" && report.ratios[0].tf_steps > 0 &&
        report.ratios[0].selector_steps == 0, "TF-only event updated the wrong module");
  {
    torch::NoGradGuard no_grad;
    f.state.xyz.fill_(10.0f);
  }
  report = f.probe(manager, 2, evaluator);
  check(report.ratios[0].tf_drift == 0.0 && report.ratios[0].tf_steps == 0,
        "Outside identity falsely triggered TF teaching");
  f.noMasterGradient();
}
void combinedDamageAndTFRecheck() {
  for (bool joint_guard : {false, true}) {
    Fixture f;
    auto config = smallConfig();
    config.consecutive_trigger_count = 1;
    DriftStabilityManager manager(config);
    manager.refreshReference(f.selector, f.tf, 0, 0);
    f.bias(0.4f);
    f.tfBias(0.5f);
    const auto evaluator = [&](std::size_t view, float, const Selector& s, const TransformField& tf) {
      if (view == 8)
        return quality(20.0 - 1.5 * std::abs(f.logit(s) - 0.4) -
                       30.0 * std::abs(f.tfValue(tf) - 0.5));
      const double tf_weight = joint_guard ? 10.0 : 100.0 * std::max(0.0, f.logit(s) - 0.38);
      return quality(f.selectorQuality(s) - tf_weight * std::abs(f.tfValue(tf)));
    };
    const auto report = f.probe(manager, 1, evaluator, joint_guard);
    check(report.selector_outcome == "IMPROVED", "Selector unexpectedly failed its own guard");
    if (joint_guard)
      check(report.tf_outcome == "REJECTED" && std::abs(f.tfValue(f.tf) - 0.5) < 1e-6,
            "Two individually tolerated drops exceeded the joint event damage budget");
    else
      check(report.tf_outcome == "SKIP" && report.ratios[0].tf_steps == 0 &&
            !report.ratios[0].tf_triggered,
            "TF used stale quality evidence after a successful Selector update");
    f.noMasterGradient();
  }
}
void qualityBasedEMAAndCandidateRejection() {
  for (bool reject : {false, true}) {
    Fixture f;
    auto config = smallConfig();
    config.reference_min_age = 1;
    config.non_regression_refresh_count = 2;
    DriftStabilityManager manager(config);
    manager.refreshReference(f.selector, f.tf, 0, 0);
    f.bias(0.4f); // Drift is > .02, but current quality has NOT degraded.
    const auto evaluator = [&](std::size_t, float, const Selector& s, const TransformField&) {
      return quality(reject && std::abs(f.logit(s) - 0.2) < 1e-5 ? 10.0 : 20.0);
    };
    check(!f.probe(manager, 1, evaluator).reference_refreshed, "One confirmation refreshed EMA");
    f.probe(manager, 2, evaluator);
    check(manager.reference().selector_reference_iter == (reject ? 0u : 2u),
          "Beneficial drift blocked EMA, or a bad candidate was published");
    if (!reject) {
      check(std::abs(f.logit(manager.reference().selector_teacher) - 0.2) < 1e-5 &&
            manager.reference().selector_full_reset_iter == 0,
            "EMA value or independent hard-age clock is incorrect");
    } else {
      f.probe(manager, 3, [](std::size_t, float, const Selector&, const TransformField&) {
        return quality(20.0);
      });
      check(manager.reference().selector_reference_iter == 3,
            "Candidate rejection unnecessarily discarded prior confirmations");
    }
  }
}
void budgetRebaseAndHardExpiry() {
  Fixture f;
  auto config = smallConfig();
  config.reference_min_age = 1;
  config.reference_horizon = 2;
  config.reference_max_age = 3;
  config.non_regression_refresh_count = 2;
  DriftStabilityManager manager(config);
  manager.refreshReference(f.selector, f.tf, 0, 0);
  const auto bad_teacher = manager.reference().selector_teacher.ptr();
  f.bias(0.4f);
  const auto evaluator = [&](std::size_t, float, const Selector& s, const TransformField&) {
    return quality(s.ptr() == bad_teacher ? 25.0 : 20.0,
                   s.ptr() == bad_teacher ? 40.0 : 4.0);
  };
  f.probe(manager, 1, evaluator);
  f.probe(manager, 2, evaluator);
  check(manager.reference().selector_reference_iter == 2 &&
        manager.reference().selector_full_reset_iter == 2 &&
        std::abs(f.logit(manager.reference().selector_teacher) - 0.4) < 1e-6,
        "Over-budget teacher was averaged into its compliant replacement");
  CaptureLog log;
  const auto expired = f.probe(manager, 5, evaluator);
  check(manager.reference().selector_full_reset_iter == 5 &&
        !expired.selector_recovery_passed && expired.ratios[0].selector_steps == 0,
        "Hard expiry kept teaching from its retired reference");
  check(log.str().find("expired_pending_qualification") != std::string::npos,
        "Hard expiry was mislabeled as successful recovery");
  manager.maybeProbeAndDistill(f.state, f.selector, f.tf, f.ratios, 1.0,
      f.aabb, {}, 8, 0, evaluator);
  check(manager.reference().selector_full_reset_iter == 8,
        "Empty history allowed a teacher to outlive its hard age");
}
void structuralScopeAndMissingEvidence() {
  Fixture f;
  DriftStabilityManager manager(smallConfig());
  manager.refreshReference(f.selector, f.tf, 0, 0);
  const auto s = manager.reference().selector_teacher.ptr();
  manager.refreshReference(f.selector, f.tf, 2, 1, 3, false, true, "aabb_change");
  check(manager.reference().selector_teacher.ptr() == s, "AABB reset replaced Selector teacher");
  check(!manager.prepareProbe(f.selector, f.tf, 3, 1), "Structural cycle did not skip DSD");
  check(manager.prepareProbe(f.selector, f.tf, 4, 1), "Structural skip leaked into future probes");
  manager.refreshReference(f.selector, f.tf, 4, 2, 4, true, true, "geometry_change");
  check(manager.prepareProbe(f.selector, f.tf, 5, 2), "Stale skip marker persisted");
  auto config = smallConfig();
  config.max_gaussians = 0;
  bool rejected = false;
  try { DriftStabilityManager invalid(config); } catch (const c10::Error&) { rejected = true; }
  check(rejected, "Invalid distillation population limit was accepted");
  config = smallConfig();
  config.budget_relative_tolerance = std::numeric_limits<float>::quiet_NaN();
  rejected = false;
  try { DriftStabilityManager invalid(config); } catch (const c10::Error&) { rejected = true; }
  check(rejected, "Non-finite budget tolerance was accepted");
}
void saturatedSelectorGradient() {
  Fixture f;
  f.bias(-15.0f);
  const auto logits = f.selector->forwardKeepLogit(f.state.xyz.detach(), f.state.rotation.detach(),
      f.state.log_scaling.detach().exp(), 0.1f, 1.0);
  check(torch::allclose(logits.sigmoid(), f.selector->forwardProbability(
      f.state.xyz.detach(), f.state.rotation.detach(), f.state.log_scaling.detach().exp(), 0.1f)),
      "Binary keep logit does not match deterministic probability");
  torch::nn::functional::binary_cross_entropy_with_logits(logits, torch::full_like(logits, 0.1f)).backward();
  check(f.selector->named_parameters()["soft_net.bias"].grad()[1].abs().item<double>() > 0.09,
        "Soft BCE retained probability-L1 saturation");
}
void deferredProbeKeepsReferenceAge() {
  Fixture f;
  auto config = smallConfig();
  config.probe_interval = 1000;
  config.reference_max_age = 1000;
  DriftStabilityManager manager(config);
  manager.refreshReference(f.selector, f.tf, 0, 0);
  const auto evaluator = [&](std::size_t, float, const Selector&, const TransformField&) {
    return quality(20.0);
  };
  check(!manager.prepareProbe(f.selector, f.tf, 1001, 0), "Non-due DSD probe was accepted");
  const auto report = manager.maybeProbeAndDistill(f.state, f.selector, f.tf, f.ratios,
      1.0, f.aabb, {}, 1001, 0, evaluator, nullptr, true);
  check(report.reference_refreshed && manager.reference().selector_full_reset_iter == 1001,
        "Deferred DSD used enhanced-step age instead of real mapping age");
  manager.refreshReference(f.selector, f.tf, 1001, 1, 1002);
  check(!manager.prepareProbe(f.selector, f.tf, 1002, 1, true),
        "Deferred DSD bypassed structural-change protection");
  config.enabled = false;
  DriftStabilityManager disabled(config);
  check(!disabled.prepareProbe(f.selector, f.tf, 1001, 0, true),
        "Deferred work enabled a disabled DSD module");
}
}  // namespace

int main() {
  recoveryDisabledByDefault();
  lazyQualityAndIndependentInitialization();
  budgetQualificationAndAttribution();
  recoveryAndInterleaving();
  rollbackAndAllRatioGuards();
  tfOnlyAndOutsideIdentity();
  combinedDamageAndTFRecheck();
  qualityBasedEMAAndCandidateRejection();
  budgetRebaseAndHardExpiry();
  structuralScopeAndMissingEvidence();
  saturatedSelectorGradient();
  deferredProbeKeepsReferenceAge();
  return 0;
}

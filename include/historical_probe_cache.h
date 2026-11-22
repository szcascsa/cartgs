#pragma once

#include <cstdint>
#include <map>
#include <limits>
#include <memory>
#include <tuple>

#include "change_management.h"
#include "frozen_teacher_target_cache.h"

struct HistoricalRenderQuality {
  double psnr;
  int64_t protected_count;
  int64_t active_mature_count;
  double target_budget, expected_active_count;
  double mature_target, mature_expected;
  HistoricalRenderQuality(double value = 0.0, int64_t protected_total = 0,
                          int64_t active_total = 0, double budget = 0.0,
                          double expected = 0.0,
                          double mature_budget = std::numeric_limits<double>::quiet_NaN(),
                          double mature_mass = std::numeric_limits<double>::quiet_NaN())
      : psnr(value), protected_count(protected_total),
        active_mature_count(active_total), target_budget(budget),
        expected_active_count(expected),
        mature_target(mature_budget), mature_expected(mature_mass) {}
};

// One instance owns one immutable Gaussian snapshot and one probe resolution.
// No-grad outputs are filled by global row as needed: visible rows first, the
// remaining render rows only when QUALITY is requested. Frozen Teacher outputs
// may seed distillation targets; never use cached Student outputs for backward.
class HistoricalProbeCache {
 public:
  using Selector = improvements::selector::GumbelNetwork;
  HistoricalProbeCache(const GaussianAttrState& current, double temperature)
      : current_{current.xyz.detach(), current.log_scaling.detach(),
                 current.rotation.detach()}, temperature_(temperature) {}

  torch::Tensor probability(const Selector& network, float ratio,
                            const torch::Tensor& indices) {
    torch::NoGradGuard no_grad;
    if (indices.numel() == 0) return torch::empty({0}, current_.xyz.options());
    if (!network) return torch::ones({indices.numel()}, current_.xyz.options());
    const ModuleKey key{id(network), ratio};
    auto it = selectors_.find(key);
    if (it == selectors_.end())
      it = selectors_.emplace(key, SelectorEntry{network,
          std::make_unique<FrozenTeacherTargetCache>(current_.xyz.size(0),
                                                     current_.xyz.options())}).first;
    return it->second.targets->get(indices, [&](const torch::Tensor& missing) {
      ++selector_forwards;
      selector_rows += missing.numel();
      const auto state = current_.select(missing);
      return network.ptr()->forwardProbability(state.xyz, state.rotation,
          state.log_scaling.exp(), ratio, temperature_);
    });
  }

  GaussianAttrState transformed(const TransformField& field, float ratio,
                                const torch::Tensor& indices) {
    torch::NoGradGuard no_grad;
    if (!field || indices.numel() == 0) return current_.select(indices);
    const ModuleKey key{id(field), ratio};
    auto it = fields_.find(key);
    if (it == fields_.end())
      it = fields_.emplace(key, FieldEntry{field,
          std::make_unique<FrozenTeacherTargetCache>(current_.xyz.size(0),
                                                     current_.xyz.options())}).first;
    const auto packed = it->second.targets->get(indices, [&](const torch::Tensor& missing) {
      ++tf_forwards;
      tf_rows += missing.numel();
      const auto state = field.ptr()->applyLogResidual(current_.select(missing), ratio);
      return torch::cat({state.xyz, state.log_scaling, state.rotation}, 1);
    });
    return {packed.slice(1, 0, 3), packed.slice(1, 3, 6), packed.slice(1, 6, 10)};
  }

  template <typename Render>
  HistoricalRenderQuality quality(std::size_t view, float ratio,
      const Selector& selector, const TransformField& field, Render render) {
    const QualityKey key{view, ratio, id(selector), id(field)};
    const auto found = qualities_.find(key);
    if (found != qualities_.end()) {
      ++quality_hits;
      return found->second.value;
    }
    torch::NoGradGuard no_grad;
    const auto value = render();
    ++quality_renders;
    // Retain model ownership to prevent pointer reuse by temporary candidates.
    qualities_.emplace(key, QualityEntry{selector, field, value});
    return value;
  }

  // A live Student keeps its address through optimizer updates. Explicitly
  // invalidate that module and every quality pair containing it after DSD.
  void invalidateStudent(const Selector& selector, const TransformField& field,
                         bool selector_changed, bool tf_changed) {
    if (selector_changed || tf_changed) ++revision_;
    if (selector_changed) ++selector_revision_;
    if (tf_changed) ++tf_revision_;
    eraseModule(selectors_, id(selector), selector_changed);
    eraseModule(fields_, id(field), tf_changed);
    for (auto it = qualities_.begin(); it != qualities_.end();) {
      if ((selector_changed && std::get<2>(it->first) == id(selector)) ||
          (tf_changed && std::get<3>(it->first) == id(field)))
        it = qualities_.erase(it);
      else ++it;
    }
  }

  uint64_t revision() const { return revision_; }
  uint64_t selectorRevision() const { return selector_revision_; }
  uint64_t tfRevision() const { return tf_revision_; }

  int64_t selector_forwards = 0, tf_forwards = 0;
  int64_t selector_rows = 0, tf_rows = 0;
  int64_t quality_renders = 0, quality_hits = 0;

 private:
  template <typename Model> static uintptr_t id(const Model& model) {
    return model ? reinterpret_cast<uintptr_t>(model.ptr().get()) : 0;
  }
  template <typename Entries>
  static void eraseModule(Entries& entries, uintptr_t model, bool changed) {
    if (!changed) return;
    for (auto it = entries.begin(); it != entries.end();)
      if (it->first.first == model) it = entries.erase(it);
      else ++it;
  }
  using ModuleKey = std::pair<uintptr_t, float>;
  using QualityKey = std::tuple<std::size_t, float, uintptr_t, uintptr_t>;
  struct SelectorEntry { Selector model; std::unique_ptr<FrozenTeacherTargetCache> targets; };
  struct FieldEntry { TransformField model; std::unique_ptr<FrozenTeacherTargetCache> targets; };
  struct QualityEntry { Selector selector; TransformField field; HistoricalRenderQuality value; };
  GaussianAttrState current_;
  double temperature_;
  uint64_t revision_ = 0;
  uint64_t selector_revision_ = 0, tf_revision_ = 0;
  std::map<ModuleKey, SelectorEntry> selectors_;
  std::map<ModuleKey, FieldEntry> fields_;
  std::map<QualityKey, QualityEntry> qualities_;
};

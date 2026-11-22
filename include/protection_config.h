#pragma once
#include <cmath>
#include <stdexcept>

struct ProtectionConfig {
  // Anchor Selector: retain cached-uniform/checkpoint layout, disable maturity dynamics.
  bool fixed_maturity_one = false;
  bool reliability_v2 = false;
  // 可选：仅Full图像的alpha*T贡献修正Selector分数；不改变Compensation Gate。
  bool rendering_contribution = false;
  float contribution_decay = .9f;
  float contribution_tau = 1e-5f;
  float contribution_weight = .4f;
  int contribution_every_full = 8;
  int contribution_pixel_stride = 1;
  float reliability_tau_o = 20.f;
  float reliability_tau_v = 3.f;
  float reliability_xyz_floor = 1e-4f;
  float reliability_inherit = .5f;
  float reliability_loss_weight = 0.f;
  float reliability_delta = .05f;
  bool reliability_gate = false;
  bool reliability_gate_appearance = true;
  float reliability_gate_low = .4f;
  float reliability_gate_high = .8f;
  float drift_temperature = 0.10f;
  float beta_fast = 0.60f;
  float beta_slow = 0.15f;
  float maturity_rise_rate = 0.05f;
  float maturity_fall_rate = 0.50f;
  bool maturity_diagnostics = true;
  // ChangeDistill/historical recovery compatibility only; never a TF/GI/Selector gate.
  float compatibility_mature_threshold = 0.80f;
  bool performance_logging = true;
  void validate() const {
    if (!std::isfinite(contribution_decay) || contribution_decay<0 || contribution_decay>=1 ||
        !std::isfinite(contribution_tau) || contribution_tau<=0 ||
        !std::isfinite(contribution_weight) || contribution_weight<0 || contribution_weight>1 ||
        contribution_every_full<=0 || contribution_pixel_stride<=0 || contribution_pixel_stride>32 ||
        (rendering_contribution && !reliability_v2))
      throw std::invalid_argument("Invalid Rendering Contribution configuration");
    if (!std::isfinite(reliability_tau_o) || reliability_tau_o<=0 ||
        !std::isfinite(reliability_tau_v) || reliability_tau_v<=0 ||
        !std::isfinite(reliability_xyz_floor) || reliability_xyz_floor<=0 ||
        !std::isfinite(reliability_inherit) || reliability_inherit<0 || reliability_inherit>1 ||
        !std::isfinite(reliability_loss_weight) || reliability_loss_weight<0 ||
        !std::isfinite(reliability_delta) || reliability_delta<0 || reliability_delta>1 ||
        !std::isfinite(reliability_gate_low) || !std::isfinite(reliability_gate_high) ||
        reliability_gate_low<0 || reliability_gate_high>1 || reliability_gate_low>=reliability_gate_high ||
        (!reliability_v2 && (reliability_loss_weight>0 || reliability_gate)))
      throw std::invalid_argument("Invalid Reliability Maturity V2 configuration");
    if (!std::isfinite(drift_temperature) || drift_temperature <= 0 ||
        !std::isfinite(beta_fast) || !std::isfinite(beta_slow) ||
        beta_slow <= 0 || beta_fast > 1 || beta_fast <= beta_slow ||
        !std::isfinite(maturity_rise_rate) || !std::isfinite(maturity_fall_rate) ||
        maturity_rise_rate <= 0 || maturity_fall_rate > 1 || maturity_fall_rate < maturity_rise_rate ||
        !std::isfinite(compatibility_mature_threshold) ||
        compatibility_mature_threshold < 0 || compatibility_mature_threshold > 1)
      throw std::invalid_argument("Invalid continuous geometric maturity configuration");
  }
};

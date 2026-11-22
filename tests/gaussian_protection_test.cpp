#include "include/gaussian_protection.h"
#include <iostream>
#include <stdexcept>

void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void near(double a, double b, const char* message) { check(std::abs(a-b) < 1e-5, message); }
int main(int argc, char** argv) {
  const bool cuda = argc > 1 && std::string(argv[1]) == "--cuda";
  if (cuda && !torch::cuda::is_available()) return 77;
  const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(cuda ? torch::kCUDA : torch::kCPU);
  GaussianProtection p; p.config.performance_logging = false;
  p.config.maturity_rise_rate = p.config.maturity_fall_rate = 1;
  p.reset(2, opts);
  auto xyz = torch::zeros({2, 3}, opts);
  auto scale = torch::tensor({{0.f,0.f,0.f},{float(std::log(10)),float(std::log(10)),float(std::log(10))}}, opts);
  auto q = torch::tensor({{1.f,0.f,0.f,0.f},{1.f,0.f,0.f,0.f}}, opts);
  const auto rows = torch::tensor({0,1}, opts.dtype(torch::kInt64));
  p.markTouched(rows); p.observeFull(xyz, scale, q);
  check(p.snapshot_valid.all().item<bool>() && p.maturity.eq(0).all().item<bool>() &&
        !p.drift_initialized.any().item<bool>() && !p.touched_in_cycle.any().item<bool>(),
        "A: first snapshot created maturity evidence");
  // B: refreshed snapshots without a committed touch must preserve state.
  xyz += 1; p.observeFull(xyz, scale, q);
  check(p.maturity.eq(0).all().item<bool>() && p.drift_ema.eq(0).all().item<bool>(), "B: inactivity increased maturity");
  xyz.index_put_({0,0}, xyz[0][0] + .1);
  xyz.index_put_({1,0}, xyz[1][0] + 1);
  p.markTouched(rows); p.observeFull(xyz, scale, q);
  near(p.drift_ema[0].item<double>(), .1, "First drift blended against zero");
  near(p.drift_ema[0].item<double>(), p.drift_ema[1].item<double>(), "E: scale normalization failed");
  const auto first_m = p.maturity.clone();
  p.markTouched(rows); p.observeFull(xyz, scale, -q);
  near(p.drift_ema[0].item<double>(), .085, "F: quaternion sign changed drift");
  check((p.maturity > first_m).all().item<bool>(), "C: stability did not smoothly increase maturity");
  for (int i=0; i<50; ++i) { p.markTouched(rows); p.observeFull(xyz, scale, q); }
  check(p.maturity.min().item<float>() > .999, "C: stable touched rows did not approach one");
  xyz.index_put_({0,0}, xyz[0][0] + 1);
  const auto old_v = p.drift_ema.clone();
  p.markTouched(rows); p.observeFull(xyz, scale, q);
  near(p.drift_ema[0].item<double>(), .4*old_v[0].item<double>() + .6, "D: instability did not use fast EMA");
  check(p.maturity[0].item<float>() < .003, "D: instability did not rapidly lower maturity");
  const auto unstable = p.drift_ema.clone();
  p.markTouched(rows); p.observeFull(xyz, scale, q);
  near(p.drift_ema[0].item<double>(), .85*unstable[0].item<double>(), "D: recovery did not use slow EMA");
  const auto frozen_m = p.maturity.clone(), frozen_v = p.drift_ema.clone();
  for (int i=0; i<3; ++i) p.observeFull(xyz, scale, q);
  check(p.maturity.equal(frozen_m) && p.drift_ema.equal(frozen_v), "B: untouched evidence changed");
  // Rotation magnitude and log-scale are independently included in max drift.
  GaussianProtection components; components.config.performance_logging=false; components.reset(2,opts);
  auto zero = torch::zeros_like(xyz), ls = torch::zeros_like(scale);
  components.observeFull(zero,ls,q);
  ls.index_put_({0}, .2f);
  auto rotated = q.clone(); rotated.index_put_({1}, torch::tensor({0.f,1.f,0.f,0.f},opts));
  components.markTouched(rows); components.observeFull(zero,ls,rotated);
  near(components.drift_ema[0].item<double>(), .2, "Scale RMS drift failed");
  near(components.drift_ema[1].item<double>(), 1, "Rotation angular proxy failed");
  // K: clone, split repeat order, ordinary append and exact survivor mapping.
  p.append(1, rows.slice(0,0,1)); p.append(2, torch::tensor({1,1},opts.dtype(torch::kInt64))); p.append(1);
  p.assertAligned(6);
  near(p.maturity[2].item<double>(), p.maturity[0].item<double>(), "K: clone lost maturity");
  near(p.drift_ema[3].item<double>(), p.drift_ema[1].item<double>(), "K: split lost EMA");
  check(!p.snapshot_valid.slice(0,2).any().item<bool>() && !p.touched_in_cycle.any().item<bool>() &&
        p.maturity[5].item<float>() == 0, "K: new topology inherited snapshots/touched");
  const auto before_m=p.maturity.clone(), before_xyz=p.prev_full_xyz.clone(), before_u=p.cached_uniform.clone();
  const auto survivors=torch::tensor({false,true,true,false,true,true},opts.dtype(torch::kBool));
  p.prune(survivors); p.assertAligned(4);
  check(p.maturity.equal(before_m.index({survivors})) && p.prev_full_xyz.equal(before_xyz.index({survivors})) &&
        p.cached_uniform.equal(before_u.index({survivors})), "K: prune mapping changed");
  // L: transport a known rigid+scale correction, then a zero-motion observation.
  GaussianProtection transport; transport.config.performance_logging=false;
  transport.config.maturity_rise_rate=transport.config.maturity_fall_rate=1;transport.reset(2,opts);
  transport.observeFull(xyz,scale,q); transport.markTouched(rows); transport.observeFull(xyz,scale,q);
  GaussianTransformResult correction; correction.affects_all=true; correction.population_count=2;
  correction.transform_4x4=torch::eye(4,opts); correction.transform_4x4.index_put_({3,0},2.f);
  correction.rotation_wxyz=torch::tensor({1.f,0.f,0.f,0.f},opts); correction.scale=2.f;
  correction.changed_attributes=toBits(GaussianAttrFlag::XYZ)|toBits(GaussianAttrFlag::SCALE);
  const auto moved=applyGaussianTransform({xyz,scale,q},correction);
  transport.transportSnapshot(correction); transport.markTouched(rows);
  transport.observeFull(moved.xyz,moved.log_scaling,moved.rotation);
  check(transport.maturity.eq(1).all().item<bool>(), "L: coordinate correction created false drift");
  correction.affects_all=false; correction.indices=rows.slice(0,0,1);
  correction.scale=1; correction.changed_attributes=toBits(GaussianAttrFlag::ROTATION);
  correction.rotation_wxyz=torch::tensor({0.f,1.f,0.f,0.f},opts);
  const auto subset=applyGaussianTransform(moved.select(correction.indices),correction);
  auto final_q=moved.rotation.clone(); final_q.index_copy_(0,correction.indices,subset.rotation);
  transport.transportSnapshot(correction); transport.markTouched(rows);
  transport.observeFull(moved.xyz,moved.log_scaling,final_q);
  check(transport.maturity.eq(1).all().item<bool>(), "L: partial rigid rotation created false drift");
  const auto mask=p.matureMask();
  check(mask.scalar_type()==torch::kBool && mask.dim()==1 && mask.numel()==4 && mask.device()==p.maturity.device(),
        "M: invalid compatibility mask");
  check(p.selectionMaturity(false).eq(1).all().item<bool>(), "Disabled protection did not force selection maturity one");
  std::cout << "PASS A/B/C/D/E/F/K/L/M snapshot/freeze/stability/instability/topology/transport/compatibility\n";
}

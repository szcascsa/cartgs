/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 *
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM, and modified by Dapeng Feng in 2024, as part of
 * CaRtGS.
 */

#include "include/gaussian_model.h"
#include "include/anchor_stats_cuda.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <sstream>

#include "include/gaussian_rasterizer.h"
#include "include/quaternion_utils.h"

GaussianModel::GaussianModel(const int sh_degree)
    : active_sh_degree_(0),
      spatial_lr_scale_(0.0),
      lr_delay_steps_(0),
      lr_delay_mult_(1.0),
      max_steps_(1000000) {
  this->max_sh_degree_ = sh_degree;

  // Device
  if (torch::cuda::is_available())
    this->device_type_ = torch::kCUDA;
  else
    this->device_type_ = torch::kCPU;

  GAUSSIAN_MODEL_INIT_TENSORS(this->device_type_)
}

GaussianModel::GaussianModel(const GaussianModelParams& model_params)
    : active_sh_degree_(0),
      spatial_lr_scale_(0.0),
      lr_delay_steps_(0),
      lr_delay_mult_(1.0),
      max_steps_(1000000) {
  this->max_sh_degree_ = model_params.sh_degree_;

  // Device
  if (model_params.data_device_ == "cuda")
    this->device_type_ = torch::kCUDA;
  else
    this->device_type_ = torch::kCPU;

  GAUSSIAN_MODEL_INIT_TENSORS(this->device_type_)
}

torch::Tensor GaussianModel::getScalingActivation() {
  return torch::exp(this->scaling_);
}

torch::Tensor GaussianModel::getRotationActivation() {
  return torch::nn::functional::normalize(this->rotation_);
}

torch::Tensor GaussianModel::getXYZ() { return this->xyz_; }

torch::Tensor GaussianModel::getFeatures() {
  return torch::cat({this->features_dc_.clone(), this->features_rest_.clone()},
                    /*dim=*/1);
}

torch::Tensor GaussianModel::getOpacityActivation() {
  return torch::sigmoid(this->opacity_);
}

torch::Tensor GaussianModel::getScalingRotationActivation(int scaling_modifier) {
  // build_rotation
  auto r = this->rotation_;
  auto R = general_utils::build_rotation(r);

  // build_scaling_rotation(scaling_modifier * scaling(Activation), rotation(_))
  auto s = scaling_modifier * this->getScalingActivation();
  auto L = torch::zeros(
      {s.size(0), 3, 3},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  L.select(1, 0).select(1, 0).copy_(s.index({torch::indexing::Slice(), 0}));
  L.select(1, 1).select(1, 1).copy_(s.index({torch::indexing::Slice(), 1}));
  L.select(1, 2).select(1, 2).copy_(s.index({torch::indexing::Slice(), 2}));
  L = R.matmul(L);  // L = R @ L
  return L;
}

torch::Tensor GaussianModel::getCovarianceActivation(int scaling_modifier) {
  auto L = getScalingRotationActivation(scaling_modifier);
  // build_covariance_from_scaling_rotation
  auto actual_covariance = L.matmul(L.transpose(1, 2));
  return actual_covariance;
}

void GaussianModel::oneUpShDegree() {
  if (this->active_sh_degree_ < this->max_sh_degree_)
    this->active_sh_degree_ += 1;
}

void GaussianModel::setShDegree(const int sh) {
  this->active_sh_degree_ =
      (sh > this->max_sh_degree_ ? this->max_sh_degree_ : sh);
}

void GaussianModel::enableExplicitAnchors(improvements::selector::AnchorBank bank,
                                          torch::optim::Adam* feature_optimizer,
                                          std::filesystem::path pruning_diagnostics) {
  TORCH_CHECK(xyz_.numel()==0 && bank && bank->explicitBlocks(), "Explicit map must be enabled before initialization");
  explicit_anchors_=bank;anchor_feature_optimizer_=feature_optimizer;
  anchor_opacity_accum_=torch::zeros({bank->size()},bank->features().options().dtype(torch::kFloat64).requires_grad(false));
  anchor_visits_=torch::zeros({bank->size()},bank->features().options().dtype(torch::kLong).requires_grad(false));
  anchor_slot_opacity_accum_=bank->variableMembers()?torch::zeros({0},anchor_opacity_accum_.options()):torch::zeros({bank->size(),10},anchor_opacity_accum_.options());
  anchor_slot_below_prune_=torch::zeros_like(anchor_slot_opacity_accum_,anchor_visits_.options());
  anchor_slot_below_cutoff_=torch::zeros_like(anchor_slot_below_prune_);
  anchor_pruning_diagnostics_=std::move(pruning_diagnostics);
}

void GaussianModel::finishAnchorGeometry() {
  if(explicit_anchors_) explicit_anchors_->finishBlockGeometry();
}

void GaussianModel::migrateLocalAnchors() {
  if(fixedTenAnchors()) {
    torch::NoGradGuard guard;
    auto migration=explicit_anchors_->takeFixedMigration();
    if(!migration.remove.defined())return;
    resetAnchorWindows(migration.changed);
    auto affected=migration.changed.index_select(0,explicit_anchors_->rows());
    anchor_growth_gradient_accum_.masked_fill_(affected.unsqueeze(1),0);anchor_growth_denom_.masked_fill_(affected.unsqueeze(1),0);
    xyz_gradient_accum_.masked_fill_(affected.unsqueeze(1),0);denom_.masked_fill_(affected.unsqueeze(1),0);
    max_radii2D_.masked_fill_(affected,0);
    if(explicit_anchors_->slotDeactivation()) {
      explicit_anchors_->inactiveSlots().masked_fill_(affected,false);
      explicit_anchors_->recoveredSlots().masked_fill_(affected,false);
    }
    const auto dropped=migration.remove.sum().item<int64_t>();
    if(dropped)prunePoints(migration.remove);
    local_anchor_migrated_members_+=migration.moved;local_anchor_dropped_members_+=dropped;
    explicit_anchors_->assertBlocks(xyz_.size(0));
    std::cout<<"[Fixed Ten Anchor Migration] transferred="<<migration.moved<<" dropped_collision="<<dropped
             <<" anchors="<<explicit_anchors_->size()<<" gaussians="<<xyz_.size(0)<<'\n';
    return;
  }
  auto corrected=local_anchor_moved_rows_;local_anchor_moved_rows_=torch::Tensor();
  if(!variableAnchorMembers() || !corrected.defined() || !corrected.numel())return;
  torch::NoGradGuard guard;
  auto migration=explicit_anchors_->migrateLocalMembers(xyz_,corrected,{optimizer_.get(),anchor_feature_optimizer_});
  const auto added=explicit_anchors_->size()-anchor_visits_.numel();
  anchor_opacity_accum_=torch::cat({anchor_opacity_accum_,torch::zeros({added},anchor_opacity_accum_.options())});
  anchor_visits_=torch::cat({anchor_visits_,torch::zeros({added},anchor_visits_.options())});
  resetAnchorWindows(migration.changed);
  auto affected=migration.changed.index_select(0,explicit_anchors_->rows());
  anchor_growth_gradient_accum_.masked_fill_(affected.unsqueeze(1),0);
  anchor_growth_denom_.masked_fill_(affected.unsqueeze(1),0);
  xyz_gradient_accum_.masked_fill_(affected.unsqueeze(1),0);denom_.masked_fill_(affected.unsqueeze(1),0);
  max_radii2D_.masked_fill_(affected,0);
  const auto dropped=migration.remove.sum().item<int64_t>();
  // This is collision resolution, independent of opacity/size quality pruning.
  // Reuse the existing compaction path for all attributes, Adam and row metadata.
  prunePoints(migration.remove);
  local_anchor_migrated_members_+=migration.moved;local_anchor_dropped_members_+=dropped;
  std::cout<<"[Local Anchor Migration] transferred="<<migration.moved<<" dropped_full="<<dropped
           <<" created="<<migration.created<<" anchors="<<explicit_anchors_->size()
           <<" gaussians="<<xyz_.size(0)<<" capacity=10 feature_weight=accepted_members\n";
}

void GaussianModel::addAnchorPruningStats(const torch::Tensor& visible_indices) {
  MappingAuditRange trace("full/anchor_statistics");
  if(!explicit_anchors_) return;
  torch::NoGradGuard guard;
  explicit_anchors_->assertBlocks(xyz_.size(0));
  TORCH_CHECK(anchor_opacity_accum_.numel()==explicit_anchors_->size() &&
      anchor_visits_.numel()==explicit_anchors_->size(),"Anchor pruning statistics are not aligned");
  if(variableAnchorMembers() && explicit_anchors_->fusedFullStats() && xyz_.is_cuda() &&
      explicit_anchors_->validatedMembers().defined() && visible_indices.is_contiguous()) {
    mapping_cuda::anchorFullStats(explicit_anchors_->rows(),explicit_anchors_->validatedMembers(),
        explicit_anchors_->memberCounts(),visible_indices,getOpacityActivation().detach(),
        {anchor_opacity_accum_,anchor_visits_,anchor_slot_opacity_accum_,anchor_slot_below_prune_,anchor_slot_below_cutoff_},
        explicit_anchors_->inactiveSlots(),explicit_anchors_->recoveredSlots());
    return;
  }
  // One visit per Full view if any member is in the renderer's visibility set.
  auto anchors=torch::zeros({explicit_anchors_->size()},xyz_.options().dtype(torch::kBool));
  anchors.index_fill_(0,explicit_anchors_->rows().index_select(0,visible_indices),true);
  if(variableAnchorMembers()) {
    auto owners=explicit_anchors_->rows();
    auto opacity=getOpacityActivation().detach().reshape({-1}).to(torch::kFloat64);
    explicit_anchors_->refreshSlotActivity(getOpacityActivation());
    auto active=anchors.index_select(0,owners);
    auto observed=torch::where(active,opacity,torch::zeros_like(opacity));
    anchor_opacity_accum_.index_add_(0,owners,observed);
    anchor_visits_+=anchors.to(torch::kLong);
    anchor_slot_opacity_accum_+=observed;
    anchor_slot_below_prune_+=((opacity<.02)&active).to(torch::kLong);
    anchor_slot_below_cutoff_+=((opacity<1./255)&active).to(torch::kLong);
    return;
  }
  // Octree-GS sums all positive slot opacities, without dividing by slot count.
  auto opacity=getOpacityActivation().detach().clamp_min(0).to(torch::kFloat64).reshape({explicit_anchors_->size(),10});
  explicit_anchors_->refreshSlotActivity(getOpacityActivation());
  auto active=anchors.unsqueeze(1);
  auto observed=torch::where(active,opacity,torch::zeros_like(opacity));
  anchor_opacity_accum_+=observed.sum(1);
  anchor_visits_+=anchors.to(torch::kLong);
  anchor_slot_opacity_accum_+=observed;
  anchor_slot_below_prune_+=((opacity<.02)&active).to(torch::kLong);
  anchor_slot_below_cutoff_+=((opacity<1./255)&active).to(torch::kLong);
}

void GaussianModel::saveAnchorPruningState(std::filesystem::path path, int check_interval) const {
  torch::NoGradGuard guard;
  auto denominator=variableAnchorMembers()?anchor_visits_.index_select(0,explicit_anchors_->rows()).to(torch::kFloat64):
      anchor_visits_.to(torch::kFloat64).unsqueeze(1);
  auto observed=denominator>0;
  auto missing=torch::full_like(anchor_slot_opacity_accum_,std::numeric_limits<double>::quiet_NaN());
  torch::serialize::OutputArchive archive;
  archive.write("block_uids",explicit_anchors_->blockUIDs(),true);
  archive.write("levels",explicit_anchors_->levels(),true);
  archive.write("keys",explicit_anchors_->named_buffers()["keys"],true);
  if(variableAnchorMembers()) {
    archive.write("gaussian_owner",explicit_anchors_->rows(),true);
    archive.write("member_counts",explicit_anchors_->memberCounts(),true);
    archive.write("topology_round",explicit_anchors_->topologyRound(),true);
  }
  archive.write("mapping_iteration",torch::tensor(protection_.mapping_iteration,torch::kLong),true);
  archive.write("check_interval",torch::tensor(check_interval,torch::kLong),true);
  archive.write("opacity_accum",anchor_opacity_accum_,true);
  archive.write("anchor_visits",anchor_visits_,true);
  archive.write("slot_opacity_accum",anchor_slot_opacity_accum_,true);
  archive.write("slot_below_002_count",anchor_slot_below_prune_,true);
  archive.write("slot_below_cutoff_count",anchor_slot_below_cutoff_,true);
  archive.write("slot_mean_opacity",torch::where(observed,anchor_slot_opacity_accum_/denominator,missing),true);
  archive.write("slot_below_002_fraction",torch::where(observed,anchor_slot_below_prune_/denominator,missing),true);
  archive.write("slot_below_cutoff_fraction",torch::where(observed,anchor_slot_below_cutoff_/denominator,missing),true);
  archive.write("slot_current_opacity",variableAnchorMembers()?torch::sigmoid(opacity_.detach()).reshape({-1}):
      torch::sigmoid(opacity_.detach()).reshape({explicit_anchors_->size(),10}),true);
  if (explicit_anchors_->slotDeactivation()) archive.write("slot_inactive",explicit_anchors_->inactiveSlots(),true);
  archive.save_to(path.string());
}

torch::Tensor GaussianModel::anchorPruningMask(int check_interval) {
  MappingAuditRange trace("topology/anchor_prune_window");
  torch::NoGradGuard guard;
  TORCH_CHECK(explicit_anchors_ && check_interval>0 && anchor_visits_.numel()==explicit_anchors_->size() &&
      anchor_opacity_accum_.numel()==explicit_anchors_->size(),"Invalid Anchor pruning window");
  // Match Octree-GS adjust_anchor: strict count gate and strict opacity test.
  auto enough=anchor_visits_>check_interval*.8;
  auto prune=enough & (anchor_opacity_accum_<.005*anchor_visits_.to(torch::kFloat64));
  explicit_anchors_->closeSlotWindow(enough,anchor_visits_,anchor_slot_below_cutoff_,getOpacityActivation());
  if (explicit_anchors_->slotDeactivation())
    std::cout<<"[Slot Activity] mapping="<<protection_.mapping_iteration
             <<" inactive="<<explicit_anchors_->inactiveSlots().sum().item<int64_t>()
             <<" population="<<xyz_.size(0)<<" budget_population=all_stored\n";
  if(!anchor_pruning_diagnostics_.empty()) {
    std::filesystem::create_directories(anchor_pruning_diagnostics_);
    saveAnchorPruningState(anchor_pruning_diagnostics_/(
        "window_"+std::to_string(protection_.mapping_iteration)+".pt"),check_interval);
    const auto csv=anchor_pruning_diagnostics_/"windows.csv";
    const bool header=!std::filesystem::exists(csv);
    std::ofstream output(csv,std::ios::app);
    if(header) output<<"mapping_iteration,uid,visits,sufficient,deleted,mean_opacity_sum,always_below_002_slots,always_below_cutoff_slots,other_slot_opacity_share\n";
    auto visit=anchor_visits_.cpu().contiguous(),uids=explicit_anchors_->blockUIDs().cpu().contiguous();
    auto sufficient=enough.cpu().contiguous(),removed=prune.cpu().contiguous();
    auto total=anchor_opacity_accum_.cpu().contiguous();
    const auto owners=explicit_anchors_->rows();
    auto denominator=anchor_visits_.index_select(0,owners);
    auto always=(denominator>0)&(anchor_slot_below_cutoff_.reshape({-1})==denominator);
    auto low=torch::zeros_like(anchor_visits_),dead=torch::zeros_like(anchor_visits_);
    auto other=torch::zeros_like(anchor_opacity_accum_);
    low.index_add_(0,owners,((denominator>0)&(anchor_slot_below_prune_.reshape({-1})==denominator)).to(torch::kLong));
    dead.index_add_(0,owners,always.to(torch::kLong));
    other.index_add_(0,owners,torch::where(always,torch::zeros_like(anchor_slot_opacity_accum_.reshape({-1})),
                                         anchor_slot_opacity_accum_.reshape({-1})));
    low=low.cpu().contiguous();dead=dead.cpu().contiguous();other=other.cpu().contiguous();
    for(int64_t j=0;j<explicit_anchors_->size();++j) {
      const auto n=visit.accessor<int64_t,1>()[j];
      output<<protection_.mapping_iteration<<','<<uids.accessor<int64_t,1>()[j]<<','<<n<<','
            <<sufficient.accessor<bool,1>()[j]<<','<<removed.accessor<bool,1>()[j]<<',';
      if(n>0) output<<total.accessor<double,1>()[j]/n<<','<<low.accessor<int64_t,1>()[j]<<','<<dead.accessor<int64_t,1>()[j]<<','
                    <<(total.accessor<double,1>()[j]>0?other.accessor<double,1>()[j]/total.accessor<double,1>()[j]:std::numeric_limits<double>::quiet_NaN());
      else output<<"nan,nan,nan,nan";
      output<<'\n';
    }
  }
  std::cout<<"[Anchor Opacity Prune] check_interval="<<check_interval<<" min_opacity=0.005 success_threshold=0.8"
           <<" observed_enough="<<enough.sum().item<int64_t>()<<" removed="<<prune.sum().item<int64_t>()
           <<" pending="<<(~enough).sum().item<int64_t>()<<'\n';
  // Insufficiently observed anchors carry their evidence into the next check.
  anchor_opacity_accum_.masked_fill_(enough,0);
  anchor_visits_.masked_fill_(enough,0);
  auto complete=variableAnchorMembers()?enough.index_select(0,explicit_anchors_->rows()):enough.unsqueeze(1);
  anchor_slot_opacity_accum_.masked_fill_(complete,0);
  anchor_slot_below_prune_.masked_fill_(complete,0);
  anchor_slot_below_cutoff_.masked_fill_(complete,0);
  return variableAnchorMembers()?prune.index_select(0,explicit_anchors_->rows()):prune.repeat_interleave(10);
}

void GaussianModel::appendAnchorSeeds(const improvements::selector::AnchorBankImpl::BlockSeeds& seeds, int iteration) {
  if(!seeds.xyz.size(0))return;
  auto color=seeds.color;
  auto xyz=seeds.xyz, dc=sh_utils::RGB2SH(color).unsqueeze(1).contiguous();
  auto rest=torch::zeros({xyz.size(0),(max_sh_degree_+1)*(max_sh_degree_+1)-1,3},xyz.options());
  auto rotation=torch::zeros({xyz.size(0),4},xyz.options());rotation.select(1,0).fill_(1);
  auto sigma=seeds.sigma.log();
  auto opacity=general_utils::inverse_sigmoid(torch::full({xyz.size(0),1},1-std::pow(.5,1./10),xyz.options()));
  auto birth=torch::full({xyz.size(0)},iteration,xyz.options().dtype(torch::kInt32));
  densificationPostfix(xyz,dc,rest,opacity,sigma,rotation,birth);
}

void GaussianModel::createFromPcd(std::map<point3D_id_t, Point3D> pcd,
                                  const float spatial_lr_scale) {
  this->spatial_lr_scale_ = spatial_lr_scale;
  int num_points = static_cast<int>(pcd.size());
  torch::Tensor fused_point_cloud = torch::zeros(
      {num_points, 3},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  torch::Tensor color = torch::zeros(
      {num_points, 3},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  auto pcd_it = pcd.begin();
  for (int point_idx = 0; point_idx < num_points; ++point_idx) {
    auto& point = (*pcd_it).second;
    fused_point_cloud.index({point_idx, 0}) = point.xyz_(0);
    fused_point_cloud.index({point_idx, 1}) = point.xyz_(1);
    fused_point_cloud.index({point_idx, 2}) = point.xyz_(2);
    color.index({point_idx, 0}) = point.color_(0);
    color.index({point_idx, 1}) = point.color_(1);
    color.index({point_idx, 2}) = point.color_(2);
    ++pcd_it;
  }

  sparse_points_xyz_ = fused_point_cloud;
  sparse_points_color_ = color;

  torch::Tensor block_sigma;
  if(explicit_anchors_) {
    TORCH_CHECK(explicit_anchors_->size()==0,"Explicit map cannot be reinitialized from an arbitrary PCD");
    auto seeds=explicit_anchors_->observeBlocks(fused_point_cloud,color,{optimizer_.get(),anchor_feature_optimizer_});
    fused_point_cloud=seeds.xyz;color=seeds.color;block_sigma=seeds.sigma;
    anchor_opacity_accum_=torch::zeros({explicit_anchors_->size()},fused_point_cloud.options().dtype(torch::kFloat64));
    anchor_visits_=torch::zeros({explicit_anchors_->size()},fused_point_cloud.options().dtype(torch::kLong));
    anchor_slot_opacity_accum_=variableAnchorMembers()?torch::zeros({fused_point_cloud.size(0)},anchor_opacity_accum_.options()):
        torch::zeros({explicit_anchors_->size(),10},anchor_opacity_accum_.options());
    anchor_slot_below_prune_=torch::zeros_like(anchor_slot_opacity_accum_,anchor_visits_.options());
    anchor_slot_below_cutoff_=torch::zeros_like(anchor_slot_below_prune_);
    if(variableAnchorMembers()) {
      anchor_growth_gradient_accum_=torch::zeros({fused_point_cloud.size(0),1},fused_point_cloud.options());
      anchor_growth_denom_=torch::zeros_like(anchor_growth_gradient_accum_);
    }
  }

  torch::Tensor fused_color = sh_utils::RGB2SH(color);
  auto temp = this->max_sh_degree_ + 1;
  torch::Tensor features = torch::zeros(
      {fused_color.size(0), 3, temp * temp},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) =
      fused_color;
  features.index({torch::indexing::Slice(),
                  torch::indexing::Slice(3, features.size(1)),
                  torch::indexing::Slice(1, features.size(2))}) = 0.0f;

  // std::cout << "[Gaussian Model]Number of points at initialization : " <<
  // fused_point_cloud.size(0) << std::endl;

  torch::Tensor point_cloud_copy = fused_point_cloud.clone();
  torch::Tensor scales;
  if(block_sigma.defined()) scales=block_sigma.log();
  else {
    auto dist2=torch::clamp_min(distCUDA2(point_cloud_copy),0.0000001);
    scales=torch::log(torch::sqrt(dist2)*0.1).unsqueeze(1).repeat({1,3});
  }
  torch::Tensor rots =
      torch::zeros({fused_point_cloud.size(0), 4},
                   torch::TensorOptions().device(device_type_));
  rots.index({torch::indexing::Slice(), 0}) = 1;

  torch::Tensor opacities = general_utils::inverse_sigmoid(
      (explicit_anchors_ ? float(1-std::pow(.5,1./10)) : 0.5f) *
      torch::ones(
          {fused_point_cloud.size(0), 1},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

  this->exist_since_iter_ = torch::zeros(
      {fused_point_cloud.size(0)},
      torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
  protection_.reset(fused_point_cloud.size(0), torch::TensorOptions().device(device_type_));
  this->last_elastic_supervised_iter_ = torch::full({fused_point_cloud.size(0)}, protection_.mapping_iteration,
      fused_point_cloud.options().dtype(torch::kInt64).requires_grad(false));
  this->xyz_ = fused_point_cloud.requires_grad_();
  this->features_dc_ =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(0, 1)})
          .transpose(1, 2)
          .contiguous()
          .requires_grad_();
  this->features_rest_ =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(1, features.size(2))})
          .transpose(1, 2)
          .contiguous()
          .requires_grad_();
  this->scaling_ = scales.requires_grad_();
  this->rotation_ = rots.requires_grad_();
  this->opacity_ = opacities.requires_grad_();

  GAUSSIAN_MODEL_TENSORS_TO_VEC

  this->max_radii2D_ = torch::zeros(
      {this->getXYZ().size(0)}, torch::TensorOptions().device(device_type_));
  protection_.assertAligned(this->xyz_.size(0));
  assert(this->xyz_.size(0) == this->last_elastic_supervised_iter_.size(0));
  if (online_gi_reset_callback_)
    online_gi_reset_callback_(this->xyz_.size(0));
  if(private_compensation_)private_compensation_->append(xyz_.size(0),{}, {optimizer_.get(),anchor_feature_optimizer_});
}

void GaussianModel::increasePcd(std::vector<float> points,
                                std::vector<float> colors,
                                const int iteration) {
  // auto time1 = std::chrono::steady_clock::now();
  assert(points.size() == colors.size());
  assert(points.size() % 3 == 0);
  auto num_new_points = static_cast<int>(points.size() / 3);
  if (num_new_points == 0) return;

  torch::Tensor new_point_cloud =
      torch::from_blob(points.data(), {num_new_points, 3},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_);
  // torch::zeros({num_new_points, 3}, xyz_.options());
  torch::Tensor new_colors =
      torch::from_blob(colors.data(), {num_new_points, 3},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_);
  if(explicit_anchors_) { increasePcd(new_point_cloud,new_colors,iteration); return; }
  // torch::zeros({num_new_points, 3}, xyz_.options());

  sparse_points_xyz_ =
      torch::cat({sparse_points_xyz_, new_point_cloud}, /*dim=*/0);
  sparse_points_color_ =
      torch::cat({sparse_points_color_, new_colors}, /*dim=*/0);

  torch::Tensor new_fused_colors = sh_utils::RGB2SH(new_colors);
  auto temp = this->max_sh_degree_ + 1;
  torch::Tensor features = torch::zeros(
      {new_fused_colors.size(0), 3, temp * temp},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) =
      new_fused_colors;
  features.index({torch::indexing::Slice(),
                  torch::indexing::Slice(3, features.size(1)),
                  torch::indexing::Slice(1, features.size(2))}) = 0.0f;

  // std::cout << "[Gaussian Model]Number of points increase : "
  //           << num_new_points << std::endl;

  torch::Tensor dist2 =
      torch::clamp_min(distCUDA2(new_point_cloud.clone()), 0.0000001);
  torch::Tensor scales = torch::log(torch::sqrt(dist2) * 0.1);
  auto scales_ndimension = scales.ndimension();
  scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
  torch::Tensor rots =
      torch::zeros({new_point_cloud.size(0), 4},
                   torch::TensorOptions().device(device_type_));
  rots.index({torch::indexing::Slice(), 0}) = 1;
  torch::Tensor opacities = general_utils::inverse_sigmoid(
      0.5f *
      torch::ones(
          {new_point_cloud.size(0), 1},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

  torch::Tensor new_exist_since_iter = torch::full(
      {new_point_cloud.size(0)}, iteration,
      torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

  auto new_xyz = new_point_cloud;
  auto new_features_dc =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(0, 1)})
          .transpose(1, 2)
          .contiguous();
  auto new_features_rest =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(1, features.size(2))})
          .transpose(1, 2)
          .contiguous();
  auto new_opacities = opacities;
  auto new_scaling = scales;
  auto new_rotation = rots;
  // auto time2 = std::chrono::steady_clock::now();
  // auto time =
  // std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
  // std::cout << "increasePcd(umap) preparation time: " << time << " ms"
  // <<std::endl;

  densificationPostfix(new_xyz, new_features_dc, new_features_rest,
                       new_opacities, new_scaling, new_rotation,
                       new_exist_since_iter);

  c10::cuda::CUDACachingAllocator::emptyCache();
  // auto time3 = std::chrono::steady_clock::now();
  // time =
  // std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
  // std::cout << "increasePcd(umap) postfix time: " << time << " ms"
  // <<std::endl;
}

void GaussianModel::increasePcd(torch::Tensor& new_point_cloud,
                                torch::Tensor& new_colors,
                                const int iteration) {
  // auto time1 = std::chrono::steady_clock::now();
  auto num_new_points = new_point_cloud.size(0);
  if (num_new_points == 0) return;

  sparse_points_xyz_ =
      torch::cat({sparse_points_xyz_, new_point_cloud}, /*dim=*/0);
  sparse_points_color_ =
      torch::cat({sparse_points_color_, new_colors}, /*dim=*/0);

  if(explicit_anchors_) {
    appendAnchorSeeds(explicit_anchors_->observeBlocks(new_point_cloud,new_colors,
        {optimizer_.get(),anchor_feature_optimizer_}),iteration);
    return;
  }

  torch::Tensor new_fused_colors = sh_utils::RGB2SH(new_colors);
  auto temp = this->max_sh_degree_ + 1;
  torch::Tensor features = torch::zeros(
      {new_fused_colors.size(0), 3, temp * temp},
      torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
  features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) =
      new_fused_colors;
  features.index({torch::indexing::Slice(),
                  torch::indexing::Slice(3, features.size(1)),
                  torch::indexing::Slice(1, features.size(2))}) = 0.0f;

  // std::cout << "[Gaussian Model]Number of points increase : "
  //           << num_new_points << std::endl;

  torch::Tensor dist2 =
      torch::clamp_min(distCUDA2(new_point_cloud.clone()), 0.0000001);
  torch::Tensor scales = torch::log(torch::sqrt(dist2) * 0.1);
  auto scales_ndimension = scales.ndimension();
  scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
  torch::Tensor rots =
      torch::zeros({new_point_cloud.size(0), 4},
                   torch::TensorOptions().device(device_type_));
  rots.index({torch::indexing::Slice(), 0}) = 1;
  torch::Tensor opacities = general_utils::inverse_sigmoid(
      0.5f *
      torch::ones(
          {new_point_cloud.size(0), 1},
          torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

  torch::Tensor new_exist_since_iter = torch::full(
      {new_point_cloud.size(0)}, iteration,
      torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

  auto new_xyz = new_point_cloud;
  auto new_features_dc =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(0, 1)})
          .transpose(1, 2)
          .contiguous();
  auto new_features_rest =
      features
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(1, features.size(2))})
          .transpose(1, 2)
          .contiguous();
  auto new_opacities = opacities;
  auto new_scaling = scales;
  auto new_rotation = rots;
  // auto time2 = std::chrono::steady_clock::now();
  // auto time =
  // std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
  // std::cout << "increasePcd(tensor) preparation time: " << time << " ms"
  // <<std::endl;

  densificationPostfix(new_xyz, new_features_dc, new_features_rest,
                       new_opacities, new_scaling, new_rotation,
                       new_exist_since_iter);

  c10::cuda::CUDACachingAllocator::emptyCache();

  // auto time3 = std::chrono::steady_clock::now();
  // time =
  // std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
  // std::cout << "increasePcd(tensor) postfix time: " << time << " ms"
  // <<std::endl;
}

namespace {
GaussianTransformResult makeCorrection(const Sophus::SE3f& pose, float scale,
                                        const torch::TensorOptions& options) {
  GaussianTransformResult result;
  result.transform_4x4 = tensor_utils::EigenMatrix2TorchTensor(
      pose.matrix(), options.device().type()).to(options.device())
          .transpose(0, 1).detach().clone();
  const auto q = pose.unit_quaternion();
  result.rotation_wxyz = torch::tensor({q.w(), q.x(), q.y(), q.z()},
                                        options.dtype(torch::kFloat32));
  result.scale = scale;
  constexpr float kChangeEps = 1e-5f;
  const bool scale_changed = std::abs(scale - 1.0f) > kChangeEps;
  const bool rotation_changed = !pose.rotationMatrix().isApprox(
      Eigen::Matrix3f::Identity(), kChangeEps);
  // XYZ always follows the exact correction, including small translations.
  result.changed_attributes = toBits(GaussianAttrFlag::XYZ);
  if (scale_changed)
    result.changed_attributes |= toBits(GaussianAttrFlag::SCALE);
  if (rotation_changed)
    result.changed_attributes |= toBits(GaussianAttrFlag::ROTATION);
  return result;
}
}  // namespace

GaussianTransformResult GaussianModel::applyScaledTransformation(const float s,
                                               const Sophus::SE3f T) {
  if (!std::isfinite(s) || s <= 0.0f)
    throw std::invalid_argument("Sim(3) scale must be positive and finite");
  torch::NoGradGuard no_grad;
  auto correction = makeCorrection(T, s, xyz_.options());
  correction.affects_all = true;
  correction.population_count = xyz_.size(0);
  if (correction.population_count == 0) {
    correction.changed_attributes = toBits(GaussianAttrFlag::NONE);
    return correction;
  }
  auto transformed = applyGaussianTransform(
      {xyz_, scaling_, getRotationActivation()}, correction);
  xyz_ = replaceTensorToOptimizer(transformed.xyz, 0);
  Tensor_vec_xyz_ = {xyz_};
  if (correction.changed_attributes & toBits(GaussianAttrFlag::SCALE)) {
    scaling_ = replaceTensorToOptimizer(transformed.log_scaling, 4);
    Tensor_vec_scaling_ = {scaling_};
  }
  if (correction.changed_attributes & toBits(GaussianAttrFlag::ROTATION)) {
    rotation_ = replaceTensorToOptimizer(transformed.rotation, 5);
    Tensor_vec_rotation_ = {rotation_};
  }
  protection_.transportSnapshot(correction);
  if(explicit_anchors_) explicit_anchors_->transportBlocks(correction);
  if(private_compensation_)private_compensation_->transport(correction,{optimizer_.get(),anchor_feature_optimizer_});
  return correction;
}

GaussianTransformResult GaussianModel::scaledTransformVisiblePointsOfKeyframe(
    torch::Tensor& point_not_transformed_flags,
    const Sophus::SE3f& diff_pose,
    torch::Tensor& kf_world_view_transform,
    torch::Tensor& kf_full_proj_transform,
    const int kf_creation_iter,
    const int stable_num_iter_existence,
    int& num_transformed,
    const float scale) {
  if (!std::isfinite(scale) || scale <= 0.0f)
    throw std::invalid_argument("Sim(3) scale must be positive and finite");
  torch::NoGradGuard no_grad;

  torch::Tensor points = this->getXYZ();
  torch::Tensor rots = this->getRotationActivation();
  // torch::Tensor scales = this->scaling_;// * scale;

  torch::Tensor point_unstable_flags =
      torch::where(torch::abs(this->exist_since_iter_ - kf_creation_iter) <
                       stable_num_iter_existence,
                   true, false);

  auto transformed_mask = markVisibleUntransformedPoints(
      points, point_not_transformed_flags, point_unstable_flags,
      kf_world_view_transform, kf_full_proj_transform);
  auto indices = torch::nonzero(transformed_mask).reshape({-1});
  auto correction = makeCorrection(diff_pose, scale, points.options());
  correction.indices = indices.detach();
  correction.population_count = points.size(0);
  if (indices.numel() == 0) {
    correction.changed_attributes = toBits(GaussianAttrFlag::NONE);
    return correction;
  }
  auto transformed = applyGaussianTransform(
      GaussianAttrState{points, scaling_, rots}.select(indices), correction);
  points.index_copy_(0, indices, transformed.xyz);
  if (correction.changed_attributes & toBits(GaussianAttrFlag::ROTATION)) {
    // Preserve raw rotations outside the actual correction population.
    rotation_.index_copy_(0, indices, transformed.rotation);
    rotation_ = replaceTensorToOptimizer(rotation_, 5);
    Tensor_vec_rotation_ = {rotation_};
  }
  if (correction.changed_attributes & toBits(GaussianAttrFlag::SCALE)) {
    scaling_.index_copy_(0, indices, transformed.log_scaling);
    scaling_ = replaceTensorToOptimizer(scaling_, 4);
    Tensor_vec_scaling_ = {scaling_};
  }
  point_not_transformed_flags.index_put_({indices}, false);
  num_transformed += static_cast<int>(indices.numel());

  // torch::Tensor point_cloud_copy = points.clone();
  // torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy),
  // 0.0000001); torch::Tensor scales = torch::log(torch::sqrt(dist2)); auto
  // scales_ndimension = scales.ndimension(); scales =
  // scales.unsqueeze(scales_ndimension).repeat({1, 3});

  // Postfix
  // ==================================
  // param_groups[0] = xyz_
  // param_groups[1] = feature_dc_
  // param_groups[2] = feature_rest_
  // param_groups[3] = opacity_
  // param_groups[4] = scaling_
  // param_groups[5] = rotation_
  // ==================================
  torch::Tensor optimizable_xyz = this->replaceTensorToOptimizer(points, 0);
  this->xyz_ = optimizable_xyz;
  this->Tensor_vec_xyz_ = {this->xyz_};
  protection_.transportSnapshot(correction);
  if(variableAnchorMembers() && !fixedTenAnchors())
    local_anchor_moved_rows_=local_anchor_moved_rows_.defined()?torch::cat({local_anchor_moved_rows_,indices}):indices.clone();
  else if(explicit_anchors_) explicit_anchors_->transportBlocks(correction);
  if(private_compensation_)private_compensation_->transport(correction,{optimizer_.get(),anchor_feature_optimizer_});
  return correction;
}

void GaussianModel::trainingSetup(
    const GaussianOptimizationParams& training_args) {
  setPercentDense(training_args.percent_dense_);
  this->xyz_gradient_accum_ = torch::zeros(
      {this->getXYZ().size(0), 1}, torch::TensorOptions().device(device_type_));
  this->denom_ = torch::zeros({this->getXYZ().size(0), 1},
                              torch::TensorOptions().device(device_type_));

  torch::optim::AdamOptions adam_options;
  adam_options.set_lr(0.0);
  adam_options.eps() = 1e-15;

  // this->optimizer_.reset(new torch::optim::Adam(Tensor_vec_xyz_,
  // adam_options));
  this->optimizer_.reset(new SparseGaussianAdam(Tensor_vec_xyz_, adam_options));
  optimizer_->param_groups()[0].options().set_lr(
      training_args.position_lr_init_ * this->spatial_lr_scale_);

  optimizer_->add_param_group(Tensor_vec_feature_dc_);
  optimizer_->param_groups()[1].options().set_lr(training_args.feature_lr_);

  optimizer_->add_param_group(Tensor_vec_feature_rest_);
  optimizer_->param_groups()[2].options().set_lr(training_args.feature_lr_ /
                                                 20.0);

  optimizer_->add_param_group(Tensor_vec_opacity_);
  optimizer_->param_groups()[3].options().set_lr(training_args.opacity_lr_);

  optimizer_->add_param_group(Tensor_vec_scaling_);
  optimizer_->param_groups()[4].options().set_lr(training_args.scaling_lr_);

  optimizer_->add_param_group(Tensor_vec_rotation_);
  optimizer_->param_groups()[5].options().set_lr(training_args.rotation_lr_);

  // get_expon_lr_func
  lr_init_ = training_args.position_lr_init_ * this->spatial_lr_scale_;
  lr_final_ = training_args.position_lr_final_ * this->spatial_lr_scale_;
  lr_delay_mult_ = training_args.position_lr_delay_mult_;
  max_steps_ = training_args.position_lr_max_steps_;
}

float GaussianModel::updateLearningRate(int step) {
  // def update_learning_rate(self, iteration):
  //     ''' Learning rate scheduling per step '''
  //     for param_group in self.optimizer.param_groups:
  //         if param_group["name"] == "xyz":
  //             lr = self.xyz_scheduler_args(iteration)
  //             param_group['lr'] = lr
  //             return lr
  float lr = this->exponLrFunc(step);
  optimizer_->param_groups()[0].options().set_lr(lr);  // Tensor_vec_xyz_
  return lr;
}

// ==================================
// param_groups[0] = xyz_
// param_groups[1] = feature_dc_
// param_groups[2] = feature_rest_
// param_groups[3] = opacity_
// param_groups[4] = scaling_
// param_groups[5] = rotation_
// ==================================
void GaussianModel::setPositionLearningRate(float position_lr) {
  optimizer_->param_groups()[0].options().set_lr(position_lr *
                                                 this->spatial_lr_scale_);
}
void GaussianModel::setFeatureLearningRate(float feature_lr) {
  optimizer_->param_groups()[1].options().set_lr(feature_lr);
  optimizer_->param_groups()[2].options().set_lr(feature_lr / 20.0);
}
void GaussianModel::setOpacityLearningRate(float opacity_lr) {
  optimizer_->param_groups()[3].options().set_lr(opacity_lr);
}
void GaussianModel::setScalingLearningRate(float scaling_lr) {
  optimizer_->param_groups()[4].options().set_lr(scaling_lr);
}
void GaussianModel::setRotationLearningRate(float rot_lr) {
  optimizer_->param_groups()[5].options().set_lr(rot_lr);
}

void GaussianModel::resetOpacity() {
  torch::Tensor opacities_new = general_utils::inverse_sigmoid(
      torch::min(this->getOpacityActivation(),
                 torch::ones_like(this->getOpacityActivation() * 0.01)));
  torch::Tensor optimizable_tensors =
      this->replaceTensorToOptimizer(opacities_new, 3);  // "opacity"
  this->opacity_ = optimizable_tensors;
  this->Tensor_vec_opacity_ = {this->opacity_};
}

torch::Tensor GaussianModel::replaceTensorToOptimizer(torch::Tensor& tensor,
                                                      int tensor_idx) {
  auto& param = this->optimizer_->param_groups()[tensor_idx].params()[0];
  auto& state = optimizer_->state();
  auto key = param.unsafeGetTensorImpl();
  auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
  auto new_state = std::make_unique<torch::optim::AdamParamState>();
  new_state->step(stored_state.step());
  new_state->exp_avg(torch::zeros_like(tensor));
  new_state->exp_avg_sq(torch::zeros_like(tensor));
  // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone()); // needed
  // only when options.amsgrad(true), which is false by default

  state.erase(key);
  param = tensor.requires_grad_();
  key = param.unsafeGetTensorImpl();
  state[key] = std::move(new_state);

  auto optimizable_tensors = param;
  return optimizable_tensors;
}

void GaussianModel::prunePoints(torch::Tensor& mask) {
  MappingAuditRange trace("topology/gaussian_prune_and_moments");
  if(fixedTenAnchors()) {
    auto removed=torch::zeros_like(explicit_anchors_->memberCounts());
    removed.index_add_(0,explicit_anchors_->rows(),mask.to(torch::kLong));
    TORCH_CHECK(((removed==0)|(removed==10)).all().item<bool>(),"Fixed-ten mode cannot prune individual Gaussian members");
  }
  if(explicit_anchors_&&!variableAnchorMembers()) mask=explicit_anchors_->wholeBlockPrune(mask);
  if (!last_elastic_supervised_iter_.defined() ||
      last_elastic_supervised_iter_.numel() != xyz_.size(0)) {
    last_elastic_supervised_iter_ = torch::full({xyz_.size(0)}, protection_.mapping_iteration, xyz_.options().dtype(torch::kInt64).requires_grad(false));
  }
  auto valid_points_mask = ~mask;
  if(private_compensation_)private_compensation_->select(torch::nonzero(valid_points_mask).reshape({-1}),
      {optimizer_.get(),anchor_feature_optimizer_});
  if(variableAnchorMembers()) {
    auto counts=torch::zeros_like(explicit_anchors_->memberCounts());
    auto surviving_owners=explicit_anchors_->rows().index({valid_points_mask});
    counts.index_add_(0,surviving_owners,torch::ones_like(surviving_owners));
    auto changed=counts!=explicit_anchors_->memberCounts();
    auto alive=explicit_anchors_->pruneVariableMembers(valid_points_mask,{optimizer_.get(),anchor_feature_optimizer_});
    anchor_opacity_accum_=anchor_opacity_accum_.index_select(0,alive);
    anchor_visits_=anchor_visits_.index_select(0,alive);
    anchor_slot_opacity_accum_=anchor_slot_opacity_accum_.index({valid_points_mask});
    anchor_slot_below_prune_=anchor_slot_below_prune_.index({valid_points_mask});
    anchor_slot_below_cutoff_=anchor_slot_below_cutoff_.index({valid_points_mask});
    anchor_growth_gradient_accum_=anchor_growth_gradient_accum_.index({valid_points_mask});
    anchor_growth_denom_=anchor_growth_denom_.index({valid_points_mask});
    resetAnchorWindows(changed.index_select(0,alive));
  } else if(explicit_anchors_) {
    auto keep=valid_points_mask.reshape({explicit_anchors_->size(),10}).select(1,0);
    anchor_opacity_accum_=anchor_opacity_accum_.index({keep});
    anchor_visits_=anchor_visits_.index({keep});
    anchor_slot_opacity_accum_=anchor_slot_opacity_accum_.index({keep});
    anchor_slot_below_prune_=anchor_slot_below_prune_.index({keep});
    anchor_slot_below_cutoff_=anchor_slot_below_cutoff_.index({keep});
    explicit_anchors_->pruneBlocks(valid_points_mask,{optimizer_.get(),anchor_feature_optimizer_});
  }

  // _prune_optimizer
  std::vector<torch::Tensor> optimizable_tensors(6);
  auto& param_groups = this->optimizer_->param_groups();
  auto& state = this->optimizer_->state();
  for (int group_idx = 0; group_idx < 6; ++group_idx) {
    auto& param = param_groups[group_idx].params()[0];
    auto key = param.unsafeGetTensorImpl();
    if (state.find(key) != state.end()) {
      auto& stored_state =
          static_cast<torch::optim::AdamParamState&>(*state[key]);
      auto new_state = std::make_unique<torch::optim::AdamParamState>();
      new_state->step(stored_state.step());
      new_state->exp_avg(
          stored_state.exp_avg().index({valid_points_mask}).clone());
      new_state->exp_avg_sq(
          stored_state.exp_avg_sq().index({valid_points_mask}).clone());
      // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone()); //
      // needed only when options.amsgrad(true), which is false by default

      state.erase(key);
      param = param.index({valid_points_mask}).requires_grad_();
      key = param.unsafeGetTensorImpl();
      state[key] = std::move(new_state);
      optimizable_tensors[group_idx] = param;
    } else {
      param = param.index({valid_points_mask}).requires_grad_();
      optimizable_tensors[group_idx] = param;
    }
  }

  // ==================================
  // param_groups[0] = xyz_
  // param_groups[1] = feature_dc_
  // param_groups[2] = feature_rest_
  // param_groups[3] = opacity_
  // param_groups[4] = scaling_
  // param_groups[5] = rotation_
  // ==================================
  this->xyz_ = optimizable_tensors[0];
  this->features_dc_ = optimizable_tensors[1];
  this->features_rest_ = optimizable_tensors[2];
  this->opacity_ = optimizable_tensors[3];
  this->scaling_ = optimizable_tensors[4];
  this->rotation_ = optimizable_tensors[5];

  GAUSSIAN_MODEL_TENSORS_TO_VEC

  this->exist_since_iter_ = this->exist_since_iter_.index({valid_points_mask});
  protection_.prune(valid_points_mask);
  this->last_elastic_supervised_iter_ =
      this->last_elastic_supervised_iter_.index({valid_points_mask});

  this->xyz_gradient_accum_ =
      this->xyz_gradient_accum_.index({valid_points_mask});

  this->denom_ = this->denom_.index({valid_points_mask});
  this->max_radii2D_ = this->max_radii2D_.index({valid_points_mask});
  protection_.assertAligned(this->xyz_.size(0));
  assert(this->xyz_.size(0) == this->last_elastic_supervised_iter_.size(0));
  if (online_gi_prune_callback_)
    online_gi_prune_callback_(valid_points_mask);
}

torch::Tensor GaussianModel::getSelectorMatureMask() const {
  return protection_.matureMask();
}

void GaussianModel::updateElasticSupervision(
    const torch::Tensor& indices, std::int64_t mapping_iter) {
  if (!indices.defined() || indices.numel() == 0 ||
      this->last_elastic_supervised_iter_.numel() == 0)
    return;

  torch::NoGradGuard no_grad;
  auto long_indices = indices.to(torch::kLong);
  this->last_elastic_supervised_iter_.index_put_(
      {long_indices}, mapping_iter);
}

void GaussianModel::densificationPostfix(torch::Tensor& new_xyz,
                                         torch::Tensor& new_features_dc,
                                         torch::Tensor& new_features_rest,
                                         torch::Tensor& new_opacities,
                                         torch::Tensor& new_scaling,
                                         torch::Tensor& new_rotation,
                                         torch::Tensor& new_exist_since_iter,
                                         AppendKind append_kind,
                                         const torch::Tensor& parent_indices) {
  MappingAuditRange trace("topology/gaussian_append_and_moments");
  if(explicit_anchors_&&!variableAnchorMembers()) {
    TORCH_CHECK(append_kind==AppendKind::Append && new_xyz.size(0)%10==0,
                "Explicit map forbids pointwise clone/split/append");
    explicit_anchors_->assertBlocks(xyz_.size(0)+new_xyz.size(0));
  }
  if (!last_elastic_supervised_iter_.defined() ||
      last_elastic_supervised_iter_.numel() != xyz_.size(0)) {
    last_elastic_supervised_iter_ = torch::full({xyz_.size(0)}, protection_.mapping_iteration, xyz_.options().dtype(torch::kInt64).requires_grad(false));
  }
  // cat_tensors_to_optimizer
  if(variableAnchorMembers()) {
    if(parent_indices.defined()) {
      TORCH_CHECK(parent_indices.numel()==new_xyz.size(0),"Variable append requires one owner per child");
      explicit_anchors_->appendVariableMembers(parent_indices);
    }
    TORCH_CHECK(explicit_anchors_->rows().numel()==xyz_.size(0)+new_xyz.size(0),
                "Variable append lost Gaussian ownership");
  }
  std::vector<torch::Tensor> optimizable_tensors(6);
  std::vector<torch::Tensor> tensors_dict = {new_xyz,           new_features_dc,
                                             new_features_rest, new_opacities,
                                             new_scaling,       new_rotation};
  if(private_compensation_)private_compensation_->append(new_xyz.size(0),parent_indices,
      {optimizer_.get(),anchor_feature_optimizer_});
  auto& param_groups = this->optimizer_->param_groups();
  auto& state = this->optimizer_->state();
  for (int group_idx = 0; group_idx < 6; ++group_idx) {
    auto& group = param_groups[group_idx];
    assert(group.params().size() == 1);
    auto& extension_tensor = tensors_dict[group_idx];
    auto& param = group.params()[0];
    auto key = param.unsafeGetTensorImpl();
    if (state.find(key) != state.end()) {
      auto& stored_state =
          static_cast<torch::optim::AdamParamState&>(*state[key]);
      auto new_state = std::make_unique<torch::optim::AdamParamState>();
      new_state->step(stored_state.step());
      new_state->exp_avg(torch::cat(
          {stored_state.exp_avg().clone(), torch::zeros_like(extension_tensor)},
          /*dim=*/0));
      new_state->exp_avg_sq(torch::cat({stored_state.exp_avg_sq().clone(),
                                        torch::zeros_like(extension_tensor)},
                                       /*dim=*/0));
      // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone());  //
      // needed only when options.amsgrad(true), which is false by default

      state.erase(key);
      param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
      key = param.unsafeGetTensorImpl();
      state[key] = std::move(new_state);

      optimizable_tensors[group_idx] = param;
    } else {
      param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
      optimizable_tensors[group_idx] = param;
    }
  }

  // ==================================
  // param_groups[0] = xyz_
  // param_groups[1] = feature_dc_
  // param_groups[2] = feature_rest_
  // param_groups[3] = opacity_
  // param_groups[4] = scaling_
  // param_groups[5] = rotation_
  // ==================================
  this->xyz_ = optimizable_tensors[0];
  this->features_dc_ = optimizable_tensors[1];
  this->features_rest_ = optimizable_tensors[2];
  this->opacity_ = optimizable_tensors[3];
  this->scaling_ = optimizable_tensors[4];
  this->rotation_ = optimizable_tensors[5];

  GAUSSIAN_MODEL_TENSORS_TO_VEC

  this->exist_since_iter_ =
      torch::cat({this->exist_since_iter_, new_exist_since_iter}, /*dim=*/0);
  protection_.append(new_xyz.size(0), parent_indices);
  this->last_elastic_supervised_iter_ = torch::cat(
      {this->last_elastic_supervised_iter_, torch::full({new_xyz.size(0)},
       protection_.mapping_iteration, this->last_elastic_supervised_iter_.options())});

  if(variableAnchorMembers()) {
    const auto added_anchors=explicit_anchors_->size()-anchor_visits_.numel();
    anchor_opacity_accum_=torch::cat({anchor_opacity_accum_,torch::zeros({added_anchors},anchor_opacity_accum_.options())});
    anchor_visits_=torch::cat({anchor_visits_,torch::zeros({added_anchors},anchor_visits_.options())});
    anchor_slot_opacity_accum_=torch::cat({anchor_slot_opacity_accum_,torch::zeros({new_xyz.size(0)},anchor_slot_opacity_accum_.options())});
    anchor_slot_below_prune_=torch::cat({anchor_slot_below_prune_,torch::zeros({new_xyz.size(0)},anchor_slot_below_prune_.options())});
    anchor_slot_below_cutoff_=torch::cat({anchor_slot_below_cutoff_,torch::zeros({new_xyz.size(0)},anchor_slot_below_cutoff_.options())});
    anchor_growth_gradient_accum_=torch::cat({anchor_growth_gradient_accum_,torch::zeros({new_xyz.size(0),1},xyz_.options().requires_grad(false))});
    anchor_growth_denom_=torch::cat({anchor_growth_denom_,torch::zeros({new_xyz.size(0),1},xyz_.options().requires_grad(false))});
    auto changed=torch::zeros({explicit_anchors_->size()},explicit_anchors_->rows().options().dtype(torch::kBool));
    changed.index_fill_(0,explicit_anchors_->rows().slice(0,-new_xyz.size(0)),true);
    resetAnchorWindows(changed);
    xyz_gradient_accum_=torch::cat({xyz_gradient_accum_,torch::zeros({new_xyz.size(0),1},xyz_gradient_accum_.options())});
    denom_=torch::cat({denom_,torch::zeros({new_xyz.size(0),1},denom_.options())});
    max_radii2D_=torch::cat({max_radii2D_,torch::zeros({new_xyz.size(0)},max_radii2D_.options())});
  } else if(explicit_anchors_) {
    const auto added_anchors=new_xyz.size(0)/10;
    anchor_opacity_accum_=torch::cat({anchor_opacity_accum_,torch::zeros({added_anchors},anchor_opacity_accum_.options())});
    anchor_visits_=torch::cat({anchor_visits_,torch::zeros({added_anchors},anchor_visits_.options())});
    anchor_slot_opacity_accum_=torch::cat({anchor_slot_opacity_accum_,torch::zeros({added_anchors,10},anchor_slot_opacity_accum_.options())});
    anchor_slot_below_prune_=torch::cat({anchor_slot_below_prune_,torch::zeros({added_anchors,10},anchor_slot_below_prune_.options())});
    anchor_slot_below_cutoff_=torch::cat({anchor_slot_below_cutoff_,torch::zeros({added_anchors,10},anchor_slot_below_cutoff_.options())});
    xyz_gradient_accum_=torch::cat({xyz_gradient_accum_,torch::zeros({new_xyz.size(0),1},xyz_gradient_accum_.options())});
    denom_=torch::cat({denom_,torch::zeros({new_xyz.size(0),1},denom_.options())});
    max_radii2D_=torch::cat({max_radii2D_,torch::zeros({new_xyz.size(0)},max_radii2D_.options())});
  } else {
    xyz_gradient_accum_=torch::zeros({getXYZ().size(0),1},torch::TensorOptions().device(device_type_));
    denom_=torch::zeros({getXYZ().size(0),1},torch::TensorOptions().device(device_type_));
    max_radii2D_=torch::zeros({getXYZ().size(0)},torch::TensorOptions().device(device_type_));
  }
  protection_.assertAligned(this->xyz_.size(0));
  assert(this->xyz_.size(0) == this->last_elastic_supervised_iter_.size(0));
  if (online_gi_append_callback_ && new_xyz.size(0) > 0)
    online_gi_append_callback_(new_xyz.size(0));
  if (gi_append_event_callback_ && new_xyz.size(0) > 0)
    gi_append_event_callback_(new_xyz.size(0), append_kind);
}

void GaussianModel::setOnlineGIStateCallbacks(
    OnlineGIAppendCallback on_append,
    OnlineGIPruneCallback on_prune,
    OnlineGIResetCallback on_reset, GIAppendEventCallback on_append_event) {
  online_gi_append_callback_ = std::move(on_append);
  online_gi_prune_callback_ = std::move(on_prune);
  online_gi_reset_callback_ = std::move(on_reset);
  gi_append_event_callback_ = std::move(on_append_event);
}

void GaussianModel::densifyAndSplit(torch::Tensor& grads,
                                    float grad_threshold,
                                    float scene_extent,
                                    int N) {
  TORCH_CHECK(!explicit_anchors_,"Explicit map disables Gaussian split");
  int n_init_points = this->getXYZ().size(0);
  // Extract points that satisfy the gradient condition
  auto padded_grad = torch::zeros({n_init_points},
                                  torch::TensorOptions().device(device_type_));
  padded_grad.slice(/*dim=*/0L, /*start=*/0, /*end=*/grads.size(0))
      .copy_(grads.squeeze());
  auto selected_pts_mask =
      torch::where(padded_grad >= grad_threshold, true, false);
  selected_pts_mask = torch::logical_and(
      selected_pts_mask,
      std::get<0>(torch::max(this->getScalingActivation(), /*dim=*/1)) >
          percentDense() * scene_extent);

  auto new_xyz = this->getXYZ().index({selected_pts_mask}).repeat({N, 1});
  auto samples = torch::normal(0.0f, 1.0f, new_xyz.sizes()).to(device_type_);
  samples.index_put_({samples.abs() > 3.0f}, 0.0f);
  auto new_transforms = this->getScalingRotationActivation()
                      .index({selected_pts_mask})
                      .repeat({N, 1, 1});
  new_xyz += new_transforms.matmul(samples.unsqueeze(-1)).squeeze(-1);
  auto new_scaling = torch::log(
      this->getScalingActivation().index({selected_pts_mask}).repeat({N, 1}) /
      (0.8 * N));  // scaling_inverse_activation
  auto new_rotation = this->rotation_.index({selected_pts_mask}).repeat({N, 1});
  auto new_features_dc =
      this->features_dc_.index({selected_pts_mask}).repeat({N, 1, 1});
  auto new_features_rest =
      this->features_rest_.index({selected_pts_mask}).repeat({N, 1, 1});
  auto new_opacity = this->opacity_.index({selected_pts_mask}).repeat({N, 1});

  auto new_exist_since_iter =
      this->exist_since_iter_.index({selected_pts_mask}).repeat({N});
  const auto parent_indices = torch::nonzero(selected_pts_mask).reshape({-1}).repeat({N});
  this->densificationPostfix(new_xyz, new_features_dc, new_features_rest,
                             new_opacity, new_scaling, new_rotation,
                             new_exist_since_iter, AppendKind::Split, parent_indices);

  auto prune_filter = torch::cat(
      {selected_pts_mask,
       torch::zeros(
           {(N * selected_pts_mask.sum()).item<int>()},
           torch::TensorOptions().device(device_type_).dtype(torch::kBool))});
  this->prunePoints(prune_filter);
}

void GaussianModel::densifyAndClone(torch::Tensor& grads,
                                    float grad_threshold,
                                    float scene_extent) {
  TORCH_CHECK(!explicit_anchors_,"Explicit map disables Gaussian clone");
  // Extract points that satisfy the gradient condition
  auto selected_pts_mask = torch::where(
      torch::frobenius_norm(grads, /*dim=*/-1) >= grad_threshold, true, false);
  selected_pts_mask = torch::logical_and(
      selected_pts_mask,
      std::get<0>(torch::max(this->getScalingActivation(), /*dim=*/1)) <=
          percentDense() * scene_extent);

  auto new_xyz = this->xyz_.index({selected_pts_mask});
  auto new_features_dc = this->features_dc_.index({selected_pts_mask});
  auto new_features_rest = this->features_rest_.index({selected_pts_mask});
  auto new_opacities = this->opacity_.index({selected_pts_mask});
  auto new_scaling = this->scaling_.index({selected_pts_mask});
  auto new_rotation = this->rotation_.index({selected_pts_mask});

  auto new_exist_since_iter =
      this->exist_since_iter_.index({selected_pts_mask});
  const auto parent_indices = torch::nonzero(selected_pts_mask).reshape({-1});
  this->densificationPostfix(new_xyz, new_features_dc, new_features_rest,
                             new_opacities, new_scaling, new_rotation,
                             new_exist_since_iter, AppendKind::Clone, parent_indices);
}

void GaussianModel::resetAnchorWindows(const torch::Tensor& changed) {
  if(!variableAnchorMembers())return;
  auto affected=changed.index_select(0,explicit_anchors_->rows());
  anchor_opacity_accum_.masked_fill_(changed,0);anchor_visits_.masked_fill_(changed,0);
  anchor_slot_opacity_accum_.masked_fill_(affected,0);
  anchor_slot_below_prune_.masked_fill_(affected,0);anchor_slot_below_cutoff_.masked_fill_(affected,0);
}

void GaussianModel::densifyVariableMembers(const torch::Tensor& gradients,float threshold,float extent) {
  MappingAuditRange trace("topology/gaussian_candidates_clone_split");
  const auto total=xyz_.size(0);if(!total)return;
  // The legacy transaction returns without touching leaves or moments when
  // no finite gradient passes the threshold. Keep that exact no-change case
  // on device instead of copying four populations and stable-sorting N rows.
  if(explicit_anchors_->fusedFullStats() && gradients.is_cuda()) {
    const auto candidate=torch::isfinite(gradients)&(gradients>=threshold);
    if(!candidate.any().item<bool>())return;
  }
  auto grad=gradients.reshape({-1}).to(torch::kCPU,torch::kFloat64).contiguous();
  auto counts=explicit_anchors_->memberCounts().cpu().contiguous();
  auto owners=explicit_anchors_->rows().cpu().contiguous();
  auto large=(std::get<0>(getScalingActivation().max(1))>percentDense()*extent).cpu().contiguous();
  auto g=grad.accessor<double,1>();auto c=counts.accessor<int64_t,1>(),a=owners.accessor<int64_t,1>();
  std::vector<int64_t> order(total),clone,split;
  std::iota(order.begin(),order.end(),0);
  { MappingAuditRange sorting_trace("topology/cpu_stable_sort");
  std::stable_sort(order.begin(),order.end(),[&](int64_t i,int64_t j){
    const double x=std::isfinite(g[i])?g[i]:-1,y=std::isfinite(g[j])?g[j]:-1;
    return x==y?i<j:x>y;
  });
  }
  std::vector<int64_t> remaining(counts.numel());for(int64_t j=0;j<counts.numel();++j)remaining[j]=10-c[j];
  for(auto i:order)if(std::isfinite(g[i])&&g[i]>=threshold&&remaining[a[i]]>0) {
    (large.accessor<bool,1>()[i]?split:clone).push_back(i);--remaining[a[i]];
  }
  if(clone.empty()&&split.empty())return;
  auto options=explicit_anchors_->rows().options();
  auto clones=torch::tensor(clone,options),splits=torch::tensor(split,options);
  auto parents=torch::cat({clones,splits.repeat({2})});
  auto xyz=xyz_.index_select(0,parents).clone(),scales=scaling_.index_select(0,parents).clone();
  if(!split.empty()) {
    auto split_xyz=xyz.slice(0,clones.numel());
    auto samples=torch::normal(0.f,1.f,split_xyz.sizes()).to(xyz_.device());
    samples.index_put_({samples.abs()>3.f},0.f);
    auto transform=getScalingRotationActivation().index_select(0,splits).repeat({2,1,1});
    split_xyz.add_(transform.matmul(samples.unsqueeze(-1)).squeeze(-1));
    scales.slice(0,clones.numel()).sub_(std::log(1.6));
  }
  auto dc=features_dc_.index_select(0,parents),rest=features_rest_.index_select(0,parents);
  auto opacity=opacity_.index_select(0,parents),rotation=rotation_.index_select(0,parents);
  auto birth=exist_since_iter_.index_select(0,parents);
  // One graph-free transaction: insert children, then remove split parents.
  // The capacity check uses net +1 per split; no render sees the transient rows.
  auto append=[&](int64_t first,int64_t last,AppendKind kind) {
    if(first==last)return;
    auto x=xyz.slice(0,first,last),d=dc.slice(0,first,last),f=rest.slice(0,first,last);
    auto o=opacity.slice(0,first,last),s=scales.slice(0,first,last),r=rotation.slice(0,first,last),b=birth.slice(0,first,last);
    densificationPostfix(x,d,f,o,s,r,b,kind,parents.slice(0,first,last));
  };
  append(0,clones.numel(),AppendKind::Clone);append(clones.numel(),parents.numel(),AppendKind::Split);
  auto remove=torch::zeros({xyz_.size(0)},options.dtype(torch::kBool));remove.index_fill_(0,splits,true);
  prunePoints(remove);
  std::cout<<"[Variable Gaussian Densify] clones="<<clone.size()<<" split_parents="<<split.size()
           <<" net_added="<<clone.size()+split.size()<<'\n';
}

void GaussianModel::densifyAndPrune(float max_grad,
                                    float min_opacity,
                                    float extent,
                                    int max_screen_size,
                                    int anchor_check_interval) {
  MappingAuditRange trace("topology/complete_event");
  auto grads = this->xyz_gradient_accum_ / this->denom_;
  grads.index_put_({grads.isnan()}, 0.0f);
  if(variableAnchorMembers()) {
    torch::NoGradGuard guard;
    const auto round=explicit_anchors_->topologyRound().item<int64_t>();
    const bool gaussian_round=!fixedTenAnchors() && round%2==0;
    std::cout<<"[Alternating Topology] round="<<round<<" phase="<<(gaussian_round?"Gaussian":"Anchor")
             <<" mapping="<<protection_.mapping_iteration<<'\n';
    if(gaussian_round) {
      int64_t removed=0;
      if(!anchor_only_pruning_) {
        auto remove=(getOpacityActivation()<min_opacity).reshape({-1});
        if(max_screen_size)remove|=(max_radii2D_>max_screen_size)|
            (std::get<0>(getScalingActivation().max(1))>.1f*extent);
        removed=remove.sum().item<int64_t>();gaussian_pruned_members_+=removed;
        grads=grads.index({~remove});
        prunePoints(remove);
      }
      std::cout<<"[Variable Gaussian Prune] enabled="<<!anchor_only_pruning_<<" removed="<<removed<<'\n';
      densifyVariableMembers(grads,max_grad,extent);
      xyz_gradient_accum_.zero_();denom_.zero_();max_radii2D_.zero_();
    } else {
      auto anchor_grads=anchor_growth_gradient_accum_/anchor_growth_denom_.clamp_min(1);
      if(anchor_growth_threshold_==0) {
        auto root=explicit_anchors_->levels().index_select(0,explicit_anchors_->rows())==0;
        auto valid=root & (anchor_growth_denom_.reshape({-1})>0) & torch::isfinite(anchor_grads.reshape({-1})) &
            (anchor_grads.reshape({-1})>0);
        auto values=anchor_grads.reshape({-1}).index({valid});
        if(values.numel())anchor_growth_threshold_=torch::quantile(values,.9).item<double>();
      }
      if(anchor_growth_threshold_>0) {
        auto color=(features_dc_.squeeze(1)*0.28209479177387814+.5).clamp(0,1);
        appendAnchorSeeds(explicit_anchors_->growBlocks(xyz_,color,anchor_grads,anchor_growth_denom_,anchor_growth_threshold_,2,
            {optimizer_.get(),anchor_feature_optimizer_},anchor_refine_requires_full_),int(protection_.mapping_iteration));
      }
      auto remove=anchorPruningMask(anchor_check_interval);
      anchor_pruned_members_+=remove.sum().item<int64_t>();prunePoints(remove);
      anchor_growth_gradient_accum_.zero_();anchor_growth_denom_.zero_();
      if(fixedTenAnchors()){xyz_gradient_accum_.zero_();denom_.zero_();max_radii2D_.zero_();}
    }
    explicit_anchors_->topologyRound().add_(1);
    explicit_anchors_->assertBlocks(xyz_.size(0));
    { MappingAuditRange cache_trace("topology/empty_cache");c10::cuda::CUDACachingAllocator::emptyCache(); }
    return;
  }
  if(explicit_anchors_) {
    if(anchor_growth_threshold_==0) {
      auto root=explicit_anchors_->levels().index_select(0,explicit_anchors_->rows())==0;
      auto valid=root & (denom_.reshape({-1})>0) & torch::isfinite(grads.reshape({-1})) & (grads.reshape({-1})>0);
      auto values=grads.reshape({-1}).index({valid});
      if(values.numel()) anchor_growth_threshold_=torch::quantile(values,.9).item<double>();
      std::cout<<"[Explicit Anchor Calibration] Full-only native gradient q90 tau0="<<anchor_growth_threshold_
               <<" visible_positive_L0="<<values.numel()<<" alpha=2\n";
    }
    if(anchor_growth_threshold_>0) {
      auto color=(features_dc_.squeeze(1)*0.28209479177387814+.5).clamp(0,1);
      appendAnchorSeeds(explicit_anchors_->growBlocks(xyz_,color,grads,denom_,anchor_growth_threshold_,2,
          {optimizer_.get(),anchor_feature_optimizer_}),int(protection_.mapping_iteration));
    }
  } else {
    this->densifyAndClone(grads, max_grad, extent);
    this->densifyAndSplit(grads, max_grad, extent, 2);
  }

  auto prune_mask = explicit_anchors_ ? anchorPruningMask(anchor_check_interval) :
      (this->getOpacityActivation() < min_opacity).squeeze();
  if (!explicit_anchors_ && max_screen_size) {
    auto big_points_vs = this->max_radii2D_ > max_screen_size;
    auto big_points_ws =
        std::get<0>(this->getScalingActivation().max(/*dim=*/1)) >
        0.1f * extent;
    prune_mask = torch::logical_or(torch::logical_or(prune_mask, big_points_vs),
                                   big_points_ws);
  }
  this->prunePoints(prune_mask);
  if(explicit_anchors_) { xyz_gradient_accum_.zero_();denom_.zero_();max_radii2D_.zero_(); }

  c10::cuda::CUDACachingAllocator::emptyCache();  // torch.cuda.empty_cache()
}

void GaussianModel::addDensificationStats(torch::Tensor& viewspace_point_tensor,
                                          torch::Tensor& update_filter) {
  if(variableAnchorMembers()) {
    anchor_growth_gradient_accum_.index_put_({update_filter},
        torch::frobenius_norm(viewspace_point_tensor.grad().index({update_filter,torch::indexing::Slice(0,2)}),-1,true),true);
    anchor_growth_denom_.index_put_({update_filter},anchor_growth_denom_.index({update_filter})+1.0);
  }
  this->xyz_gradient_accum_.index_put_(
      {update_filter},
      torch::frobenius_norm(viewspace_point_tensor.grad().index(
                                {update_filter, torch::indexing::Slice(0, 2)}),
                            /*dim=*/-1,
                            /*keepdim=*/true),
      /*accumulate=*/true);

  this->denom_.index_put_({update_filter},
                          this->denom_.index({update_filter}) + 1);
}

// void GaussianModel::increasePointsIterationsOfExistence(const int i)
// {
//     this->exist_since_iter_ += i;
// }

void GaussianModel::loadPly(std::filesystem::path ply_path) {
  std::ifstream instream_binary(ply_path, std::ios::binary);
  if (!instream_binary.is_open() || instream_binary.fail())
    throw std::runtime_error("Fail to open ply file at " + ply_path.string());
  instream_binary.seekg(0, std::ios::beg);

  tinyply::PlyFile ply_file;
  ply_file.parse_header(instream_binary);

  std::cout << "\t[ply_header] Type: "
            << (ply_file.is_binary_file() ? "binary" : "ascii") << std::endl;
  for (const auto& c : ply_file.get_comments())
    std::cout << "\t[ply_header] Comment: " << c << std::endl;
  for (const auto& c : ply_file.get_info())
    std::cout << "\t[ply_header] Info: " << c << std::endl;

  for (const auto& e : ply_file.get_elements()) {
    std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")"
              << std::endl;
    for (const auto& p : e.properties) {
      std::cout << "\t[ply_header] \tproperty: " << p.name
                << " (type=" << tinyply::PropertyTable[p.propertyType].str
                << ")";
      if (p.isList)
        std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str
                  << ")";
      std::cout << std::endl;
    }
  }

  std::shared_ptr<tinyply::PlyData> xyz, f_dc, f_rest, opacity, scales, rot;

  try {
    xyz = ply_file.request_properties_from_element("vertex", {"x", "y", "z"});
  } catch (const std::exception& e) {
    std::cerr << "tinyply exception: " << e.what() << std::endl;
  }

  try {
    f_dc = ply_file.request_properties_from_element(
        "vertex", {"f_dc_0", "f_dc_1", "f_dc_2"});
  } catch (const std::exception& e) {
    std::cerr << "tinyply exception: " << e.what() << std::endl;
  }

  int n_f_rest = ((max_sh_degree_ + 1) * (max_sh_degree_ + 1) - 1) * 3;
  if (n_f_rest >= 0) {
    std::vector<std::string> f_rest_element_names(n_f_rest);
    for (int i = 0; i < n_f_rest; ++i)
      f_rest_element_names[i] = "f_rest_" + std::to_string(i);
    try {
      f_rest = ply_file.request_properties_from_element("vertex",
                                                        f_rest_element_names);
    } catch (const std::exception& e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }
  }

  try {
    opacity = ply_file.request_properties_from_element("vertex", {"opacity"});
  } catch (const std::exception& e) {
    std::cerr << "tinyply exception: " << e.what() << std::endl;
  }

  try {
    scales = ply_file.request_properties_from_element(
        "vertex", {"scale_0", "scale_1", "scale_2"});
  } catch (const std::exception& e) {
    std::cerr << "tinyply exception: " << e.what() << std::endl;
  }

  try {
    rot = ply_file.request_properties_from_element(
        "vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});
  } catch (const std::exception& e) {
    std::cerr << "tinyply exception: " << e.what() << std::endl;
  }

  ply_file.read(instream_binary);

  if (xyz) std::cout << "\tRead " << xyz->count << " total xyz " << std::endl;
  if (f_dc)
    std::cout << "\tRead " << f_dc->count << " total f_dc " << std::endl;
  if (f_rest)
    std::cout << "\tRead " << f_rest->count << " total f_rest " << std::endl;
  if (opacity)
    std::cout << "\tRead " << opacity->count << " total opacity " << std::endl;
  if (scales)
    std::cout << "\tRead " << scales->count << " total scales " << std::endl;
  if (rot) std::cout << "\tRead " << rot->count << " total rot " << std::endl;

  // Data to std::vector
  const int num_points = xyz->count;

  const std::size_t n_xyz_bytes = xyz->buffer.size_bytes();
  std::vector<float> xyz_vector(xyz->count * 3);
  std::memcpy(xyz_vector.data(), xyz->buffer.get(), n_xyz_bytes);

  const std::size_t n_f_dc_bytes = f_dc->buffer.size_bytes();
  std::vector<float> f_dc_vector(f_dc->count * 3);
  std::memcpy(f_dc_vector.data(), f_dc->buffer.get(), n_f_dc_bytes);

  const std::size_t n_f_rest_bytes = f_rest->buffer.size_bytes();
  std::vector<float> f_rest_vector(f_rest->count * n_f_rest);
  std::memcpy(f_rest_vector.data(), f_rest->buffer.get(), n_f_rest_bytes);

  const std::size_t n_opacity_bytes = opacity->buffer.size_bytes();
  std::vector<float> opacity_vector(opacity->count * 1);
  std::memcpy(opacity_vector.data(), opacity->buffer.get(), n_opacity_bytes);

  const std::size_t n_scales_bytes = scales->buffer.size_bytes();
  std::vector<float> scales_vector(scales->count * 3);
  std::memcpy(scales_vector.data(), scales->buffer.get(), n_scales_bytes);

  const std::size_t n_rot_bytes = rot->buffer.size_bytes();
  std::vector<float> rot_vector(rot->count * 4);
  std::memcpy(rot_vector.data(), rot->buffer.get(), n_rot_bytes);

  // std::vector to torch::Tensor
  this->xyz_ = torch::from_blob(xyz_vector.data(), {num_points, 3},
                                torch::TensorOptions().dtype(torch::kFloat32))
                   .to(device_type_);

  this->features_dc_ =
      torch::from_blob(f_dc_vector.data(), {num_points, 3, 1},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_)
          .transpose(1, 2)
          .contiguous();

  this->features_rest_ =
      torch::from_blob(f_rest_vector.data(), {num_points, 3, n_f_rest / 3},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_)
          .transpose(1, 2)
          .contiguous();

  this->opacity_ =
      torch::from_blob(opacity_vector.data(), {num_points, 1},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_);

  this->scaling_ =
      torch::from_blob(scales_vector.data(), {num_points, 3},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_);

  this->rotation_ =
      torch::from_blob(rot_vector.data(), {num_points, 4},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(device_type_);

  protection_.reset(num_points, torch::TensorOptions().device(device_type_));
  this->last_elastic_supervised_iter_ = torch::full({xyz_.size(0)}, protection_.mapping_iteration, xyz_.options().dtype(torch::kInt64).requires_grad(false));

  GAUSSIAN_MODEL_TENSORS_TO_VEC

  this->active_sh_degree_ = this->max_sh_degree_;
  protection_.assertAligned(this->xyz_.size(0));
  assert(this->xyz_.size(0) == this->last_elastic_supervised_iter_.size(0));
  if (online_gi_reset_callback_)
    online_gi_reset_callback_(this->xyz_.size(0));
}

void GaussianModel::loadSelectorMetadataPly(std::filesystem::path ply_path) {
  std::ifstream input(ply_path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open maturity metadata: " + ply_path.string());
  tinyply::PlyFile file;
  file.parse_header(input);
  std::string schema;
  int64_t mapping_iteration = -1;
  for (const auto& comment : file.get_comments()) {
    std::istringstream stream(comment);
    std::string key; stream >> key;
    if (key == "protection_schema") stream >> schema;
    if (key == "protection_mapping_iteration") stream >> mapping_iteration;
  }
  TORCH_CHECK(schema == "continuous_cycle_geometric" && mapping_iteration >= 0,
      "Checkpoint requires continuous_cycle_geometric maturity metadata");
  std::map<std::string, std::shared_ptr<tinyply::PlyData>> fields;
  for (const auto* name : {"maturity", "drift_ema", "cached_uniform"})
    fields[name] = file.request_properties_from_element("vertex", {name});
  file.read(input);
  const int64_t count = xyz_.size(0);
  protection_.reset(count, xyz_.options());
  protection_.mapping_iteration = mapping_iteration;
  for (auto entry : {std::make_pair("maturity", &protection_.maturity),
                     std::make_pair("drift_ema", &protection_.drift_ema),
                     std::make_pair("cached_uniform", &protection_.cached_uniform)}) {
    const auto& field = fields.at(entry.first);
    TORCH_CHECK(field->count == count && field->t == tinyply::Type::FLOAT32,
        "Maturity metadata count/dtype differs from Gaussian geometry");
    *entry.second = torch::from_blob(field->buffer.get(), {count}, torch::kFloat32).clone().to(device_type_);
    TORCH_CHECK(torch::isfinite(*entry.second).all().item<bool>(), "Non-finite maturity metadata");
  }
  TORCH_CHECK(((protection_.maturity >= 0) & (protection_.maturity <= 1)).all().item<bool>() &&
      (protection_.drift_ema >= 0).all().item<bool>() &&
      ((protection_.cached_uniform >= 0) & (protection_.cached_uniform < 1)).all().item<bool>(),
      "Maturity checkpoint values outside their domains");
  // Restored EMA is evidence; runtime snapshots/touched state start invalid.
  protection_.drift_initialized.copy_((protection_.maturity > 0) | (protection_.drift_ema > 0));
  if (protection_.config.fixed_maturity_one) {
    protection_.maturity.fill_(1);
    protection_.drift_ema.zero_();
    protection_.drift_initialized.zero_();
    protection_.first_valid_maturity_cycle_iteration = protection_.mapping_iteration;
  }
  protection_.assertAligned(count);
  if(protection_.config.reliability_v2)
    protection_.loadReliability((ply_path.parent_path()/"reliability_state.pt").string());
}

void GaussianModel::saveSelectorMetadataPly(std::filesystem::path result_path) {
  protection_.assertAligned(xyz_.size(0));
  const int64_t count = xyz_.size(0);
  tinyply::PlyFile file;
  file.get_comments().push_back("protection_schema continuous_cycle_geometric");
  file.get_comments().push_back(std::string("maturity_mode ") +
      (protection_.config.fixed_maturity_one ? "fixed_one" : "continuous_cycle_geometric"));
  file.get_comments().push_back("protection_mapping_iteration " + std::to_string(protection_.mapping_iteration));
  std::map<std::string, torch::Tensor> tensors;
  tensors["maturity"] = protection_.maturity.cpu().contiguous();
  tensors["drift_ema"] = protection_.drift_ema.cpu().contiguous();
  tensors["cached_uniform"] = protection_.cached_uniform.cpu().contiguous();
  for (const auto& entry : tensors)
    file.add_properties_to_element("vertex", {entry.first}, tinyply::Type::FLOAT32, count,
        reinterpret_cast<const uint8_t*>(entry.second.data_ptr()), tinyply::Type::INVALID, 0);
  std::ofstream output(result_path, std::ios::binary);
  if (!output) throw std::runtime_error("Cannot write maturity metadata: " + result_path.string());
  file.write(output, true);
  protection_.saveReliability((result_path.parent_path()/"reliability_state.pt").string());
}

void GaussianModel::savePly(std::filesystem::path result_path) {
  // Prepare data to write
  torch::Tensor xyz = this->xyz_.detach().cpu();
  torch::Tensor normals = torch::zeros_like(xyz);
  torch::Tensor f_dc =
      this->features_dc_.detach().transpose(1, 2).flatten(1).contiguous().cpu();
  torch::Tensor f_rest = this->features_rest_.detach()
                             .transpose(1, 2)
                             .flatten(1)
                             .contiguous()
                             .cpu();
  torch::Tensor opacities = this->opacity_.detach().cpu();
  torch::Tensor scale = this->scaling_.detach().cpu();
  torch::Tensor rotation = this->rotation_.detach().cpu();

  std::filebuf fb_binary;
  fb_binary.open(result_path, std::ios::out | std::ios::binary);
  std::ostream outstream_binary(&fb_binary);
  if (outstream_binary.fail())
    throw std::runtime_error("failed to open " + result_path.string());

  tinyply::PlyFile result_file;

  // xyz
  result_file.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, xyz.size(0),
      reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()), tinyply::Type::INVALID,
      0);

  // normals
  result_file.add_properties_to_element(
      "vertex", {"nx", "ny", "nz"}, tinyply::Type::FLOAT32, normals.size(0),
      reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // f_dc
  std::size_t n_f_dc = this->features_dc_.size(1) * this->features_dc_.size(2);
  std::vector<std::string> property_names_f_dc(n_f_dc);
  for (int i = 0; i < n_f_dc; ++i)
    property_names_f_dc[i] = "f_dc_" + std::to_string(i);

  result_file.add_properties_to_element(
      "vertex", property_names_f_dc, tinyply::Type::FLOAT32,
      this->features_dc_.size(0),
      reinterpret_cast<uint8_t*>(f_dc.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // f_rest
  std::size_t n_f_rest =
      this->features_rest_.size(1) * this->features_rest_.size(2);
  std::vector<std::string> property_names_f_rest(n_f_rest);
  for (int i = 0; i < n_f_rest; ++i)
    property_names_f_rest[i] = "f_rest_" + std::to_string(i);

  result_file.add_properties_to_element(
      "vertex", property_names_f_rest, tinyply::Type::FLOAT32,
      this->features_rest_.size(0),
      reinterpret_cast<uint8_t*>(f_rest.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // opacities
  result_file.add_properties_to_element(
      "vertex", {"opacity"}, tinyply::Type::FLOAT32, opacities.size(0),
      reinterpret_cast<uint8_t*>(opacities.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // scale
  std::size_t n_scale = scale.size(1);
  std::vector<std::string> property_names_scale(n_scale);
  for (int i = 0; i < n_scale; ++i)
    property_names_scale[i] = "scale_" + std::to_string(i);

  result_file.add_properties_to_element(
      "vertex", property_names_scale, tinyply::Type::FLOAT32, scale.size(0),
      reinterpret_cast<uint8_t*>(scale.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // rotation
  std::size_t n_rotation = rotation.size(1);
  std::vector<std::string> property_names_rotation(n_rotation);
  for (int i = 0; i < n_rotation; ++i)
    property_names_rotation[i] = "rot_" + std::to_string(i);

  result_file.add_properties_to_element(
      "vertex", property_names_rotation, tinyply::Type::FLOAT32,
      rotation.size(0), reinterpret_cast<uint8_t*>(rotation.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // Write the file
  result_file.write(outstream_binary, true);

  fb_binary.close();
}

void GaussianModel::saveSparsePointsPly(std::filesystem::path result_path) {
  // Prepare data to write
  torch::Tensor xyz = this->sparse_points_xyz_.detach().cpu();
  torch::Tensor normals = torch::zeros_like(xyz);
  torch::Tensor color = (this->sparse_points_color_ * 255.0f)
                            .toType(torch::kUInt8)
                            .detach()
                            .cpu();

  std::filebuf fb_binary;
  fb_binary.open(result_path, std::ios::out | std::ios::binary);
  std::ostream outstream_binary(&fb_binary);
  if (outstream_binary.fail())
    throw std::runtime_error("failed to open " + result_path.string());

  tinyply::PlyFile result_file;

  // xyz
  result_file.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, xyz.size(0),
      reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()), tinyply::Type::INVALID,
      0);

  // normals
  result_file.add_properties_to_element(
      "vertex", {"nx", "ny", "nz"}, tinyply::Type::FLOAT32, normals.size(0),
      reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
      tinyply::Type::INVALID, 0);

  // color
  result_file.add_properties_to_element(
      "vertex", {"red", "green", "blue"}, tinyply::Type::UINT8, color.size(0),
      reinterpret_cast<uint8_t*>(color.data_ptr<uint8_t>()),
      tinyply::Type::INVALID, 0);

  // Write the file
  result_file.write(outstream_binary, true);

  fb_binary.close();
}

float GaussianModel::percentDense() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return percent_dense_;
}

void GaussianModel::setPercentDense(const float percent_dense) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  percent_dense_ = percent_dense;
}

/**
 * @brief get_expon_lr_func
 * @details Modified from Plenoxels
 *  Continuous learning rate decay function. Adapted from JaxNeRF
 *  The returned rate is lr_init when step=0 and lr_final when step=max_steps,
 * and is log-linearly interpolated elsewhere (equivalent to exponential decay).
 *  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
 *  function of lr_delay_mult, such that the initial learning rate is
 *  lr_init*lr_delay_mult at the beginning of optimization but will be eased
 * back to the normal learning rate when steps>lr_delay_steps. :param conf:
 * config subtree 'lr' or similar :param max_steps: int, the number of steps
 * during optimization. :return HoF which takes step as input
 * @param iteration
 * @return float
 */
float GaussianModel::exponLrFunc(int step) {
  if (step < 0 || (lr_init_ == 0.0f && lr_final_ == 0.0f)) return 0.0f;

  float delay_rate;
  if (lr_delay_steps_ > 0)
    delay_rate = lr_delay_mult_ +
                 (1.0f - lr_delay_mult_) *
                     std::sin(M_PI_2f32 * std::clamp(static_cast<float>(step) /
                                                         lr_delay_steps_,
                                                     0.0f, 1.0f));
  else
    delay_rate = 1.0f;
  float t = std::clamp(static_cast<float>(step) / max_steps_, 0.0f, 1.0f);
  float log_lerp =
      std::exp(std::log(lr_init_) * (1 - t) + std::log(lr_final_) * t);
  return delay_rate * log_lerp;
}

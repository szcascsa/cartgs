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
 * as part of Photo-SLAM.
 */

#pragma once

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>
#include "../improvements/selector/gaussian_compensation_bank.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"
#include "gaussian_parameters.h"
#include "gaussian_protection.h"
#include "gaussian_transform_ops.h"
#include "general_utils.h"
#include "operate_points.h"
#include "point3d.h"
#include "sh_utils.h"
#include "tensor_utils.h"
#include "third_party/simple-knn/spatial.h"
#include "third_party/tinyply/tinyply.h"
#include "types.h"
#include "improvements/selector/anchor_bank.h"

#define GAUSSIAN_MODEL_TENSORS_TO_VEC                      \
  this->Tensor_vec_xyz_ = {this->xyz_};                    \
  this->Tensor_vec_feature_dc_ = {this->features_dc_};     \
  this->Tensor_vec_feature_rest_ = {this->features_rest_}; \
  this->Tensor_vec_opacity_ = {this->opacity_};            \
  this->Tensor_vec_scaling_ = {this->scaling_};            \
  this->Tensor_vec_rotation_ = {this->rotation_};

#define GAUSSIAN_MODEL_INIT_TENSORS(device_type)                              \
  this->xyz_ = torch::empty(0, torch::TensorOptions().device(device_type));   \
  this->features_dc_ =                                                        \
      torch::empty(0, torch::TensorOptions().device(device_type));            \
  this->features_rest_ =                                                      \
      torch::empty(0, torch::TensorOptions().device(device_type));            \
  this->scaling_ =                                                            \
      torch::empty(0, torch::TensorOptions().device(device_type));            \
  this->rotation_ =                                                           \
      torch::empty(0, torch::TensorOptions().device(device_type));            \
  this->opacity_ =                                                            \
      torch::empty(0, torch::TensorOptions().device(device_type));            \
  this->max_radii2D_ =                                                        \
      torch::empty(0, torch::TensorOptions().device(device_type));            \
  this->xyz_gradient_accum_ =                                                 \
      torch::empty(0, torch::TensorOptions().device(device_type));            \
  this->denom_ = torch::empty(0, torch::TensorOptions().device(device_type)); \
  this->protection_.reset(0, torch::TensorOptions().device(device_type)); \
  this->last_elastic_supervised_iter_ = torch::empty(0, torch::TensorOptions().dtype(torch::kInt64).device(device_type)); \
  GAUSSIAN_MODEL_TENSORS_TO_VEC

class GaussianModel {
 public:
  enum class AppendKind { Append, Clone, Split };
  using GIAppendEventCallback = std::function<void(std::int64_t, AppendKind)>;
  using OnlineGIAppendCallback = std::function<void(std::int64_t)>;
  using OnlineGIPruneCallback =
      std::function<void(const torch::Tensor& survivor_mask)>;
  using OnlineGIResetCallback = std::function<void(std::int64_t)>;

  explicit GaussianModel(const int sh_degree);
  explicit GaussianModel(const GaussianModelParams& model_params);

  torch::Tensor getScalingActivation();
  torch::Tensor getRotationActivation();
  torch::Tensor getXYZ();
  torch::Tensor getFeatures();
  torch::Tensor getOpacityActivation();
  torch::Tensor getScalingRotationActivation(int scaling_modifier = 1);
  torch::Tensor getCovarianceActivation(int scaling_modifier = 1);

  void oneUpShDegree();
  void setShDegree(const int sh);
  void enableExplicitAnchors(improvements::selector::AnchorBank bank, torch::optim::Adam* feature_optimizer,
                            std::filesystem::path pruning_diagnostics = {});
  bool explicitAnchors() const { return static_cast<bool>(explicit_anchors_); }
  void enablePrivateCompensation(improvements::selector::GaussianCompensationBank bank) {
    TORCH_CHECK(variableAnchorMembers() && xyz_.numel()==0,"Private compensation must precede variable map initialization");
    private_compensation_=bank;
  }
  double anchorGrowthThreshold() const { return anchor_growth_threshold_; }
  bool variableAnchorMembers() const { return explicit_anchors_ && explicit_anchors_->variableMembers(); }
  bool fixedTenAnchors() const { return explicit_anchors_ && explicit_anchors_->fixedTenMembers(); }
  void setAnchorTopologyPolicy(int anchor_only_pruning, bool refine_requires_full) {
    TORCH_CHECK(anchor_only_pruning==0 || anchor_only_pruning==1 || anchor_only_pruning==3 || anchor_only_pruning==4,
                "Selector.anchor_only_pruning must be 0, 1, 3 or 4");
    TORCH_CHECK(variableAnchorMembers() || (!anchor_only_pruning && !refine_requires_full),
                "Anchor topology policy requires variable members");
    if(anchor_only_pruning>=3) {
      TORCH_CHECK(!xyz_.numel() || (fixedTenAnchors() && explicit_anchors_->fixedTenMode()==anchor_only_pruning),
                  "Fixed-ten mode must be selected before initializing the map");
      explicit_anchors_->enableFixedTenMembers(anchor_only_pruning);
    } else TORCH_CHECK(!fixedTenAnchors(),"Cannot change a fixed-ten map to legacy topology");
    anchor_only_pruning_=anchor_only_pruning;anchor_refine_requires_full_=refine_requires_full;
  }
  uint64_t gaussianPrunedMembers() const { return gaussian_pruned_members_; }
  uint64_t anchorPrunedMembers() const { return anchor_pruned_members_; }
  uint64_t localAnchorMigratedMembers() const { return local_anchor_migrated_members_; }
  uint64_t localAnchorDroppedMembers() const { return local_anchor_dropped_members_; }
  void finishAnchorGeometry();
  void migrateLocalAnchors();
  void addAnchorPruningStats(const torch::Tensor& visible_indices);
  torch::Tensor anchorPruningMask(int check_interval);
  const torch::Tensor& anchorOpacityAccum() const { return anchor_opacity_accum_; }
  const torch::Tensor& anchorVisits() const { return anchor_visits_; }
  const torch::Tensor& anchorSlotOpacityAccum() const { return anchor_slot_opacity_accum_; }
  const torch::Tensor& anchorSlotBelowPrune() const { return anchor_slot_below_prune_; }
  const torch::Tensor& anchorSlotBelowCutoff() const { return anchor_slot_below_cutoff_; }
  void saveAnchorPruningState(std::filesystem::path path, int check_interval) const;

  void createFromPcd(std::map<point3D_id_t, Point3D> pcd,
                     const float spatial_lr_scale);

  void increasePcd(std::vector<float> points,
                   std::vector<float> colors,
                   const int iteration);
  void increasePcd(torch::Tensor& new_point_cloud,
                   torch::Tensor& new_colors,
                   const int iteration);

  GaussianTransformResult applyScaledTransformation(
      const float s = 1.0,
      const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(),
                                          Eigen::Vector3f::Zero()));
  GaussianTransformResult scaledTransformVisiblePointsOfKeyframe(
      torch::Tensor& point_not_transformed_flags,
      const Sophus::SE3f& diff_pose,
      torch::Tensor& kf_world_view_transform,
      torch::Tensor& kf_full_proj_transform,
      const int kf_creation_iter,
      const int stable_num_iter_existence,
      int& num_transformed,
      const float scale = 1.0f);

  void trainingSetup(const GaussianOptimizationParams& training_args);
  float updateLearningRate(int step);
  void setPositionLearningRate(float position_lr);
  void setFeatureLearningRate(float feature_lr);
  void setOpacityLearningRate(float opacity_lr);
  void setScalingLearningRate(float scaling_lr);
  void setRotationLearningRate(float rot_lr);

  void resetOpacity();
  torch::Tensor replaceTensorToOptimizer(torch::Tensor& t, int tensor_idx);

  void prunePoints(torch::Tensor& mask);

  torch::Tensor getSelectorMatureMask() const;
  void updateElasticSupervision(const torch::Tensor& indices,
                                std::int64_t mapping_iter);

  void densificationPostfix(torch::Tensor& new_xyz,
                            torch::Tensor& new_features_dc,
                            torch::Tensor& new_features_rest,
                            torch::Tensor& new_opacities,
                            torch::Tensor& new_scaling,
                            torch::Tensor& new_rotation,
                            torch::Tensor& new_exist_since_iter,
                            AppendKind append_kind = AppendKind::Append,
                            const torch::Tensor& parent_indices = {});

  void densifyAndSplit(torch::Tensor& grads,
                       float grad_threshold,
                       float scene_extent,
                       int N);

  void densifyAndClone(torch::Tensor& grads,
                       float grad_threshold,
                       float scene_extent);

  // Clone/split inherit convergence state; native existence remains inherited.
  void densifyAndPrune(float max_grad,
                       float min_opacity,
                       float extent,
                       int max_screen_size,
                       int anchor_check_interval = 500);

  void addDensificationStats(torch::Tensor& viewspace_point_tensor,
                             torch::Tensor& update_filter);

  // void increasePointsIterationsOfExistence(const int i = 1);

  void loadPly(std::filesystem::path ply_path);
  void savePly(std::filesystem::path result_path);
  void loadSelectorMetadataPly(std::filesystem::path ply_path);
  void saveSelectorMetadataPly(std::filesystem::path result_path);
  void saveSparsePointsPly(std::filesystem::path result_path);

  void setOnlineGIStateCallbacks(OnlineGIAppendCallback on_append,
                                 OnlineGIPruneCallback on_prune,
                                 OnlineGIResetCallback on_reset,
                                 GIAppendEventCallback on_append_event = {});

  float percentDense();
  void setPercentDense(const float percent_dense);

 protected:
  float exponLrFunc(int step);

 public:
  torch::DeviceType device_type_;

  int active_sh_degree_;
  int max_sh_degree_;

  torch::Tensor xyz_;
  torch::Tensor features_dc_;
  torch::Tensor features_rest_;
  torch::Tensor scaling_;
  torch::Tensor rotation_;
  torch::Tensor opacity_;
  torch::Tensor max_radii2D_;
  torch::Tensor xyz_gradient_accum_;
  torch::Tensor denom_;
  torch::Tensor exist_since_iter_;
  GaussianProtection protection_;
  torch::Tensor last_elastic_supervised_iter_;

  std::vector<torch::Tensor> Tensor_vec_xyz_, Tensor_vec_feature_dc_,
      Tensor_vec_feature_rest_, Tensor_vec_opacity_, Tensor_vec_scaling_,
      Tensor_vec_rotation_;

  std::shared_ptr<torch::optim::Adam> optimizer_;
  float percent_dense_;
  float spatial_lr_scale_;

  torch::Tensor sparse_points_xyz_;
  torch::Tensor sparse_points_color_;

 protected:
  float lr_init_;
  float lr_final_;
  int lr_delay_steps_;
  float lr_delay_mult_;
  int max_steps_;

  std::mutex mutex_settings_;

  OnlineGIAppendCallback online_gi_append_callback_;
  OnlineGIPruneCallback online_gi_prune_callback_;
  OnlineGIResetCallback online_gi_reset_callback_;
  GIAppendEventCallback gi_append_event_callback_;
  improvements::selector::AnchorBank explicit_anchors_{nullptr};
  improvements::selector::GaussianCompensationBank private_compensation_{nullptr};
  torch::optim::Adam* anchor_feature_optimizer_ = nullptr;
  double anchor_growth_threshold_ = 0.0;
  int anchor_only_pruning_ = 0;
  bool anchor_refine_requires_full_ = false;
  uint64_t gaussian_pruned_members_ = 0, anchor_pruned_members_ = 0;
  uint64_t local_anchor_migrated_members_ = 0, local_anchor_dropped_members_ = 0;
  torch::Tensor local_anchor_moved_rows_;
  torch::Tensor anchor_opacity_accum_, anchor_visits_;
  torch::Tensor anchor_slot_opacity_accum_, anchor_slot_below_prune_, anchor_slot_below_cutoff_;
  torch::Tensor anchor_growth_gradient_accum_, anchor_growth_denom_;
  std::filesystem::path anchor_pruning_diagnostics_;
  void appendAnchorSeeds(const improvements::selector::AnchorBankImpl::BlockSeeds& seeds, int iteration);
  void resetAnchorWindows(const torch::Tensor& changed);
  void densifyVariableMembers(const torch::Tensor& gradients, float threshold, float extent);
};

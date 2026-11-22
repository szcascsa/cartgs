#include "cuda_rasterizer/anchor_si_capture.h"
#include "cuda_rasterizer/rendering_contribution.h"
/**
 * This file is part of CaRtGS, modified from Photo-SLAM under GPL v3 license.
 *
 * Copyright (C) 2024 Dapeng Feng, Sun Yat-sen University.
 *
 * CaRtGS is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * CaRtGS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * CaRtGS. If not, see <http://www.gnu.org/licenses/>.
 */

#include "include/gaussian_mapper.h"
#include "include/reliability_maturity.h"
#include "include/anchor_frustum.h"
#include "include/flexgs_importance.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <torch/nn/functional/upsampling.h>

#include "include/gaussian_renderer.h"
#include "include/mapping_cuda_optimizer.h"
#include "include/anchor_selector_cuda.h"
#include "include/selector_sampling_cuda.h"
#include "cuda_rasterizer/render_workspace.h"
#include "cuda_rasterizer/ssim_backend.h"
#include "include/loss_utils.h"

void trainingReport(int iteration,
                    int num_iterations,
                    torch::Tensor& Ll1,
                    torch::Tensor& loss,
                    float ema_loss_for_log,
                    int64_t elapsed_time,
                    GaussianModel& gaussians,
                    GaussianScene& scene,
                    GaussianPipelineParams& pipe,
                    torch::Tensor& background) {
  std::ostringstream line;
  line << std::fixed << std::setprecision(4) << "Training iteration "
       << iteration << "/" << num_iterations
       << ", time elapsed:" << elapsed_time / 1000.0 << "s"
       << ", ema_loss:" << ema_loss_for_log
       << ", num_points:" << gaussians.xyz_.size(0);
  std::cout << line.str() << std::endl;
}

GaussianMapper::GaussianMapper(std::shared_ptr<ORB_SLAM3::System> pSLAM,
                               std::filesystem::path gaussian_config_file_path,
                               std::filesystem::path result_dir,
                               int seed,
                               torch::DeviceType device_type,
                               std::optional<bool> selector_enabled_override,
                               std::filesystem::path selector_config_file_path)
    : pSLAM_(pSLAM),
      initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      SLAM_ended_(false),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      large_rot_th_(1e-1f),
      large_trans_th_(1e-2f),
      training_report_interval_(0) {
  // Random seed
  initialization_seed_ = seed;
  std::srand(seed);
  torch::manual_seed(seed);

  // Device
  if (device_type == torch::kCUDA && torch::cuda::is_available()) {
    std::cout << "[Gaussian Mapper]CUDA available! Training on GPU."
              << std::endl;
    device_type_ = torch::kCUDA;
    model_params_.data_device_ = "cuda";
  } else {
    std::cout << "[Gaussian Mapper]Training on CPU." << std::endl;
    device_type_ = torch::kCPU;
    model_params_.data_device_ = "cpu";
  }

  result_dir_ = result_dir;
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
  config_file_path_ = gaussian_config_file_path;
  selector_config_file_path_ = selector_config_file_path;
  readConfigFromFile(gaussian_config_file_path);
  if (selector_config_file_path_.empty()) {
    throw std::invalid_argument(
        "GaussianMapper requires --selector-config for experiment parameters");
  }
  readSelectorConfigFromFile(selector_config_file_path_);
  if (selector_enabled_override.has_value()) {
    selector_enabled_ = *selector_enabled_override;
  }
  if (ultrafast_rotation_.enabled && !selector_enabled_)
    throw std::invalid_argument("Ultrafast requires Selector.enabled: 1");
  if(selector_explicit_anchor_blocks_ && !selector_enabled_)
    throw std::invalid_argument("Explicit Anchor map requires its Anchor Selector");
  if (ultra_light_mode_ == 2 && gradient_constraint_config_.enabled)
    throw std::invalid_argument("Mode2 requires GradientConstraint.enable: 0; auxiliary stage Guard is not supported");
  if (ultra_light_mode_ == 2)
    std::cout << "[Mode2 CONFIG] full_resolution=gaus_pyramid full_order=P normal_gaussian_detach=1"
              << " lod_render_scale=" << selector_lod_render_scale_
              << " lod_start_iteration=" << lod_start_iteration_
              << " physical_scale_additive=" << selector_physical_scale_additive_
              << " softplus_scale=" << selector_softplus_scale_
              << " selector_enabled=" << selector_enabled_ << " auxiliary_guard=unsupported\n";

  std::vector<float> bg_color;
  if (model_params_.white_background_)
    bg_color = {1.0f, 1.0f, 1.0f};
  else
    bg_color = {0.0f, 0.0f, 0.0f};
  background_ = torch::tensor(
      bg_color,
      torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

  override_color_ =
      torch::empty(0, torch::TensorOptions().device(device_type_));

  // Initialize scene and model
  gaussians_ = std::make_shared<GaussianModel>(model_params_);
  gaussians_->protection_.config = protection_config_;
  gaussians_->protection_.reset(0, gaussians_->getXYZ().options());
  scene_ = std::make_shared<GaussianScene>(model_params_);
  recent_window_importance_ =
      std::make_unique<RecentWindowImportanceManager>(
          device_type_, recent_window_config_);
  if (global_gi_config_.streaming() && pSLAM_)
    global_gi_ = std::make_unique<GlobalGIManager>(device_type_, global_gi_config_);
  if (global_gi_ && (global_gi_config_.mode == "streaming_marginal_geo" ||
      global_gi_config_.marginal_shadow || global_gi_config_.marginal_diagnostics)) {
    geometry_marginal_ = std::make_unique<GeometryMarginalImportance>();
    geometry_marginal_->reset(gaussians_->getXYZ().size(0), gaussians_->getXYZ().options());
  }
  historical_outlier_selector_ =
      std::make_unique<HistoricalOutlierSelector>(historical_outlier_config_);
  conflict_replay_manager_ = std::make_unique<ConflictReplayManager>(conflict_replay_config_);
  gaussians_->setOnlineGIStateCallbacks(
      [this](std::int64_t count) {
        if (selector_network_ && selector_network_->anchorConditioned() && !selector_explicit_anchor_blocks_)
          selector_network_->anchors()->append(gaussians_->getXYZ().slice(0, -count),
              {gaussians_->optimizer_.get(), selector_optimizer_.get()});
        if (recent_window_importance_) recent_window_importance_->appendGaussians(count);
        ++topology_version_;
      },
      [this](const torch::Tensor& survivor_mask) {
        if (selector_network_ && selector_network_->anchorConditioned() && !selector_explicit_anchor_blocks_)
          selector_network_->anchors()->selectRows(torch::nonzero(survivor_mask).reshape({-1}),
              {gaussians_->optimizer_.get(), selector_optimizer_.get()});
        ++correspondence_version_;
        if (recent_window_importance_) recent_window_importance_->pruneGaussians(survivor_mask);
        if (global_gi_) global_gi_->pruneGaussians(survivor_mask);
        if (geometry_marginal_) geometry_marginal_->prune(survivor_mask);
        ++topology_version_;
      },
      [this](std::int64_t count) {
        if (selector_network_ && selector_network_->anchorConditioned() && !selector_explicit_anchor_blocks_)
          selector_network_->anchors()->reset(gaussians_->getXYZ(),
              {gaussians_->optimizer_.get(), selector_optimizer_.get()});
        if (recent_window_importance_) recent_window_importance_->resetForGaussianCount(count);
        if (global_gi_) global_gi_->resetForGaussianCount(count);
        if (geometry_marginal_) geometry_marginal_->reset(count, gaussians_->getXYZ().options());
        if (conflict_replay_manager_) conflict_replay_manager_->reset();
        last_guard_objectives_.clear();
        ++topology_version_;
        ++correspondence_version_;
      },
      [this](std::int64_t count, GaussianModel::AppendKind kind) {
        if (!global_gi_) return;
        const auto origin = kind == GaussianModel::AppendKind::Clone ? GITopologyKind::Clone
            : kind == GaussianModel::AppendKind::Split ? GITopologyKind::Split : GITopologyKind::Append;
        global_gi_->appendGaussians(count, origin);
        if (geometry_marginal_) geometry_marginal_->append(count);
        global_gi_->assertAligned(gaussians_->getXYZ());
      });
  if (selector_enabled_) {
    selector_network_ = improvements::selector::GumbelNetwork(
        selector_architecture_ == "anchor" ? selector_anchor_voxel_size_ : 0,
        selector_anchor_gaussian_attributes_, selector_anchor_global_xyz_, selector_explicit_anchor_blocks_,
        selector_anchor_ratio_features_ ? std::vector<double>(selector_target_ratios_.begin(),selector_target_ratios_.end()) : std::vector<double>{},
        selector_anchor_feature_dim_,selector_anchor_ratio_shared_dim_);
    selector_network_->setBackend(selector_backend_);
    if(selector_anchor_ratio_scalar_input_!="none")selector_network_->enableRatioScalarInput(
        selector_anchor_ratio_scalar_input_=="log"?1:2);
    selector_network_->setFusedInputPack(fused_anchor_input_pack_);
    selector_network_->setCachedAnchorContext(cached_anchor_context_);
    selector_network_->to(torch::Device(device_type_));
    if (selector_anchor_adaptive_) selector_network_->anchors()->enableAdaptive();
    if (selector_variable_anchor_members_) selector_network_->anchors()->enableVariableMembers();
    if (selector_slot_deactivation_) selector_network_->anchors()->enableSlotDeactivation();
    if(selector_network_->anchorConditioned())selector_network_->anchors()->setFusedFullStats(fused_anchor_full_stats_);
    if(selector_compensation_method_=="spatial_localxyz" || selector_compensation_method_=="spatial_early16" || selector_compensation_method_=="spatial_early16_direct_ratio" || (selector_compensation_method_=="spatial_physical19_ratio32"||(((selector_compensation_method_=="spatial_offset3_ratio48"||selector_compensation_method_=="spatial_offset3_residual48"||selector_compensation_method_=="spatial_offset3_bandwidth48")||selector_compensation_method_=="spatial_physical19_ratio54")||(selector_compensation_method_=="spatial_anchor128_gate"||(selector_compensation_method_=="spatial_anchor128_offset_gate"||(selector_compensation_method_=="spatial_anchor128_offset_bound2"||selector_compensation_method_=="spatial_anchor128_offset_bound2_shared_selector")))))) {
      at::globalContext().setAllowTF32CuBLAS(false);
      const bool ratio32=(selector_compensation_method_=="spatial_physical19_ratio32"||(((selector_compensation_method_=="spatial_offset3_ratio48"||selector_compensation_method_=="spatial_offset3_residual48"||selector_compensation_method_=="spatial_offset3_bandwidth48")||selector_compensation_method_=="spatial_physical19_ratio54")||(selector_compensation_method_=="spatial_anchor128_gate"||(selector_compensation_method_=="spatial_anchor128_offset_gate"||(selector_compensation_method_=="spatial_anchor128_offset_bound2"||selector_compensation_method_=="spatial_anchor128_offset_bound2_shared_selector")))));
      std::vector<double> knots;
      if(ratio32){knots.assign(selector_target_ratios_.begin(),selector_target_ratios_.end());std::sort(knots.begin(),knots.end());}
      const bool ratio48=(selector_compensation_method_=="spatial_offset3_ratio48"||selector_compensation_method_=="spatial_offset3_residual48"||selector_compensation_method_=="spatial_offset3_bandwidth48");
      const int residual_mode=selector_compensation_method_=="spatial_offset3_residual48"?1:(selector_compensation_method_=="spatial_offset3_bandwidth48"?2:0);
      const bool shared128=selector_compensation_method_=="spatial_anchor128_gate"||selector_compensation_method_=="spatial_anchor128_offset_gate"||selector_compensation_method_=="spatial_anchor128_offset_bound2"||selector_compensation_method_=="spatial_anchor128_offset_bound2_shared_selector";
      const bool offset_only=ratio48||(shared128&&selector_compensation_method_!="spatial_anchor128_gate");
      const bool bounded_xyz=selector_compensation_method_=="spatial_anchor128_offset_bound2"||selector_compensation_method_=="spatial_anchor128_offset_bound2_shared_selector";
      selector_network_->enableSpatialCompensation(selector_compensation_method_!="spatial_localxyz"&&!ratio32,selector_compensation_method_=="spatial_early16_direct_ratio",selector_compensation_fourier_bands_,knots,ratio48||selector_compensation_method_=="spatial_physical19_ratio54"||shared128,shared128,offset_only,bounded_xyz,ratio48?48:32,residual_mode);
      if(const char* variant=std::getenv("GEOMETRY_HEAD_VARIANT"))
        selector_network_->spatialCompensation()->configureGeometryHeads(std::stoi(variant),initialization_seed_);
      if(selector_geometry_head_variant_)
        selector_network_->spatialCompensation()->configureGeometryHeads(selector_geometry_head_variant_,initialization_seed_);
      if(selector_physical_scale_additive_)
        selector_network_->spatialCompensation()->enablePhysicalScaleAdditive();
      if(selector_softplus_scale_)
        selector_network_->spatialCompensation()->enableSoftplusScale();
      selector_network_->spatialCompensation()->setFused(fused_spatial_compensation_);
      selector_network_->spatialCompensation()->setRatioCodeCache(cache_ratio_code_);
      selector_network_->spatialCompensation()->setAnchorInputFusion(fused_anchor128_input_);
      selector_network_->spatialCompensation()->setResidualHeadFusion(fused_residual48_heads_);
      selector_network_->spatialCompensation()->setResidualHeadParallel(parallel_residual48_heads_);
      if(selector_compensation_method_=="spatial_anchor128_offset_bound2_shared_selector")selector_network_->enableSelectorSpatialFeatures();
    } else if(selector_compensation_)selector_network_->enableCompensation();
    if(selector_compensation_method_=="anchor_ratio_latent14")selector_network_->enableAnchorRatioCompensation(
        std::vector<double>(selector_target_ratios_.begin(),selector_target_ratios_.end()));
    else if(selector_compensation_method_=="s00r00e08d12" || selector_compensation_method_=="s00r00e50d12")selector_network_->enableAnchorRatioCompensation(
        std::vector<double>(selector_target_ratios_.begin(),selector_target_ratios_.end()),selector_compensation_method_=="s00r00e50d12"?50:8);
    else if(selector_compensation_method_!="shared_mlp" && selector_compensation_method_!="spatial_localxyz" && selector_compensation_method_!="spatial_early16" && selector_compensation_method_!="spatial_early16_direct_ratio" && (selector_compensation_method_!="spatial_physical19_ratio32"&&selector_compensation_method_!="spatial_physical19_ratio54"&&selector_compensation_method_!="spatial_offset3_ratio48"&&selector_compensation_method_!="spatial_offset3_residual48"&&selector_compensation_method_!="spatial_offset3_bandwidth48"&&selector_compensation_method_!="spatial_anchor128_gate"&&selector_compensation_method_!="spatial_anchor128_offset_gate"&&selector_compensation_method_!="spatial_anchor128_offset_bound2"&&selector_compensation_method_!="spatial_anchor128_offset_bound2_shared_selector"))selector_network_->enablePrivateCompensation(
        selector_compensation_method_=="r3_independent14",
        std::vector<double>(selector_target_ratios_.begin(),selector_target_ratios_.end()));
    if(selector_anchor_set_encoder_)selector_network_->enableSelectorSetEncoder();
    if(selector_anchor_set_statistics_)selector_network_->enableSelectorSetEncoder(3);
    if(selector_anchor_context_residual_)selector_network_->enableSelectorContextResidual();
    if(selector_view_residual_enabled_||view_budget_view_residual_)
      selector_network_->enableSelectorViewResidual(view_budget_residual_input_=="physical11"?2:1);
    if(selector_view_input_enabled_)selector_network_->enableSelectorViewInput();
    if(view_budget_conditioning_)selector_network_->enableJointViewConditioning(view_budget_conditioning_);
    torch::optim::AdamOptions selector_adam_options(selector_learning_rate_);
    selector_optimizer_.reset(new torch::optim::Adam(
        selector_network_->parameters(), selector_adam_options));
    if(selector_explicit_anchor_blocks_) {
      gaussians_->enableExplicitAnchors(selector_network_->anchors(),selector_optimizer_.get(),
          result_dir_ / "anchor_pruning");
      gaussians_->setAnchorTopologyPolicy(selector_anchor_only_pruning_,selector_anchor_refine_requires_full_);
      if(selector_network_->privateCompensation())gaussians_->enablePrivateCompensation(selector_network_->privateCompensation());
    }
  }

  dynamic_aabb_manager_ =
      std::make_unique<DynamicAABBManager>(dynamic_aabb_config_);
  change_distill_manager_ =
      std::make_unique<ChangeDistillManager>(change_distill_config_);
  change_coordinator_ = std::make_unique<ChangeCoordinator>(
      dynamic_aabb_manager_.get(), change_distill_manager_.get());
  if (drift_stability_config_.enabled)
    drift_stability_manager_ =
        std::make_unique<DriftStabilityManager>(drift_stability_config_);
  std::cout << "[Gaussian Mapper] Historical recovery DSD="
            << drift_stability_config_.enabled
            << " (DriftStability.enable; independent of Change CD)" << std::endl;

  // Mode
  if (!pSLAM) {
    // NO SLAM
    return;
  }

  // Sensors
  switch (pSLAM->getSensorType()) {
    case ORB_SLAM3::System::MONOCULAR:
    case ORB_SLAM3::System::IMU_MONOCULAR: {
      this->sensor_type_ = MONOCULAR;
    } break;
    case ORB_SLAM3::System::STEREO:
    case ORB_SLAM3::System::IMU_STEREO: {
      this->sensor_type_ = STEREO;
      this->stereo_baseline_length_ = pSLAM->getSettings()->b();
      this->stereo_cv_sgm_ = cv::cuda::createStereoSGM(
          this->stereo_min_disparity_, this->stereo_num_disparity_);
      this->stereo_Q_ = pSLAM->getSettings()->Q().clone();
      stereo_Q_.convertTo(stereo_Q_, CV_32FC3, 1.0);
    } break;
    case ORB_SLAM3::System::RGBD:
    case ORB_SLAM3::System::IMU_RGBD: {
      this->sensor_type_ = RGBD;
    } break;
    default: {
      throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    } break;
  }

  // Cameras
  // TODO: not only monocular
  auto settings = pSLAM->getSettings();
  cv::Size SLAM_im_size = settings->newImSize();
  UndistortParams undistort_params(SLAM_im_size,
                                   settings->camera1DistortionCoef());

  auto vpCameras = pSLAM->getAtlas()->GetAllCameras();
  std::cout << "Num. of Camera is " << vpCameras.size() << std::endl;
  for (auto& SLAM_camera : vpCameras) {
    Camera camera;
    camera.camera_id_ = SLAM_camera->GetId();
    if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
      camera.setModelId(Camera::CameraModelType::PINHOLE);
      float SLAM_fx = SLAM_camera->getParameter(0);
      float SLAM_fy = SLAM_camera->getParameter(1);
      float SLAM_cx = SLAM_camera->getParameter(2);
      float SLAM_cy = SLAM_camera->getParameter(3);

      // Old K, i.e. K in SLAM
      cv::Mat K = (cv::Mat_<float>(3, 3) << SLAM_fx, 0.f, SLAM_cx, 0.f, SLAM_fy,
                   SLAM_cy, 0.f, 0.f, 1.f);

      // camera.width_ = this->sensor_type_ == STEREO ?
      // undistort_params.old_size_.width
      //                                              :
      //                                              graphics_utils::roundToIntegerMultipleOf16(
      //                                                    undistort_params.old_size_.width);
      camera.width_ = undistort_params.old_size_.width;
      float x_ratio =
          static_cast<float>(camera.width_) / undistort_params.old_size_.width;

      // camera.height_ = this->sensor_type_ == STEREO ?
      // undistort_params.old_size_.height
      //                                               :
      //                                               graphics_utils::roundToIntegerMultipleOf16(
      //                                                     undistort_params.old_size_.height);
      camera.height_ = undistort_params.old_size_.height;
      float y_ratio = static_cast<float>(camera.height_) /
                      undistort_params.old_size_.height;

      camera.num_gaus_pyramid_sub_levels_ = num_gaus_pyramid_sub_levels_;
      camera.gaus_pyramid_width_.resize(num_gaus_pyramid_sub_levels_);
      camera.gaus_pyramid_height_.resize(num_gaus_pyramid_sub_levels_);
      for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        camera.gaus_pyramid_width_[l] =
            camera.width_ * this->kf_gaus_pyramid_factors_[l];
        camera.gaus_pyramid_height_[l] =
            camera.height_ * this->kf_gaus_pyramid_factors_[l];
      }

      camera.params_[0] /*new fx*/ = SLAM_fx * x_ratio;
      camera.params_[1] /*new fy*/ = SLAM_fy * y_ratio;
      camera.params_[2] /*new cx*/ = SLAM_cx * x_ratio;
      camera.params_[3] /*new cy*/ = SLAM_cy * y_ratio;

      cv::Mat K_new =
          (cv::Mat_<float>(3, 3) << camera.params_[0], 0.f, camera.params_[2],
           0.f, camera.params_[1], camera.params_[3], 0.f, 0.f, 1.f);

      // Undistortion
      if (this->sensor_type_ == MONOCULAR || this->sensor_type_ == RGBD)
        undistort_params.dist_coeff_.copyTo(camera.dist_coeff_);

      camera.initUndistortRectifyMapAndMask(K, SLAM_im_size, K_new, true);

      undistort_mask_[camera.camera_id_] =
          tensor_utils::cvMat2TorchTensor_Float32(camera.undistort_mask,
                                                  device_type_);

      cv::Mat viewer_sub_undistort_mask;
      int viewer_image_height_ = camera.height_ * rendered_image_viewer_scale_;
      int viewer_image_width_ = camera.width_ * rendered_image_viewer_scale_;
      cv::resize(camera.undistort_mask, viewer_sub_undistort_mask,
                 cv::Size(viewer_image_width_, viewer_image_height_));
      viewer_sub_undistort_mask_[camera.camera_id_] =
          tensor_utils::cvMat2TorchTensor_Float32(viewer_sub_undistort_mask,
                                                  device_type_);

      cv::Mat viewer_main_undistort_mask;
      int viewer_image_height_main_ =
          camera.height_ * rendered_image_viewer_scale_main_;
      int viewer_image_width_main_ =
          camera.width_ * rendered_image_viewer_scale_main_;
      cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                 cv::Size(viewer_image_width_main_, viewer_image_height_main_));
      viewer_main_undistort_mask_[camera.camera_id_] =
          tensor_utils::cvMat2TorchTensor_Float32(viewer_main_undistort_mask,
                                                  device_type_);

      if (this->sensor_type_ == STEREO) {
        camera.stereo_bf_ = stereo_baseline_length_ * camera.params_[0];
        if (this->stereo_Q_.cols != 4) {
          this->stereo_Q_ = cv::Mat(4, 4, CV_32FC1);
          this->stereo_Q_.setTo(0.0f);
          this->stereo_Q_.at<float>(0, 0) = 1.0f;
          this->stereo_Q_.at<float>(0, 3) = -camera.params_[2];
          this->stereo_Q_.at<float>(1, 1) = 1.0f;
          this->stereo_Q_.at<float>(1, 3) = -camera.params_[3];
          this->stereo_Q_.at<float>(2, 3) = camera.params_[0];
          this->stereo_Q_.at<float>(3, 2) = 1.0f / stereo_baseline_length_;
        }
      }
    } else if (SLAM_camera->GetType() ==
               ORB_SLAM3::GeometricCamera::CAM_FISHEYE) {
      camera.setModelId(Camera::CameraModelType::FISHEYE);
    } else {
      camera.setModelId(Camera::CameraModelType::INVALID);
    }

    if (!viewer_camera_id_set_) {
      viewer_camera_id_ = camera.camera_id_;
      viewer_camera_id_set_ = true;
    }
    this->scene_->addCamera(camera);
  }
}

void GaussianMapper::readConfigFromFile(std::filesystem::path cfg_path) {
  cv::FileStorage settings_file(cfg_path.string().c_str(),
                                cv::FileStorage::READ);
  if (!settings_file.isOpened()) {
    std::cerr << "[Gaussian Mapper]Failed to open settings file at: "
              << cfg_path << std::endl;
    exit(-1);
  }

  std::cout << "[Gaussian Mapper]Reading parameters from " << cfg_path
            << std::endl;
  std::unique_lock<std::mutex> lock(mutex_settings_);

  // Model parameters
  model_params_.sh_degree_ = settings_file["Model.sh_degree"].operator int();
  model_params_.resolution_ =
      settings_file["Model.resolution"].operator float();
  model_params_.white_background_ =
      (settings_file["Model.white_background"].operator int()) != 0;
  model_params_.eval_ = (settings_file["Model.eval"].operator int()) != 0;

  // Pipeline Parameters
  z_near_ = settings_file["Camera.z_near"].operator float();
  z_far_ = settings_file["Camera.z_far"].operator float();

  monocular_inactive_geo_densify_max_pixel_dist_ =
      settings_file["Monocular.inactive_geo_densify_max_pixel_dist"]
          .operator float();
  stereo_min_disparity_ = settings_file["Stereo.min_disparity"].operator int();
  stereo_num_disparity_ = settings_file["Stereo.num_disparity"].operator int();
  RGBD_min_depth_ = settings_file["RGBD.min_depth"].operator float();
  RGBD_max_depth_ = settings_file["RGBD.max_depth"].operator float();

  inactive_geo_densify_ =
      (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
  max_depth_cached_ = settings_file["Mapper.depth_cache"].operator int();
  min_num_initial_map_kfs_ = static_cast<unsigned long>(
      settings_file["Mapper.min_num_initial_map_kfs"].operator int());
  new_keyframe_times_of_use_ =
      settings_file["Mapper.new_keyframe_times_of_use"].operator int();
  local_BA_increased_times_of_use_ =
      settings_file["Mapper.local_BA_increased_times_of_use"].operator int();
  loop_closure_increased_times_of_use_ =
      settings_file["Mapper.loop_closure_increased_times_of_use_"]
          .operator int();
  cull_keyframes_ =
      (settings_file["Mapper.cull_keyframes"].operator int()) != 0;
  large_rot_th_ =
      settings_file["Mapper.large_rotation_threshold"].operator float();
  large_trans_th_ =
      settings_file["Mapper.large_translation_threshold"].operator float();
  stable_num_iter_existence_ =
      settings_file["Mapper.stable_num_iter_existence"].operator int();

  pipe_params_.convert_SHs_ =
      (settings_file["Pipeline.convert_SHs"].operator int()) != 0;
  pipe_params_.compute_cov3D_ =
      (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;

  do_gaus_pyramid_training_ =
      (settings_file["GausPyramid.do"].operator int()) != 0;
  num_gaus_pyramid_sub_levels_ =
      settings_file["GausPyramid.num_sub_levels"].operator int();
  int sub_level_times_of_use =
      settings_file["GausPyramid.sub_level_times_of_use"].operator int();
  kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
  kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
  for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
    kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
    kf_gaus_pyramid_factors_[l] =
        std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
  }

  keyframe_record_interval_ =
      settings_file["Record.keyframe_record_interval"].operator int();
  all_keyframes_record_interval_ =
      settings_file["Record.all_keyframes_record_interval"].operator int();
  record_rendered_image_ =
      (settings_file["Record.record_rendered_image"].operator int()) != 0;
  record_ground_truth_image_ =
      (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
  record_loss_image_ =
      (settings_file["Record.record_loss_image"].operator int()) != 0;
  training_report_interval_ =
      settings_file["Record.training_report_interval"].operator int();
  record_loop_ply_ =
      (settings_file["Record.record_loop_ply"].operator int()) != 0;

  // Optimization Parameters
  opt_params_.iterations_ =
      settings_file["Optimization.max_num_iterations"].operator int();
  opt_params_.position_lr_init_ =
      settings_file["Optimization.position_lr_init"].operator float();
  opt_params_.position_lr_final_ =
      settings_file["Optimization.position_lr_final"].operator float();
  opt_params_.position_lr_delay_mult_ =
      settings_file["Optimization.position_lr_delay_mult"].operator float();
  opt_params_.position_lr_max_steps_ =
      settings_file["Optimization.position_lr_max_steps"].operator int();
  opt_params_.feature_lr_ =
      settings_file["Optimization.feature_lr"].operator float();
  opt_params_.opacity_lr_ =
      settings_file["Optimization.opacity_lr"].operator float();
  opt_params_.scaling_lr_ =
      settings_file["Optimization.scaling_lr"].operator float();
  opt_params_.rotation_lr_ =
      settings_file["Optimization.rotation_lr"].operator float();
  opt_params_.smooth_l1_ =
      (settings_file["Optimization.smooth_l1"].operator int()) != 0;
  opt_params_.opacity_reg_ =
      settings_file["Optimization.opacity_reg"].operator float();

  opt_params_.percent_dense_ =
      settings_file["Optimization.percent_dense"].operator float();
  opt_params_.lambda_dssim_ =
      settings_file["Optimization.lambda_dssim"].operator float();
  opt_params_.densification_interval_ =
      settings_file["Optimization.densification_interval"].operator int();
  opt_params_.opacity_reset_interval_ =
      settings_file["Optimization.opacity_reset_interval"].operator int();
  opt_params_.densify_from_iter_ =
      settings_file["Optimization.densify_from_iter"].operator int();
  opt_params_.densify_until_iter_ =
      settings_file["Optimization.densify_until_iter"].operator int();
  opt_params_.densify_grad_threshold_ =
      settings_file["Optimization.densify_grad_threshold"].operator float();
  opt_params_.auto_distribute_ =
      settings_file["Optimization.auto_distribute"].operator int();

  prune_big_point_after_iter_ =
      settings_file["Optimization.prune_big_point_after_iter"].operator int();
  densify_min_opacity_ =
      settings_file["Optimization.densify_min_opacity"].operator float();

  // Viewer Parameters
  rendered_image_viewer_scale_ =
      settings_file["GaussianViewer.image_scale"].operator float();
  rendered_image_viewer_scale_main_ =
      settings_file["GaussianViewer.image_scale_main"].operator float();
}

void GaussianMapper::readSelectorConfigFromFile(
    std::filesystem::path cfg_path) {
  cv::FileStorage settings_file(cfg_path.string().c_str(),
                                cv::FileStorage::READ);
  if (!settings_file.isOpened()) {
    throw std::runtime_error("[Gaussian Mapper]Failed to open selector config: " +
                             cfg_path.string());
  }

  std::cout << "[Gaussian Mapper]Reading selector experiment parameters from "
            << cfg_path << std::endl;
  std::unique_lock<std::mutex> lock(mutex_settings_);

  auto readInt = [&](const char* key, int& value) {
    const cv::FileNode node = settings_file[key];
    if (!node.empty()) value = node.operator int();
  };
  auto readFloat = [&](const char* key, float& value) {
    const cv::FileNode node = settings_file[key];
    if (!node.empty()) value = node.operator float();
  };

  int iteration_logging = 0;
  readInt("Mapping.iteration_logging", iteration_logging);
  mapping_iteration_logging_ = iteration_logging != 0;
  mapping_profiler_.enabled = mapping_iteration_logging_ || std::getenv("CARTGS_MAPPING_TRACE") != nullptr;
  std::cout << "[Mapping LOGGING] iteration_logging=" << mapping_iteration_logging_
            << " synchronized_timing=" << mapping_iteration_logging_ << std::endl;

  ultra_light_mode_ = 1;
  const auto mode_node = settings_file["Training.ultra_light_mode"];
  if (!mode_node.empty()) {
    if (!mode_node.isInt())
      throw std::invalid_argument("Training.ultra_light_mode must be an integer 0 or 1 (mode2 retired)");
    ultra_light_mode_ = mode_node.operator int();
  }
  if (ultra_light_mode_ < 0 || ultra_light_mode_ > 1)
    throw std::invalid_argument("Training.ultra_light_mode must be 0 or 1 (mode2 retired)");
  ultrafast_rotation_.enabled = ultra_light_mode_ == 1;
  int real_lod=0;readInt("Training.real_lod",real_lod);real_lod_.enabled=real_lod!=0;
  if(real_lod_.enabled)ultrafast_rotation_.enabled=false; // Full-only until its dedicated LoD phase.
  int flexgs_importance=real_lod_.enabled?1:0;readInt("FlexGSImportance.enabled",flexgs_importance);
  flexgs_importance_enabled_=flexgs_importance!=0;
  readInt("FlexGSImportance.refresh_interval",flexgs_importance_interval_);
  if(flexgs_importance_interval_<=0 || (flexgs_importance_enabled_&&!real_lod_.enabled))
    throw std::invalid_argument("FlexGSImportance requires RealLoD and a positive refresh interval");
  readInt("Training.lod_start_iteration",lod_start_iteration_);
  if(lod_start_iteration_<0)
    throw std::invalid_argument("Training.lod_start_iteration must be nonnegative");
  int serial_full_lod = 0;
  readInt("Training.serial_full_lod",serial_full_lod);
  ultrafast_rotation_.serial_full_lod = serial_full_lod != 0;
  int tail_rotation=1; readInt("Training.tail_ratio_rotation",tail_rotation);
  ultrafast_rotation_.rotate_tail=tail_rotation!=0;
  int serial_lod_detach_master = 1;
  readInt("Training.serial_lod_detach_master",serial_lod_detach_master);
  ultrafast_rotation_.serial_lod_detach_master = serial_lod_detach_master != 0;
  if (ultrafast_rotation_.serial_full_lod && !ultrafast_rotation_.enabled)
    throw std::invalid_argument("Training.serial_full_lod requires mode1");
  readInt("Training.post_topology_full_steps",ultrafast_rotation_.post_topology_full_steps);
  if ((ultrafast_rotation_.post_topology_full_steps != 0 && ultrafast_rotation_.post_topology_full_steps != 100) ||
      (ultrafast_rotation_.post_topology_full_steps && !ultrafast_rotation_.enabled))
    throw std::invalid_argument("Training.post_topology_full_steps must be 0 or 100, and requires joint mode1");
  if (ultrafast_rotation_.serial_full_lod && ultrafast_rotation_.post_topology_full_steps)
    throw std::invalid_argument("Serial Full/LoD requires post_topology_full_steps=0");
  if (ultra_light_mode_ == 2) {
    selector_lod_render_scale_ = 0.5f;
    transform_field_config_.start_iter = 3000;
  }
  std::cout << "[Mapping CONFIG] ultra_light_mode=" << ultra_light_mode_
            << " explicit_mode=" << !mode_node.empty() << std::endl;

  int force_tail_enabled = 0, tail_steps = -1;
  readInt("Training.force_tail_optimization_enabled", force_tail_enabled);
  readInt("Training.tail_optimization_steps", tail_steps);
  tail_optimization_schedule_.enabled = force_tail_enabled != 0;
  tail_optimization_schedule_.steps = tail_steps;
  tail_optimization_schedule_.validate();
  std::cout << "[TailOptimization CONFIG] enabled=" << tail_optimization_schedule_.enabled
            << " steps=" << tail_optimization_schedule_.steps << std::endl;

  int enabled = selector_enabled_ ? 1 : 0;
  readInt("Selector.enabled", enabled);
  selector_enabled_ = enabled != 0;
  const auto architecture_node = settings_file["Selector.architecture"];
  if (!architecture_node.empty()) architecture_node >> selector_architecture_;
  if (selector_architecture_ != "legacy" && selector_architecture_ != "anchor")
    throw std::invalid_argument("Selector.architecture must be legacy or anchor");
  readFloat("Selector.anchor_voxel_size", selector_anchor_voxel_size_);
  int anchor_attributes = 0;
  readInt("Selector.anchor_gaussian_attributes", anchor_attributes);
  selector_anchor_gaussian_attributes_ = anchor_attributes != 0;
  int anchor_global_xyz = 0;
  readInt("Selector.anchor_global_xyz", anchor_global_xyz);
  selector_anchor_global_xyz_ = anchor_global_xyz != 0;
  if (selector_anchor_global_xyz_ && !selector_anchor_gaussian_attributes_)
    throw std::invalid_argument("Selector.anchor_global_xyz requires Anchor Gaussian attributes");
  int anchor_adaptive = 0;
  readInt("Selector.anchor_adaptive", anchor_adaptive);
  selector_anchor_adaptive_ = anchor_adaptive != 0;
  int explicit_blocks=0;
  readInt("Selector.explicit_anchor_blocks",explicit_blocks);
  selector_explicit_anchor_blocks_=explicit_blocks!=0;
  int variable_members=0;readInt("Selector.variable_anchor_members",variable_members);
  selector_variable_anchor_members_=variable_members!=0;
  int ratio_features=0;readInt("Selector.anchor_ratio_features",ratio_features);
  selector_anchor_ratio_features_=ratio_features!=0;
  readInt("Selector.anchor_ratio_shared_dim",selector_anchor_ratio_shared_dim_);
  if(selector_anchor_ratio_shared_dim_!=8 && selector_anchor_ratio_shared_dim_!=16)
    throw std::invalid_argument("Selector.anchor_ratio_shared_dim must be 8 or 16");
  const auto scalar_input=settings_file["Selector.anchor_ratio_scalar_input"];
  if(!scalar_input.empty())scalar_input >> selector_anchor_ratio_scalar_input_;
  if(selector_anchor_ratio_scalar_input_!="none" &&
      (selector_anchor_ratio_scalar_input_!="log" && selector_anchor_ratio_scalar_input_!="raw"))
    throw std::invalid_argument("Selector.anchor_ratio_scalar_input must be none/log/raw");
  if(selector_anchor_ratio_scalar_input_!="none" && !selector_anchor_ratio_features_)
    throw std::invalid_argument("Extra ratio scalar input requires Anchor ratio features");
  readInt("Selector.anchor_feature_dim",selector_anchor_feature_dim_);
  if(selector_anchor_feature_dim_!=16 && ((selector_anchor_feature_dim_!=0 && selector_anchor_feature_dim_!=32) || selector_anchor_ratio_features_ || !selector_explicit_anchor_blocks_))
    throw std::invalid_argument("0D/32D Anchor features require the non-ratio explicit Anchor path");
  if(selector_anchor_ratio_features_ && (!selector_variable_anchor_members_ || !selector_anchor_gaussian_attributes_))
    throw std::invalid_argument("Ratio Anchor features require variable explicit Anchors with Gaussian attributes");
  int anchor_only_pruning=real_lod_.enabled?1:0,refine_requires_full=0;
  readInt("Selector.anchor_only_pruning",anchor_only_pruning);
  readInt("Selector.anchor_refine_requires_full",refine_requires_full);
  if(anchor_only_pruning!=0 && anchor_only_pruning!=1 && anchor_only_pruning!=3 && anchor_only_pruning!=4)
    throw std::invalid_argument("Selector.anchor_only_pruning must be 0, 1, 3 or 4");
  selector_anchor_only_pruning_=anchor_only_pruning;
  selector_anchor_refine_requires_full_=refine_requires_full!=0;
  if((selector_anchor_only_pruning_ || selector_anchor_refine_requires_full_) && !selector_variable_anchor_members_)
    throw std::invalid_argument("Anchor topology options require variable members");
  int compensation=0;readInt("Selector.compensation",compensation);
  int fused_adam=1;readInt("CUDA.fused_dense_adam",fused_adam);
  fused_dense_adam_=fused_adam!=0;
  int fused_decode=1,fused_pack=1;
  readInt("CUDA.fused_compensation_decode",fused_decode);
  readInt("CUDA.fused_anchor_input_pack",fused_pack);
  fused_compensation_decode_=fused_decode!=0;
  fused_anchor_input_pack_=fused_pack!=0;
  int fused_spatial=1;readInt("CUDA.fused_spatial_compensation",fused_spatial);fused_spatial_compensation_=fused_spatial!=0;
  int fused_stats=1;readInt("CUDA.fused_anchor_full_stats",fused_stats);fused_anchor_full_stats_=fused_stats!=0;
  int fused_sampling=1;readInt("CUDA.fused_selector_compaction",fused_sampling);fused_selector_compaction_=fused_sampling!=0;
  int fused_flexgs=1;readInt("CUDA.fused_flexgs_selector",fused_flexgs);fused_flexgs_selector_=fused_flexgs!=0;
  int cached_anchor_context=0;readInt("CUDA.cached_anchor_context",cached_anchor_context);cached_anchor_context_=cached_anchor_context!=0;
  int reuse_workspace=1;readInt("CUDA.reuse_renderer_workspace",reuse_workspace);reuse_renderer_workspace_=reuse_workspace!=0;
  int optimized_ssim=1;readInt("CUDA.optimized_ssim_backward",optimized_ssim);optimized_ssim_backward_=optimized_ssim!=0;
  int cached_ssim=1;readInt("CUDA.cached_ssim_statistics",cached_ssim);cached_ssim_statistics_=cached_ssim!=0;
  int cache_ratio_code=1;readInt("CUDA.cache_ratio_code",cache_ratio_code);cache_ratio_code_=cache_ratio_code!=0;
  int elide_gate=1;readInt("CUDA.elide_inactive_maturity_gate",elide_gate);elide_inactive_maturity_gate_=elide_gate!=0;
  int reuse_softmax=1;readInt("CUDA.reuse_maturity_softmax",reuse_softmax);reuse_maturity_softmax_=reuse_softmax!=0;
  int fused_anchor=1;readInt("CUDA.fused_anchor128_input",fused_anchor);fused_anchor128_input_=fused_anchor!=0;
  int fused_residual=1;readInt("CUDA.fused_residual48_input",fused_residual);fused_residual48_input_=fused_residual!=0;
  int training_residual=1;readInt("CUDA.fused_residual48_training_input",training_residual);fused_residual48_training_input_=training_residual!=0;
  int residual_heads=0;readInt("CUDA.fused_residual48_heads",residual_heads);fused_residual48_heads_=residual_heads!=0;
  int parallel_heads=0;readInt("CUDA.parallel_residual48_heads",parallel_heads);parallel_residual48_heads_=parallel_heads!=0;
  int fused_maturity=1;readInt("CUDA.fused_maturity_loss",fused_maturity);fused_maturity_loss_=fused_maturity!=0;
  int fused_l1=1;readInt("CUDA.fused_l1_loss",fused_l1);fused_l1_loss_=fused_l1!=0;
  int fused_empty_stats=1;readInt("CUDA.fused_empty_tf_statistics",fused_empty_stats);fused_empty_tf_statistics_=fused_empty_stats!=0;
  int anchor_si_enabled=0;
  readInt("AnchorSI.enabled",anchor_si_enabled);anchor_si_config_.enabled=anchor_si_enabled!=0;
  readFloat("AnchorSI.loss_weight",anchor_si_config_.weight);
  readFloat("AnchorSI.ema_decay",anchor_si_config_.decay);
  readFloat("AnchorSI.temperature",anchor_si_config_.temperature);
  readInt("AnchorSI.collect_every_full",anchor_si_config_.collect_every_full);
  readInt("AnchorSI.minimum_visits",anchor_si_config_.minimum_visits);
  readInt("AnchorSI.maximum_age_full",anchor_si_config_.maximum_age_full);
  if(anchor_si_config_.weight<0 || anchor_si_config_.decay<0 || anchor_si_config_.decay>=1 ||
      anchor_si_config_.temperature<=0 || anchor_si_config_.collect_every_full<=0 ||
      anchor_si_config_.minimum_visits<=0 || anchor_si_config_.maximum_age_full<=0)
    throw std::invalid_argument("Invalid AnchorSI configuration");
  anchor_si_.configure(anchor_si_config_);
  selector_compensation_=compensation!=0;
  const auto compensation_method_node=settings_file["Selector.compensation_method"];
  if(!compensation_method_node.empty())compensation_method_node >> selector_compensation_method_;
  if(selector_compensation_method_!="shared_mlp" && selector_compensation_method_!="r2_shared14" && selector_compensation_method_!="r3_independent14" && selector_compensation_method_!="anchor_ratio_latent14" && selector_compensation_method_!="s00r00e08d12" && selector_compensation_method_!="s00r00e50d12" && selector_compensation_method_!="spatial_localxyz" && selector_compensation_method_!="spatial_early16" && selector_compensation_method_!="spatial_early16_direct_ratio" && (selector_compensation_method_!="spatial_physical19_ratio32"&&selector_compensation_method_!="spatial_physical19_ratio54"&&selector_compensation_method_!="spatial_offset3_ratio48"&&selector_compensation_method_!="spatial_offset3_residual48"&&selector_compensation_method_!="spatial_offset3_bandwidth48"&&selector_compensation_method_!="spatial_anchor128_gate"&&selector_compensation_method_!="spatial_anchor128_offset_gate"&&selector_compensation_method_!="spatial_anchor128_offset_bound2"&&selector_compensation_method_!="spatial_anchor128_offset_bound2_shared_selector"))
    throw std::invalid_argument("Unknown Selector.compensation_method");
  if(selector_compensation_ && (selector_anchor_feature_dim_==0)!=(selector_compensation_method_=="s00r00e08d12" || selector_compensation_method_=="s00r00e50d12" || selector_compensation_method_=="spatial_localxyz" || selector_compensation_method_=="spatial_early16" || selector_compensation_method_=="spatial_early16_direct_ratio" || (selector_compensation_method_=="spatial_physical19_ratio32"||(((selector_compensation_method_=="spatial_offset3_ratio48"||selector_compensation_method_=="spatial_offset3_residual48"||selector_compensation_method_=="spatial_offset3_bandwidth48")||selector_compensation_method_=="spatial_physical19_ratio54")||(selector_compensation_method_=="spatial_anchor128_gate"||(selector_compensation_method_=="spatial_anchor128_offset_gate"||(selector_compensation_method_=="spatial_anchor128_offset_bound2"||selector_compensation_method_=="spatial_anchor128_offset_bound2_shared_selector")))))))
    throw std::invalid_argument("S00R00 requires feature_dim=0 and compensation_method=s00r00e08d12/s00r00e50d12");
  if(selector_compensation_method_!="shared_mlp" && (!selector_compensation_ || !selector_variable_anchor_members_))
    throw std::invalid_argument("Private14 compensation requires enabled variable Anchor compensation");
  if(selector_compensation_method_=="anchor_ratio_latent14" && selector_anchor_feature_dim_!=16)
    throw std::invalid_argument("Anchor ratio latent14 requires Anchor16 or shared8/ratio8");
  readInt("Selector.compensation_start_iter",selector_compensation_start_iter_);
  readInt("Selector.geometry_head_variant",selector_geometry_head_variant_);
  TORCH_CHECK(selector_geometry_head_variant_>=0&&selector_geometry_head_variant_<=2,"Invalid geometry head variant");
  int physical_scale_additive=0;
  readInt("Selector.physical_scale_additive",physical_scale_additive);
  selector_physical_scale_additive_=physical_scale_additive!=0;
  int softplus_scale=0;
  readInt("Selector.softplus_scale",softplus_scale);
  selector_softplus_scale_=softplus_scale!=0;
  if(selector_softplus_scale_ && (selector_physical_scale_additive_ || !selector_compensation_ || selector_compensation_method_.rfind("spatial_",0)!=0))
    throw std::invalid_argument("Softplus Scale requires spatial compensation and physical_scale_additive=0");
  readInt("Selector.compensation_fourier_bands",selector_compensation_fourier_bands_);
  if((selector_compensation_fourier_bands_!=0 && selector_compensation_fourier_bands_!=3 && selector_compensation_fourier_bands_!=6) ||
     (selector_compensation_fourier_bands_!=6 && selector_compensation_method_!="spatial_early16_direct_ratio" && (selector_compensation_method_!="spatial_physical19_ratio32"&&selector_compensation_method_!="spatial_physical19_ratio54"&&selector_compensation_method_!="spatial_offset3_ratio48"&&selector_compensation_method_!="spatial_offset3_residual48"&&selector_compensation_method_!="spatial_offset3_bandwidth48"&&selector_compensation_method_!="spatial_anchor128_gate"&&selector_compensation_method_!="spatial_anchor128_offset_gate"&&selector_compensation_method_!="spatial_anchor128_offset_bound2"&&selector_compensation_method_!="spatial_anchor128_offset_bound2_shared_selector")) ||
     ((selector_compensation_method_=="spatial_physical19_ratio32"||(((selector_compensation_method_=="spatial_offset3_ratio48"||selector_compensation_method_=="spatial_offset3_residual48"||selector_compensation_method_=="spatial_offset3_bandwidth48")||selector_compensation_method_=="spatial_physical19_ratio54")||(selector_compensation_method_=="spatial_anchor128_gate"||(selector_compensation_method_=="spatial_anchor128_offset_gate"||(selector_compensation_method_=="spatial_anchor128_offset_bound2"||selector_compensation_method_=="spatial_anchor128_offset_bound2_shared_selector")))))&&selector_compensation_fourier_bands_!=0))
    throw std::invalid_argument("Fourier ablation requires Early16 direct ratio and bands 0/3/6");
  if(selector_compensation_ && (!selector_explicit_anchor_blocks_ || selector_compensation_start_iter_<0))
    throw std::invalid_argument("Compensation requires explicit Anchors and a nonnegative start iteration");
  if(selector_variable_anchor_members_&&!selector_explicit_anchor_blocks_)
    throw std::invalid_argument("Variable members require explicit Anchor map");
  int slot_deactivation=0;
  readInt("Selector.slot_deactivation",slot_deactivation);
  selector_slot_deactivation_=slot_deactivation!=0;
  if(selector_anchor_only_pruning_>=3)selector_slot_deactivation_=selector_anchor_only_pruning_==4;
  int anchor_frustum_culling=selector_anchor_frustum_culling_?1:0;
  readInt("Selector.anchor_frustum_culling",anchor_frustum_culling);
  selector_anchor_frustum_culling_=anchor_frustum_culling!=0;
  int anchor_set_encoder=0;
  readInt("Selector.anchor_set_encoder",anchor_set_encoder);
  selector_anchor_set_encoder_=anchor_set_encoder!=0;
  int anchor_set_statistics=0;readInt("Selector.anchor_set_statistics",anchor_set_statistics);
  selector_anchor_set_statistics_=anchor_set_statistics!=0;
  int anchor_context_residual=0;readInt("Selector.anchor_context_residual",anchor_context_residual);
  selector_anchor_context_residual_=anchor_context_residual!=0;
  if(int(selector_anchor_set_encoder_)+int(selector_anchor_set_statistics_)+int(selector_anchor_context_residual_)>1)
    throw std::invalid_argument("Choose one Anchor Set Encoder, Set Statistics or Context Residual");
  if((selector_anchor_set_encoder_ || selector_anchor_set_statistics_ || selector_anchor_context_residual_) && (selector_architecture_!="anchor" || !selector_variable_anchor_members_ ||
      selector_anchor_feature_dim_!=0 || selector_anchor_ratio_features_ || selector_anchor_global_xyz_ ||
      selector_compensation_method_=="spatial_anchor128_offset_bound2_shared_selector"))
    throw std::invalid_argument("Anchor Set Encoder requires zero-feature Variable Anchors without other Selector Features");
  if (selector_slot_deactivation_ && !selector_explicit_anchor_blocks_)
    throw std::invalid_argument("Selector.slot_deactivation requires explicit Anchor blocks");
  if(selector_explicit_anchor_blocks_ && (selector_architecture_!="anchor" || !selector_anchor_gaussian_attributes_ ||
      selector_anchor_global_xyz_ || selector_anchor_adaptive_ || std::abs(selector_anchor_voxel_size_-.1f)>1e-7f))
    throw std::invalid_argument("Explicit Anchor map requires attribute inputs, h=0.1 and occupancy refinement disabled");
  if (selector_anchor_adaptive_ && (selector_architecture_ != "anchor" || !selector_anchor_gaussian_attributes_ ||
      std::abs(selector_anchor_voxel_size_ - 0.1f) > 1e-7f))
    throw std::invalid_argument("Adaptive Anchor requires 27-input Anchor Selector and h=0.1");
  if (selector_anchor_gaussian_attributes_ && selector_architecture_ != "anchor")
    throw std::invalid_argument("Selector.anchor_gaussian_attributes requires anchor architecture");
  if (!std::isfinite(selector_anchor_voxel_size_) || selector_anchor_voxel_size_ <= 0)
    throw std::invalid_argument("Selector.anchor_voxel_size must be positive");
  if (selector_architecture_ == "anchor" && selector_enabled_ && ultra_light_mode_ != 1)
    throw std::invalid_argument("Anchor Selector requires mode1 joint LoD / Master-only Full scheduling");
  const auto selector_backend_node = settings_file["Selector.backend"];
  if (!selector_backend_node.empty()) selector_backend_node >> selector_backend_;
  if (selector_architecture_ == "anchor") selector_backend_ = "legacy";
  if (selector_backend_ != "legacy" && selector_backend_ != "cuda")
    throw std::invalid_argument("Selector.backend must be legacy or cuda");
  const auto selection_mode_node = settings_file["Selector.selection_mode"];
  if (!selection_mode_node.empty()) selection_mode_node >> selector_selection_mode_;
  if (selector_selection_mode_ != "cached_bernoulli" && selector_selection_mode_ != "flexgs")
    throw std::invalid_argument("Selector.selection_mode must be cached_bernoulli or flexgs");
  int gumbel_budget_normalize = selector_gumbel_budget_normalize_ ? 1 : 0;
  readInt("Selector.gumbel_budget_normalize", gumbel_budget_normalize);
  selector_gumbel_budget_normalize_ = gumbel_budget_normalize != 0;
  if (selector_selection_mode_ == "flexgs")
    std::cout << "[Gaussian Mapper] gumbel_budget_normalize=" << selector_gumbel_budget_normalize_ << '\n';
  std::cout << "[Gaussian Mapper] selector_backend=" << selector_backend_ << std::endl;
  const cv::FileNode ratios_node = settings_file["Selector.target_ratios"];
  if(real_lod_.enabled)selector_target_ratios_={.2f,.4f,.6f,.8f};
  if ((ultra_light_mode_ == 2 || real_lod_.enabled) && !ratios_node.isNone() &&
      (!ratios_node.isSeq() || ratios_node.size() == 0))
    throw std::invalid_argument("Mode2 Selector.target_ratios must be a nonempty sequence in (0, 1]");
  if (!ratios_node.empty() && ratios_node.isSeq()) {
    std::vector<float> configured_ratios;
    for (auto it = ratios_node.begin(); it != ratios_node.end(); ++it) {
      const float ratio = (*it).operator float();
      if ((ultra_light_mode_ == 2 || real_lod_.enabled) && (!std::isfinite(ratio) || ratio <= 0.0f || ratio > 1.0f))
        throw std::invalid_argument("Mode2 Selector.target_ratios must be in (0, 1]");
      if (ratio > 0.0f && ratio <= 1.0f) configured_ratios.push_back(ratio);
    }
    if (!configured_ratios.empty()) selector_target_ratios_ = configured_ratios;
  }
  if (ultrafast_rotation_.enabled) {
    ultrafast_rotation_.configureRatios(selector_target_ratios_);
    std::cout << "[UltrafastRotation CONFIG] ratios=";
    for(auto ratio:selector_target_ratios_)std::cout<<ratio<<',';
    std::cout << "Full stop_at_online_end=0 topology_full_only=1 tail_policy=configured\n";
  }
  mode2_online_.ratios.assign(selector_target_ratios_.size(), 0);
  mode2_tail_.ratios.assign(selector_target_ratios_.size(), 0);
  int protection_enabled = selector_protection_enabled_ ? 1 : 0;
  readInt("Selector.protection_enabled", protection_enabled);
  selector_protection_enabled_ = protection_enabled != 0;
  readFloat("Protection.drift_temperature", protection_config_.drift_temperature);
  readFloat("Protection.beta_fast", protection_config_.beta_fast);
  readFloat("Protection.beta_slow", protection_config_.beta_slow);
  readFloat("Protection.maturity_rise_rate", protection_config_.maturity_rise_rate);
  readFloat("Protection.maturity_fall_rate", protection_config_.maturity_fall_rate);
  int maturity_diagnostics = protection_config_.maturity_diagnostics ? 1 : 0;
  readInt("Protection.maturity_diagnostics", maturity_diagnostics);
  protection_config_.maturity_diagnostics = maturity_diagnostics != 0;
  readFloat("Protection.compatibility_mature_threshold", protection_config_.compatibility_mature_threshold);
  int performance_logging = protection_config_.performance_logging ? 1 : 0;
  readInt("Protection.performance_logging", performance_logging);
  protection_config_.performance_logging = performance_logging != 0;
  int reliability=0,gate=0,appearance=1;
  readInt("MaturityV2.enabled",reliability);protection_config_.reliability_v2=reliability!=0;
  int contribution_enabled=0;
  readInt("MaturityV2.contribution_enabled",contribution_enabled);
  protection_config_.rendering_contribution=contribution_enabled!=0;
  readFloat("MaturityV2.contribution_decay",protection_config_.contribution_decay);
  readFloat("MaturityV2.tau_r",protection_config_.contribution_tau);
  readFloat("MaturityV2.contribution_weight",protection_config_.contribution_weight);
  readInt("MaturityV2.contribution_every_full",protection_config_.contribution_every_full);
  readInt("MaturityV2.contribution_pixel_stride",protection_config_.contribution_pixel_stride);
  readFloat("MaturityV2.tau_o",protection_config_.reliability_tau_o);
  readFloat("MaturityV2.tau_v",protection_config_.reliability_tau_v);
  readFloat("MaturityV2.xyz_scale_floor",protection_config_.reliability_xyz_floor);
  readFloat("MaturityV2.inherit_fraction",protection_config_.reliability_inherit);
  readFloat("MaturityV2.selector_loss_weight",protection_config_.reliability_loss_weight);
  readFloat("MaturityV2.selector_delta",protection_config_.reliability_delta);
  readFloat("MaturityV2.gate_low",protection_config_.reliability_gate_low);
  readFloat("MaturityV2.gate_high",protection_config_.reliability_gate_high);
  readInt("MaturityV2.compensation_gate",gate);protection_config_.reliability_gate=gate!=0;
  readInt("MaturityV2.gate_appearance",appearance);protection_config_.reliability_gate_appearance=appearance!=0;
  protection_config_.validate();
  std::cout << "[Protection CONFIG] schema=continuous_cycle_geometric drift_temperature=" << protection_config_.drift_temperature
            << " beta_fast=" << protection_config_.beta_fast << " beta_slow=" << protection_config_.beta_slow
            << " maturity_rise_rate=" << protection_config_.maturity_rise_rate
            << " maturity_fall_rate=" << protection_config_.maturity_fall_rate
            << " compatibility_mature_threshold=" << protection_config_.compatibility_mature_threshold
            << " observation_modes=1,2" << std::endl;
  readFloat("Selector.temperature", selector_temperature_);
  readFloat("Selector.learning_rate", selector_learning_rate_);
  readFloat("Selector.render_loss_weight", selector_render_loss_weight_);
  readFloat("Selector.gi_loss_weight", selector_gi_loss_weight_);
  readFloat("Selector.ratio_loss_weight", selector_ratio_loss_weight_);
  readFloat("Selector.lod_render_scale", selector_lod_render_scale_);
  if (!std::isfinite(selector_lod_render_scale_) ||
      selector_lod_render_scale_ <= 0.0f || selector_lod_render_scale_ > 1.0f)
    throw std::invalid_argument("Selector.lod_render_scale must be in (0, 1]");
  int record_keyframe_used_times = record_keyframe_used_times_ ? 1 : 0;
  readInt("Mapping.record_keyframe_used_times", record_keyframe_used_times);
  record_keyframe_used_times_ = record_keyframe_used_times != 0;
  int detach_gaussian_parameters = selector_detach_gaussian_parameters_ ? 1 : 0;
  readInt("Selector.detach_gaussian_parameters", detach_gaussian_parameters);
  selector_detach_gaussian_parameters_ = detach_gaussian_parameters != 0;
  // Old explicit frozen configurations remain LoD-only when the new key is absent.
  real_lod_full_loss_weight_ = selector_detach_gaussian_parameters_ ? 0.f : 1.f;
  readFloat("RealLoD.full_loss_weight", real_lod_full_loss_weight_);
  int view_budget=0,view_budget_control=0,view_budget_residual=0;
  readInt("ViewBudget.enabled",view_budget);
  view_budget_enabled_=view_budget!=0;
  readInt("ViewBudget.global_control",view_budget_control);
  view_budget_global_control_=view_budget_control!=0;
  readInt("ViewBudget.view_residual",view_budget_residual);
  view_budget_view_residual_=view_budget_residual!=0;
  std::string conditioning="off";
  if(!settings_file["ViewBudget.conditioning"].empty())settings_file["ViewBudget.conditioning"]>>conditioning;
  TORCH_CHECK(conditioning=="off"||conditioning=="mode"||conditioning=="view"||conditioning=="selector_view"||conditioning=="compensator_view",
      "ViewBudget.conditioning must be off, mode, view, selector_view or compensator_view");
  view_budget_conditioning_=conditioning=="mode"?1:(conditioning=="view"?2:(conditioning=="selector_view"?3:(conditioning=="compensator_view"?4:0)));
  if(!settings_file["ViewBudget.view_residual_input"].empty())
    settings_file["ViewBudget.view_residual_input"]>>view_budget_residual_input_;
  TORCH_CHECK(view_budget_residual_input_=="hidden32"||view_budget_residual_input_=="physical11",
      "ViewBudget.view_residual_input must be hidden32 or physical11");
  TORCH_CHECK(view_budget_residual_input_=="hidden32"||view_budget_view_residual_,
      "physical11 requires ViewBudget.view_residual=1");
  TORCH_CHECK(!view_budget_view_residual_||(view_budget_enabled_&&!view_budget_global_control_),
      "ViewBudget.view_residual requires enabled local ViewBudget");
  readFloat("ViewBudget.loss_weight",view_budget_loss_weight_);
  readFloat("ViewBudget.li_weight",view_budget_li_weight_);
  readInt("ViewBudget.seed",view_budget_seed_);
  TORCH_CHECK(std::isfinite(view_budget_loss_weight_)&&view_budget_loss_weight_>=0&&
      view_budget_seed_>=0,"ViewBudget weight and seed must be nonnegative");
  TORCH_CHECK(std::isfinite(view_budget_li_weight_)&&view_budget_li_weight_>=0&&
      (view_budget_li_weight_==0||(view_budget_enabled_&&view_budget_loss_weight_>0&&!view_budget_global_control_)),
      "Local LI requires an enabled local ViewBudget branch and a nonnegative weight");
  int mixed_enabled=0;
  readInt("MixedRatio.enabled",mixed_enabled);mixed_ratio_config_.enabled=mixed_enabled!=0;
  readInt("MixedRatio.seed",mixed_ratio_config_.seed);
  readInt("MixedRatio.concentration_high",mixed_ratio_config_.concentration[0]);
  readInt("MixedRatio.concentration_middle",mixed_ratio_config_.concentration[1]);
  readInt("MixedRatio.concentration_low",mixed_ratio_config_.concentration[2]);
  readFloat("MixedRatio.eta_min",mixed_ratio_config_.eta_min);
  readFloat("MixedRatio.eta_max",mixed_ratio_config_.eta_max);
  mixed_ratio_config_.validate();
  int view_residual=0;readInt("ViewResidual.enabled",view_residual);
  selector_view_residual_enabled_=view_residual!=0;
  int view_input=0;readInt("ViewInput.enabled",view_input);
  selector_view_input_enabled_=view_input!=0;
  selector_view_gi_weight_=selector_gi_loss_weight_;
  readFloat("ViewInput.single_view_gi_weight",selector_view_gi_weight_);
  TORCH_CHECK(std::isfinite(selector_view_gi_weight_)&&selector_view_gi_weight_>=0,"ViewInput single-view GI weight must be nonnegative");
  int view_gi_lowest_two=0;readInt("ViewInput.single_view_gi_lowest_two_only",view_gi_lowest_two);
  selector_view_gi_lowest_two_only_=view_gi_lowest_two!=0;
  int view_visible_budget=0;readInt("ViewInput.visible_point_budget",view_visible_budget);
  selector_view_visible_budget_=view_visible_budget!=0;
  selector_view_budget_ratio_values_.clear();
  if(selector_view_visible_budget_) {
    TORCH_CHECK(selector_view_input_enabled_,"Visible point training budget requires ViewInput");
    for(auto node:settings_file["Selector.target_ratios"])
      selector_view_budget_ratio_values_.push_back(node.operator double());
    TORCH_CHECK(selector_view_budget_ratio_values_.size()==selector_target_ratios_.size(),
        "Visible budget requires explicit Selector.target_ratios");
  }
  selector_view_ratio_indices_.clear();
  const auto view_ratios=settings_file["ViewInput.target_ratios"];
  if(!view_ratios.isNone()) {
    TORCH_CHECK(selector_view_input_enabled_&&view_ratios.isSeq()&&view_ratios.size()>0,
        "ViewInput.target_ratios requires enabled ViewInput and a nonempty sequence");
    for(auto node:view_ratios) {
      const float ratio=node.operator float();
      auto found=std::find_if(selector_target_ratios_.begin(),selector_target_ratios_.end(),
          [ratio](float value){return std::isfinite(ratio)&&std::abs(value-ratio)<1e-6f;});
      TORCH_CHECK(found!=selector_target_ratios_.end(),"ViewInput ratios must be existing Selector training ratios");
      const auto index=std::size_t(found-selector_target_ratios_.begin());
      TORCH_CHECK(selector_view_ratio_indices_.empty()||index>selector_view_ratio_indices_.back(),
          "ViewInput ratios must follow Selector order without duplicates");
      selector_view_ratio_indices_.push_back(index);
    }
  }
  int health_enabled=lod_health_guard_.enabled?1:0;
  readInt("LoDHealth.enabled",health_enabled);lod_health_guard_.enabled=health_enabled!=0;
  readInt("LoDHealth.zero_patience",lod_health_guard_.zero_patience);
  readInt("LoDHealth.saturation_patience",lod_health_guard_.saturation_patience);
  float health_probability=lod_health_guard_.probability_floor,health_scale=lod_health_guard_.scale_limit;
  float health_xyz=lod_health_guard_.xyz_limit,health_tail=lod_health_guard_.tail_fraction;
  readFloat("LoDHealth.probability_floor",health_probability);readFloat("LoDHealth.scale_factor_limit",health_scale);
  readFloat("LoDHealth.xyz_anchor_limit",health_xyz);readFloat("LoDHealth.tail_fraction",health_tail);
  lod_health_guard_.probability_floor=health_probability;lod_health_guard_.scale_limit=health_scale;
  lod_health_guard_.xyz_limit=health_xyz;lod_health_guard_.tail_fraction=health_tail;
  TORCH_CHECK(lod_health_guard_.zero_patience>0&&lod_health_guard_.saturation_patience>0&&
      std::isfinite(health_probability)&&std::isfinite(health_scale)&&std::isfinite(health_xyz)&&std::isfinite(health_tail)&&
      health_probability>0&&health_probability<.5&&health_scale>1&&health_xyz>0&&health_tail>0&&health_tail<=1,
      "Invalid LoDHealth thresholds");
  if (real_lod_.enabled && (!std::isfinite(real_lod_full_loss_weight_) ||
      real_lod_full_loss_weight_ < 0.f ||
      (real_lod_full_loss_weight_ > 0.f && selector_detach_gaussian_parameters_)))
    throw std::invalid_argument("RealLoD.full_loss_weight must be nonnegative; Full supervision requires a live Master");
  std::cout << "[Gaussian Mapper]Selector.detach_gaussian_parameters = "
            << static_cast<int>(selector_detach_gaussian_parameters_)
            << " (joint replay keeps Gaussian gradients)" << std::endl;
  int replay_enabled = selector_replay_enabled_ ? 1 : 0;
  readInt("Selector.replay_enabled", replay_enabled);
  selector_replay_enabled_ = replay_enabled != 0;
  readInt("Selector.replay_interval", selector_replay_interval_);
  readInt("Selector.replay_num_frames", selector_replay_num_frames_);
  int max_mapping_iterations = static_cast<int>(
      std::min<std::uint64_t>(selector_max_mapping_iterations_,
                              static_cast<std::uint64_t>(INT_MAX)));
  readInt("Selector.max_mapping_iterations", max_mapping_iterations);
  if (max_mapping_iterations >= 0)
    selector_max_mapping_iterations_ =
        static_cast<std::uint64_t>(max_mapping_iterations);
  std::cout << "[Gaussian Mapper]Selector.max_mapping_iterations = "
            << selector_max_mapping_iterations_ << std::endl;

  int transform_enabled = transform_field_config_.enabled ? 1 : 0;
  readInt("TransformField.enable", transform_enabled);
  transform_field_config_.enabled = transform_enabled != 0;
  const auto tf_backend = settings_file["TransformField.backend"];
  if (!tf_backend.empty()) tf_backend >> transform_field_config_.backend;
  if (transform_field_config_.backend != "legacy" && transform_field_config_.backend != "cuda")
    throw std::invalid_argument("TransformField.backend must be legacy or cuda");
  std::cout << "[Gaussian Mapper]TransformField.backend = " << transform_field_config_.backend << std::endl;
  readInt("TransformField.start_iter", transform_field_config_.start_iter);
  const auto tf_start_mode = settings_file["TransformField.start_mode"];
  if (!tf_start_mode.empty()) tf_start_mode >> transform_field_config_.start_mode;
  readFloat("TransformField.maturity_compute_epsilon", transform_field_config_.maturity_compute_epsilon);
  readFloat("Elastic.entry_maturity", transform_field_config_.elastic_entry_maturity);
  readFloat("TransformField.ready_maturity", transform_field_config_.ready_maturity);
  readInt("TransformField.min_ready_gaussians", transform_field_config_.min_ready_gaussians);
  TORCH_CHECK(std::isfinite(transform_field_config_.elastic_entry_maturity) &&
      transform_field_config_.elastic_entry_maturity >= 0 && transform_field_config_.elastic_entry_maturity < 1,
      "Elastic.entry_maturity must be in [0,1)");
  TORCH_CHECK(std::isfinite(transform_field_config_.ready_maturity) &&
      transform_field_config_.ready_maturity >= 0 && transform_field_config_.ready_maturity <= 1 &&
      transform_field_config_.min_ready_gaussians > 0, "Invalid TF population readiness");
  continuous_tf_statistics_.entry_threshold = transform_field_config_.elastic_entry_maturity;
  TORCH_CHECK(transform_field_config_.start_mode == "fixed_iteration" ||
      transform_field_config_.start_mode == "first_maturity_cycle" ||
      transform_field_config_.start_mode == "maturity_population", "Unknown TransformField.start_mode");
  TORCH_CHECK(std::isfinite(transform_field_config_.maturity_compute_epsilon) &&
      transform_field_config_.maturity_compute_epsilon >= 0 && transform_field_config_.maturity_compute_epsilon <= 1,
      "TransformField.maturity_compute_epsilon must be in [0,1]");
  readFloat("TransformField.aabb_scale", transform_field_config_.aabb_scale);
  readInt("TransformField.spatial_resolution",
          transform_field_config_.hexplane.spatial_resolution);
  readInt("TransformField.ratio_resolution",
          transform_field_config_.hexplane.ratio_resolution);
  readInt("TransformField.feature_dim",
          transform_field_config_.hexplane.feature_dim);
  const cv::FileNode multires_node = settings_file["TransformField.multires"];
  if (!multires_node.empty() && multires_node.isSeq()) {
    std::vector<int> multires;
    for (auto it = multires_node.begin(); it != multires_node.end(); ++it)
      multires.push_back((*it).operator int());
    if (!multires.empty()) transform_field_config_.hexplane.multires = multires;
  }
  readInt("TransformField.hidden_dim", transform_field_config_.hidden_dim);
  readFloat("TransformField.grid_lr_init", transform_field_config_.grid_lr_init);
  readFloat("TransformField.grid_lr_final",
            transform_field_config_.grid_lr_final);
  readFloat("TransformField.mlp_lr_init", transform_field_config_.mlp_lr_init);
  readFloat("TransformField.mlp_lr_final", transform_field_config_.mlp_lr_final);
  readFloat("TransformField.lr_delay_mult",
            transform_field_config_.lr_delay_mult);
  readInt("TransformField.lr_max_steps", transform_field_config_.lr_max_steps);
  readFloat("TransformField.spatial_smooth_weight",
            transform_field_config_.spatial_smooth_weight);
  readFloat("TransformField.ratio_smooth_weight",
            transform_field_config_.ratio_smooth_weight);
  readFloat("TransformField.ratio_l1_weight",
            transform_field_config_.ratio_l1_weight);
  int outside_identity = transform_field_config_.outside_identity ? 1 : 0;
  readInt("TransformField.outside_identity", outside_identity);
  transform_field_config_.outside_identity = outside_identity != 0;

  int dynamic_aabb_enabled = dynamic_aabb_config_.enabled ? 1 : 0;
  readInt("TransformField.dynamic_aabb", dynamic_aabb_enabled);
  dynamic_aabb_config_.enabled = dynamic_aabb_enabled != 0;
  int aabb_check_interval = static_cast<int>(
      std::clamp<std::int64_t>(dynamic_aabb_config_.check_interval, 0,
                               INT_MAX));
  readInt("TransformField.aabb_check_interval", aabb_check_interval);
  if (aabb_check_interval >= 0)
    dynamic_aabb_config_.check_interval = aabb_check_interval;
  int outside_count_threshold = static_cast<int>(std::clamp<std::int64_t>(
      dynamic_aabb_config_.outside_count_threshold, 0, INT_MAX));
  readInt("TransformField.outside_count_threshold", outside_count_threshold);
  if (outside_count_threshold >= 0)
    dynamic_aabb_config_.outside_count_threshold = outside_count_threshold;
  readFloat("TransformField.expand_margin_ratio",
            dynamic_aabb_config_.expansion_margin);
  readFloat("TransformField.bound_quantile_low",
            dynamic_aabb_config_.bound_quantile_low);
  readFloat("TransformField.bound_quantile_high",
            dynamic_aabb_config_.bound_quantile_high);
  int fixed_resolution = dynamic_aabb_config_.fixed_resolution ? 1 : 0;
  readInt("TransformField.fixed_resolution", fixed_resolution);
  dynamic_aabb_config_.fixed_resolution = fixed_resolution != 0;

  int change_distill_enabled = change_distill_config_.enabled ? 1 : 0;
  readInt("ChangeDistill.enable", change_distill_enabled);
  change_distill_config_.enabled = change_distill_enabled != 0;
  if(change_distill_config_.enabled && selector_compensation_method_!="shared_mlp")
    throw std::invalid_argument("Private14 online experiment requires ChangeDistill.enable=0");
  int distill_selector =
      change_distill_config_.distill_selector_on_geometry_change ? 1 : 0;
  readInt("ChangeDistill.distill_selector_on_geometry_change", distill_selector);
  change_distill_config_.distill_selector_on_geometry_change =
      distill_selector != 0;
  int distill_tf_geometry =
      change_distill_config_.distill_tf_on_geometry_change ? 1 : 0;
  readInt("ChangeDistill.distill_tf_on_geometry_change", distill_tf_geometry);
  change_distill_config_.distill_tf_on_geometry_change =
      distill_tf_geometry != 0;
  int distill_tf_aabb = change_distill_config_.distill_tf_on_aabb_change ? 1 : 0;
  readInt("ChangeDistill.distill_tf_on_aabb_change", distill_tf_aabb);
  change_distill_config_.distill_tf_on_aabb_change = distill_tf_aabb != 0;
  readFloat("ChangeDistill.position_weight",
            change_distill_config_.position_weight);
  readFloat("ChangeDistill.scale_weight", change_distill_config_.scale_weight);
  readFloat("ChangeDistill.rotation_weight",
            change_distill_config_.rotation_weight);
  readFloat("ChangeDistill.selector_weight",
            change_distill_config_.selector_weight);
  readFloat("ChangeDistill.selector_learning_rate",
            change_distill_config_.selector_learning_rate);
  readFloat("ChangeDistill.tf_grid_learning_rate",
            change_distill_config_.tf_grid_learning_rate);
  readFloat("ChangeDistill.tf_mlp_learning_rate",
            change_distill_config_.tf_mlp_learning_rate);
  int distill_batch_size = static_cast<int>(std::clamp<std::int64_t>(
      change_distill_config_.batch_size, 0, INT_MAX));
  readInt("ChangeDistill.batch_size", distill_batch_size);
  change_distill_config_.batch_size = distill_batch_size;
  int cd_epochs = static_cast<int>(change_distill_config_.epochs);
  readInt("ChangeDistill.epochs", cd_epochs);
  change_distill_config_.epochs = cd_epochs;
  if (!settings_file["ChangeDistill.selector_epochs"].empty()) {
    int selector_epochs = cd_epochs;
    readInt("ChangeDistill.selector_epochs", selector_epochs);
    change_distill_config_.selector_epochs = selector_epochs;
  }
  if (!settings_file["ChangeDistill.tf_epochs"].empty()) {
    int tf_epochs = cd_epochs;
    readInt("ChangeDistill.tf_epochs", tf_epochs);
    change_distill_config_.tf_epochs = tf_epochs;
  }
  int cd_seed = static_cast<int>(change_distill_config_.seed);
  readInt("ChangeDistill.seed", cd_seed);
  if (cd_seed < 0) throw std::invalid_argument("ChangeDistill.seed must be nonnegative");
  change_distill_config_.seed = static_cast<uint64_t>(cd_seed);
  int anchors_enabled = change_distill_config_.enable_unaffected_anchors ? 1 : 0;
  readInt("ChangeDistill.enable_unaffected_anchors", anchors_enabled);
  change_distill_config_.enable_unaffected_anchors = anchors_enabled != 0;
  readFloat("ChangeDistill.anchor_ratio_to_corrected",
            change_distill_config_.anchor_ratio_to_corrected);
  int anchor_max = static_cast<int>(change_distill_config_.anchor_max_count);
  readInt("ChangeDistill.anchor_max_count", anchor_max);
  change_distill_config_.anchor_max_count = anchor_max;
  readFloat("ChangeDistill.anchor_loss_weight", change_distill_config_.anchor_loss_weight);
  int post_change_replay = change_distill_config_.post_change_replay ? 1 : 0;
  readInt("ChangeDistill.post_change_replay", post_change_replay);
  change_distill_config_.post_change_replay = post_change_replay != 0;

  int dsd_enabled = drift_stability_config_.enabled ? 1 : 0;
  readInt("DriftStability.enable", dsd_enabled);
  drift_stability_config_.enabled = dsd_enabled != 0;
  int gradient_enabled = gradient_constraint_config_.enabled ? 1 : 0;
  readInt("GradientConstraint.enable", gradient_enabled);
  gradient_constraint_config_.enabled = gradient_enabled != 0;
  readInt("GradientConstraint.check_interval", gradient_constraint_config_.check_interval);
  readInt("GradientConstraint.memory_views_per_check", gradient_constraint_config_.memory_views_per_check);
  readInt("GradientConstraint.memory_ratios_per_check", gradient_constraint_config_.memory_ratios_per_check);
  readInt("GradientConstraint.max_candidate_checks", gradient_constraint_config_.max_candidate_checks);
  readInt("GradientConstraint.tf_warmup_steps", gradient_constraint_config_.tf_warmup_steps);
  readInt("GradientConstraint.log_interval", gradient_constraint_config_.log_interval);
  int fallback_enabled = gradient_constraint_config_.cancel_fallback_enable ? 1 : 0;
  readInt("GradientConstraint.cancel_fallback_enable", fallback_enabled);
  gradient_constraint_config_.cancel_fallback_enable = fallback_enabled != 0;
  float fallback_scale = static_cast<float>(gradient_constraint_config_.cancel_fallback_scale);
  readFloat("GradientConstraint.cancel_fallback_scale", fallback_scale);
  gradient_constraint_config_.cancel_fallback_scale = fallback_scale;
  readInt("ConflictReplay.trigger_cancel_count", conflict_replay_config_.trigger_cancel_count);
  readInt("ConflictReplay.source_view_count", conflict_replay_config_.source_view_count);
  readInt("ConflictReplay.views_per_source", conflict_replay_config_.views_per_source);
  readInt("ConflictReplay.remote_candidate_count", conflict_replay_config_.remote_candidate_count);
  readInt("ConflictReplay.queue_capacity", conflict_replay_config_.queue_capacity);
  gradient_constraint_config_.validate();
  int dsd_probe_interval = static_cast<int>(drift_stability_config_.probe_interval);
  readInt("DriftStability.probe_interval", dsd_probe_interval);
  drift_stability_config_.probe_interval = dsd_probe_interval;
  int dsd_min_bad_views = static_cast<int>(drift_stability_config_.min_bad_views);
  readInt("DriftStability.min_bad_views", dsd_min_bad_views);
  drift_stability_config_.min_bad_views = dsd_min_bad_views;
  int dsd_max_count = static_cast<int>(drift_stability_config_.max_gaussians);
  readInt("DriftStability.max_gaussians", dsd_max_count);
  drift_stability_config_.max_gaussians = dsd_max_count;
  int dsd_batch_size = static_cast<int>(drift_stability_config_.batch_size);
  readInt("DriftStability.batch_size", dsd_batch_size);
  drift_stability_config_.batch_size = dsd_batch_size;
  int dsd_consecutive = static_cast<int>(
      drift_stability_config_.consecutive_trigger_count);
  readInt("DriftStability.consecutive_trigger_count", dsd_consecutive);
  drift_stability_config_.consecutive_trigger_count = dsd_consecutive;
  int dsd_horizon = static_cast<int>(drift_stability_config_.reference_horizon);
  readInt("DriftStability.reference_horizon", dsd_horizon);
  drift_stability_config_.reference_horizon = dsd_horizon;
  int dsd_min_age = static_cast<int>(drift_stability_config_.reference_min_age);
  readInt("DriftStability.reference_min_age", dsd_min_age);
  drift_stability_config_.reference_min_age = dsd_min_age;
  int dsd_max_age = static_cast<int>(drift_stability_config_.reference_max_age);
  readInt("DriftStability.reference_max_age", dsd_max_age);
  drift_stability_config_.reference_max_age = dsd_max_age;
  readFloat("DriftStability.reference_ema_decay",
            drift_stability_config_.reference_ema_decay);
  int dsd_refresh_count = static_cast<int>(
      drift_stability_config_.non_regression_refresh_count);
  readInt("DriftStability.non_regression_refresh_count", dsd_refresh_count);
  drift_stability_config_.non_regression_refresh_count = dsd_refresh_count;
  readFloat("DriftStability.function_threshold",
            drift_stability_config_.function_threshold);
  readFloat("DriftStability.psnr_regression_threshold",
            drift_stability_config_.psnr_regression_threshold);
  readFloat("DriftStability.budget_relative_tolerance",
            drift_stability_config_.budget_relative_tolerance);
  readFloat("DriftStability.position_weight",
            drift_stability_config_.position_weight);
  readFloat("DriftStability.scale_weight",
            drift_stability_config_.scale_weight);
  readFloat("DriftStability.rotation_weight",
            drift_stability_config_.rotation_weight);
  readFloat("DriftStability.selector_learning_rate",
            drift_stability_config_.selector_learning_rate);
  readFloat("DriftStability.tf_grid_learning_rate",
            drift_stability_config_.tf_grid_learning_rate);
  readFloat("DriftStability.tf_mlp_learning_rate",
            drift_stability_config_.tf_mlp_learning_rate);
  int dsd_epochs = static_cast<int>(drift_stability_config_.epochs);
  readInt("DriftStability.epochs", dsd_epochs);
  drift_stability_config_.epochs = dsd_epochs;
  int dsd_seed = static_cast<int>(drift_stability_config_.seed);
  readInt("DriftStability.seed", dsd_seed);
  if (dsd_seed < 0)
    throw std::invalid_argument("DriftStability.seed must be nonnegative");
  drift_stability_config_.seed = static_cast<uint64_t>(dsd_seed);

  int outlier_count = static_cast<int>(historical_outlier_config_.count);
  readInt("HistoricalOutlier.count", outlier_count);
  if (outlier_count >= 0) historical_outlier_config_.count = outlier_count;
  int diversity = historical_outlier_config_.diversity_enabled ? 1 : 0;
  readInt("HistoricalOutlier.diversity_enabled", diversity);
  historical_outlier_config_.diversity_enabled = diversity != 0;
  readFloat("HistoricalOutlier.translation_weight",
            historical_outlier_config_.translation_weight);
  readFloat("HistoricalOutlier.rotation_weight",
            historical_outlier_config_.rotation_weight);
  readFloat("HistoricalOutlier.diversity_recent_weight",
            historical_outlier_config_.diversity_recent_weight);
  int recent_enabled = recent_window_config_.enabled ? 1 : 0;
  readInt("RecentWindow.enable", recent_enabled);
  recent_window_config_.enabled = recent_enabled != 0;
  int recent_size = static_cast<int>(recent_window_config_.size);
  readInt("RecentWindow.size", recent_size);
  if (recent_size > 0) recent_window_config_.size = recent_size;
  readFloat("RecentWindow.ema_decay", recent_window_config_.ema_decay);
  // Retired RecentWindow GI is available only through an explicit legacy mode.
  global_gi_config_.mode = "disabled";
  const auto gi_mode = settings_file["GlobalGI.mode"];
  if (!gi_mode.empty()) global_gi_config_.mode = gi_mode.operator std::string();
  auto readDouble = [&](const char* key, double& value) {
    const auto node = settings_file[key];
    if (!node.empty()) value = node.operator double();
  };
  readDouble("GlobalGI.view_ratio", global_gi_config_.view_ratio);
  readDouble("GlobalGI.coverage_anchor_fraction", global_gi_config_.coverage_anchor_fraction);
  int gi_seed = 0, gi_refresh = 1000, gi_max_views = 0, gi_logging = 1, gi_verbose = 0;
  readInt("GlobalGI.sampling_seed", gi_seed);
  readInt("GlobalGI.refresh_interval", gi_refresh);
  readInt("GlobalGI.max_views", gi_max_views);
  readInt("GlobalGI.performance_logging", gi_logging);
  readInt("GlobalGI.verbose_logging", gi_verbose);
  int marginal_shadow = 0, marginal_diagnostics = 0;
  readInt("GlobalGI.marginal_shadow", marginal_shadow);
  readInt("GlobalGI.marginal_diagnostics", marginal_diagnostics);
  global_gi_config_.marginal_shadow = marginal_shadow != 0;
  global_gi_config_.marginal_diagnostics = marginal_diagnostics != 0;
  int hierarchical_enabled = 0, hierarchical_diagnostics = 0;
  readInt("GlobalGI.HierarchicalRepresentative.enable", hierarchical_enabled);
  readInt("GlobalGI.HierarchicalRepresentative.diagnostics", hierarchical_diagnostics);
  global_gi_config_.hierarchical_representative_enabled = hierarchical_enabled != 0;
  global_gi_config_.hierarchical_representative_diagnostics = hierarchical_diagnostics != 0;
  if (selector_selection_mode_ == "flexgs" &&
      (selector_architecture_ != "anchor" || !selector_anchor_gaussian_attributes_ ||
       !selector_explicit_anchor_blocks_ || selector_anchor_global_xyz_ ||
       transform_field_config_.elastic_entry_maturity != 0 ||
       !std::isfinite(selector_temperature_) || selector_temperature_ <= 0))
    throw std::invalid_argument("FlexGS sampling requires explicit attribute-conditioned Anchors, entry=0 and positive temperature");
  if((selector_anchor_set_encoder_ || selector_anchor_set_statistics_ || selector_anchor_context_residual_) && transform_field_config_.elastic_entry_maturity!=0)
    throw std::invalid_argument("Anchor Set Encoder requires complete member sets: Elastic.entry_maturity=0");
  if (selector_architecture_ == "anchor") {
    if(selector_compensation_ && (transform_field_config_.enabled || transform_field_config_.elastic_entry_maturity!=0))
      throw std::invalid_argument("Selector compensation requires TransformField.enable=0 and Elastic.entry_maturity=0");
    if(selector_explicit_anchor_blocks_ &&
        !((global_gi_config_.mode=="disabled" && (selector_gi_loss_weight_==0 || flexgs_importance_enabled_)) ||
          global_gi_config_.mode=="streaming"))
      throw std::invalid_argument("Explicit Anchor map supports disabled or Raw streaming GI only");
    protection_config_.fixed_maturity_one = !(protection_config_.reliability_v2 || selector_protection_enabled_);
    if(protection_config_.reliability_v2)selector_protection_enabled_=false; // No probability mixing.
    if(protection_config_.reliability_v2) {
      TORCH_CHECK(selector_selection_mode_=="flexgs"&&ultra_light_mode_==1,
          "Maturity V2 currently supports Anchor FlexGS mode1 Full/LoD commits");
      protection_config_.maturity_diagnostics=false;
    }
    // Explicit SI ablations reuse GlobalGI.mode and Selector.gi_loss_weight;
    // the default Anchor configs keep both disabled.
    global_gi_config_.marginal_shadow = false;
    global_gi_config_.marginal_diagnostics = false;
    global_gi_config_.hierarchical_representative_enabled = false;
    global_gi_config_.hierarchical_representative_diagnostics = false;
    if (drift_stability_config_.enabled)
      throw std::invalid_argument("Anchor Selector does not support retired DriftStability EMA snapshots");
    std::cout << "[Anchor Selector] feature_dim=" << (selector_anchor_ratio_features_?selector_anchor_ratio_shared_dim_:selector_anchor_feature_dim_) << " ratio_features=" << selector_anchor_ratio_features_
              << " width=32 gaussian_attributes=" << selector_anchor_gaussian_attributes_
              << " set_encoder=" << selector_anchor_set_encoder_
              << " set_statistics=" << selector_anchor_set_statistics_
              << " context_residual=" << selector_anchor_context_residual_
              << " input_dim=" << (selector_anchor_set_statistics_?21:(selector_anchor_set_encoder_?50:(selector_anchor_ratio_features_?selector_anchor_ratio_shared_dim_+18+int(selector_anchor_ratio_scalar_input_!="none"):(selector_anchor_feature_dim_+4+(selector_anchor_gaussian_attributes_?7:0)+(selector_anchor_global_xyz_?3:0)))))
              << " global_xyz=" << selector_anchor_global_xyz_
              << " offset_reference=" << ((selector_anchor_global_xyz_ || selector_explicit_anchor_blocks_) ? "current_cell" : "root")
              << " explicit_blocks=" << selector_explicit_anchor_blocks_
              << " anchor_frustum_culling=" << selector_anchor_frustum_culling_
              << " topology_mode=" << selector_anchor_only_pruning_ << " slot_deactivation=" << selector_slot_deactivation_
              << " voxel_size=" << selector_anchor_voxel_size_
              << " backend=LibTorch SI/GI=" << global_gi_config_.mode
              << " gi_loss_weight=" << selector_gi_loss_weight_
              << " joint_lod=1 master_only_full=1\n";
  }
  if(anchor_si_config_.enabled && (!selector_variable_anchor_members_ ||
      selector_compensation_method_!="r3_independent14" || global_gi_config_.mode!="disabled" || selector_gi_loss_weight_!=0))
    throw std::invalid_argument("AnchorSI requires Variable Anchor R3 and disabled legacy SI/GI");
  if(real_lod_.enabled) {
    TORCH_CHECK(selector_enabled_&&selector_explicit_anchor_blocks_&&selector_selection_mode_=="flexgs"&&
        ultra_light_mode_==1&&!ultrafast_rotation_.serial_full_lod&&
        !selector_replay_enabled_&&!gradient_constraint_config_.enabled&&
        !transform_field_config_.enabled&&!anchor_si_config_.enabled,
        "RealLoD requires Anchor FlexGS mode1 without serial Full/LoD, Replay, Guard, TF or AnchorSI");
    TORCH_CHECK(!protection_config_.reliability_v2&&!selector_protection_enabled_&&
        !protection_config_.reliability_gate&&!protection_config_.rendering_contribution&&
        transform_field_config_.elastic_entry_maturity==0&&global_gi_config_.mode=="disabled",
        "RealLoD replaces maturity/streaming GI with FlexGSImportance; disable their independent paths");
    TORCH_CHECK(selector_lod_render_scale_==1.f,"RealLoD requires native resolution");
    real_lod_ratio_counts_.assign(selector_target_ratios_.size(),0);
    std::cout<<"[RealLoD CONFIG] online=full tail=full lod=ratio_only steps=online_mapping_count importance="
             <<flexgs_importance_enabled_<<" importance_interval="<<flexgs_importance_interval_
              <<" fine_full_loss_weight="<<real_lod_full_loss_weight_<<" adam_commits_per_step=1"
             <<" master_frozen="<<selector_detach_gaussian_parameters_<<'\n';
  }
  if(selector_view_residual_enabled_||selector_view_input_enabled_) {
    TORCH_CHECK(!(selector_view_residual_enabled_&&selector_view_input_enabled_),"Choose ViewInput or ViewResidual");
    TORCH_CHECK(real_lod_.enabled&&real_lod_full_loss_weight_>0&&!selector_detach_gaussian_parameters_&&
        flexgs_importance_enabled_&&selector_compensation_&&selector_compensation_method_=="spatial_offset3_residual48"&&
        selector_anchor_feature_dim_==0&&!selector_anchor_ratio_features_&&!selector_anchor_global_xyz_&&
        !selector_anchor_set_encoder_&&!selector_anchor_set_statistics_&&!selector_anchor_context_residual_&&
        !mixed_ratio_config_.enabled&&!selector_anchor_frustum_culling_,
        "ViewResidual requires joint Full+LoD, all-view GI, plain11 Residual48, full-map candidates and MixedRatio.enabled=0");
    global_view_ratio_counts_.assign(selector_target_ratios_.size(),0);
    residual_view_ratio_counts_=global_view_ratio_counts_;
    std::cout<<(selector_view_input_enabled_?"[ViewInput CONFIG] shared19->32->32->2 exact_k_GI/BCE view_gi_weight=":"[ViewResidual CONFIG] view7 hidden16 view_gi_weight=")
        <<selector_view_gi_weight_<<" full_map=1 global_round/view_round selector_base_detached_in_view="<<!selector_view_input_enabled_<<'\n';
  }
  if(view_budget_enabled_) {
    TORCH_CHECK(!view_budget_conditioning_||(!view_budget_view_residual_&&!view_budget_global_control_&&view_budget_loss_weight_>0),
        "Joint view conditioning requires local ViewBudget and no residual");
    TORCH_CHECK(real_lod_.enabled&&real_lod_full_loss_weight_>0&&!selector_detach_gaussian_parameters_&&
        flexgs_importance_enabled_&&selector_compensation_&&selector_compensation_method_=="spatial_offset3_residual48"&&
        !selector_view_input_enabled_&&!selector_view_residual_enabled_&&!mixed_ratio_config_.enabled&&
        !selector_anchor_frustum_culling_&&selector_temperature_==1.f&&selector_anchor_feature_dim_==0&&
        !selector_anchor_ratio_features_&&!selector_anchor_global_xyz_&&!selector_anchor_set_encoder_&&
        !selector_anchor_set_statistics_&&!selector_anchor_context_residual_&&selector_anchor_gaussian_attributes_&&
        selector_selection_mode_=="flexgs"&&!selector_protection_enabled_&&!protection_config_.reliability_v2&&
        !protection_config_.reliability_gate&&transform_field_config_.elastic_entry_maturity==0&&
        !transform_field_config_.enabled&&!anchor_si_config_.enabled,
        "ViewBudget requires joint Full+global LoD, plain11/Residual48 and no alternating view modes");
    view_budget_health_=lod_health_guard_;
    view_budget_ratio_counts_.assign(selector_target_ratios_.size(),0);
    std::cout<<"[ViewBudget CONFIG] full+global+auxiliary one_adam=1 weight="<<view_budget_loss_weight_
        <<" sampler=gumbel_st budget=loss gi=0 li_weight="<<view_budget_li_weight_
        <<" conditioning="<<view_budget_conditioning_<<" view_residual="<<view_budget_view_residual_<<" residual_input="<<view_budget_residual_input_<<" scope="
        <<(view_budget_global_control_?"global_control":"nominal_anchor_frustum")<<'\n';
  }
  TORCH_CHECK(!view_budget_conditioning_||view_budget_enabled_,"Joint conditioning requires ViewBudget.enabled=1");
  if(mixed_ratio_config_.enabled) {
    TORCH_CHECK(real_lod_.enabled&&selector_compensation_&&
        selector_compensation_method_=="spatial_offset3_residual48"&&
        selector_anchor_feature_dim_==0&&selector_anchor_gaussian_attributes_&&
        !selector_anchor_global_xyz_&&!selector_anchor_ratio_features_&&
        !selector_anchor_set_encoder_&&!selector_anchor_set_statistics_&&!selector_anchor_context_residual_,
        "MixedRatio requires RealLoD with plain 11D Selector and Residual48");
    TORCH_CHECK(!selector_anchor_frustum_culling_,
        "MixedRatio training uses the full map; set Selector.anchor_frustum_culling: 0");
    TORCH_CHECK(selector_target_ratios_.size()>1&&selector_target_ratios_.front()>0&&
        selector_target_ratios_.back()<=1&&std::adjacent_find(selector_target_ratios_.begin(),
          selector_target_ratios_.end(),std::greater_equal<float>())==selector_target_ratios_.end(),
        "MixedRatio needs strictly ascending training ratios");
    uniform_ratio_counts_.assign(selector_target_ratios_.size(),0);
    mixed_ratio_counts_=uniform_ratio_counts_;
    std::cout<<"[MixedRatio CONFIG] alternating=uniform_round/mixed_round candidates=full_map frustum=0"
        <<" sorting=projected_longest_side seed="<<mixed_ratio_config_.seed<<'\n';
  }
  if (gi_seed < 0 || gi_refresh <= 0 || gi_max_views < 0)
    throw std::invalid_argument("GlobalGI seed/max_views must be nonnegative and refresh_interval positive");
  global_gi_config_.sampling_seed = gi_seed;
  global_gi_config_.refresh_interval = gi_refresh;
  global_gi_config_.max_views = gi_max_views;
  global_gi_config_.performance_logging = gi_logging != 0;
  global_gi_config_.verbose_logging = gi_verbose != 0;
  global_gi_config_.validate();
  if (global_gi_config_.mode == "recent" && !recent_window_config_.enabled)
    throw std::invalid_argument("Recent GI requires RecentWindow.enable=1");
  recent_window_config_.collect_importance = global_gi_config_.mode == "recent";
  std::cout << "[GlobalGI CONFIG] mode=" << global_gi_config_.mode
            << " view_ratio=" << global_gi_config_.view_ratio
            << " anchor_fraction=" << global_gi_config_.coverage_anchor_fraction
            << " refresh_interval=" << global_gi_config_.refresh_interval
            << " verbose_logging=" << global_gi_config_.verbose_logging
            << " hierarchical_representative=" << global_gi_config_.hierarchical_representative_enabled << std::endl;
  int ratio_aware = importance_ratio_aware_weight_ ? 1 : 0;
  readInt("Importance.ratio_aware_weight", ratio_aware);
  importance_ratio_aware_weight_ = ratio_aware != 0;
  TORCH_CHECK(!mixed_ratio_config_.enabled||!importance_ratio_aware_weight_,
      "MixedRatio does not support knot-specific GI weights for continuous local ratios");
  const cv::FileNode ratio_weights_node = settings_file["Importance.ratio_weights"];
  if (!ratio_weights_node.empty() && ratio_weights_node.isSeq()) {
    std::vector<float> ratio_weights;
    for (auto it = ratio_weights_node.begin(); it != ratio_weights_node.end(); ++it)
      ratio_weights.push_back((*it).operator float());
    if (!ratio_weights.empty()) importance_ratio_weights_ = std::move(ratio_weights);
  }
  if (importance_ratio_aware_weight_ &&
      importance_ratio_weights_.size() != selector_target_ratios_.size())
    throw std::invalid_argument(
        "Importance.ratio_weights must match Selector.target_ratios");
  conflict_replay_config_.translation_weight = historical_outlier_config_.translation_weight;
  conflict_replay_config_.rotation_weight = historical_outlier_config_.rotation_weight;
  conflict_replay_config_.validate();
  if (selector_replay_enabled_ && (selector_replay_interval_ <= 0 || selector_replay_num_frames_ <= 0))
    throw std::invalid_argument("Enabled Replay needs positive interval and num_frames");
  std::ostringstream replay_settings;
  replay_settings << std::fixed << std::setprecision(4)
      << "[Gaussian Mapper] Joint Replay enabled=" << selector_replay_enabled_
      << " interval=" << selector_replay_interval_ << " frames=" << selector_replay_num_frames_
      << " trigger=" << conflict_replay_config_.trigger_cancel_count
      << " sources=" << conflict_replay_config_.source_view_count
      << " views_per_source=" << conflict_replay_config_.views_per_source
      << " remote_pool=" << conflict_replay_config_.remote_candidate_count
      << " capacity=" << conflict_replay_config_.queue_capacity
      << " distance_weights(T/R)=" << conflict_replay_config_.translation_weight << '/'
      << conflict_replay_config_.rotation_weight
      << " recent_window=" << recent_window_config_.size << " ema_decay=" << recent_window_config_.ema_decay
      << " guard=" << gradient_constraint_config_.enabled
      << " fallback=" << gradient_constraint_config_.cancel_fallback_enable
      << " fallback_scale=" << gradient_constraint_config_.cancel_fallback_scale;
  std::cout << replay_settings.str() << std::endl;
}

void GaussianMapper::run() {
  // First loop: Initial gaussian mapping
  while (!isStopped()) {
    // Check conditions for initial mapping
    if (hasMetInitialMappingConditions()) {
      pSLAM_->getAtlas()->clearMappingOperation();

      // Get initial sparse map
      auto pMap = pSLAM_->getAtlas()->GetCurrentMap();
      std::vector<ORB_SLAM3::KeyFrame*> vpKFs;
      std::vector<ORB_SLAM3::MapPoint*> vpMPs;
      {
        std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
        vpKFs = pMap->GetAllKeyFrames();
        vpMPs = pMap->GetAllMapPoints();
        for (const auto& pMP : vpMPs) {
          Point3D point3D;
          auto pos = pMP->GetWorldPos();
          point3D.xyz_(0) = pos.x();
          point3D.xyz_(1) = pos.y();
          point3D.xyz_(2) = pos.z();
          auto color = pMP->GetColorRGB();
          point3D.color_(0) = color(0);
          point3D.color_(1) = color(1);
          point3D.color_(2) = color(2);
          scene_->cachePoint3D(pMP->mnId, point3D);
        }
        for (const auto& pKF : vpKFs) {
          std::shared_ptr<GaussianKeyframe> new_kf =
              std::make_shared<GaussianKeyframe>(pKF->mnId, getIteration());
          new_kf->zfar_ = z_far_;
          new_kf->znear_ = z_near_;
          // Pose
          auto pose = pKF->GetPose();
          new_kf->setPose(pose.unit_quaternion().cast<double>(),
                          pose.translation().cast<double>());
          cv::Mat imgRGB_undistorted, imgAux_undistorted;
          try {
            // Camera
            Camera& camera = scene_->cameras_.at(pKF->mpCamera->GetId());
            new_kf->setCameraParams(camera);

            imgRGB_undistorted = pKF->imgLeftRGB;
            imgAux_undistorted = pKF->imgAuxiliary;

            new_kf->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(
                imgRGB_undistorted, device_type_);
            new_kf->img_filename_ = pKF->mNameFile;
            new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
            new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
            new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
          } catch (std::out_of_range) {
            throw std::runtime_error(
                "[GaussianMapper::run]KeyFrame Camera not found!");
          }
          new_kf->computeTransformTensors();
          scene_->addKeyframe(new_kf, &kfid_shuffled_);
          historical_keyframes_[new_kf->fid_] = new_kf;

          increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

          // Features
          std::vector<float> pixels;
          std::vector<float> pointsLocal;
          pKF->GetKeypointInfo(pixels, pointsLocal);
          new_kf->kps_pixel_ = std::move(pixels);
          new_kf->kps_point_local_ = std::move(pointsLocal);
          new_kf->img_undist_ = imgRGB_undistorted;
          new_kf->img_auxiliary_undist_ = imgAux_undistorted;
        }
      }

      // Prepare multi resolution images for training
      for (auto& kfit : scene_->keyframes()) {
        auto pkf = kfit.second;
        if (device_type_ == torch::kCUDA) {
          cv::cuda::GpuMat img_gpu;
          img_gpu.upload(pkf->img_undist_);
          pkf->gaus_pyramid_original_image_.resize(
              num_gaus_pyramid_sub_levels_);
          for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::cuda::GpuMat img_resized;
            cv::cuda::resize(img_gpu, img_resized,
                             cv::Size(pkf->gaus_pyramid_width_[l],
                                      pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
          }
        } else {
          pkf->gaus_pyramid_original_image_.resize(
              num_gaus_pyramid_sub_levels_);
          for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::Mat img_resized;
            cv::resize(pkf->img_undist_, img_resized,
                       cv::Size(pkf->gaus_pyramid_width_[l],
                                pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvMat2TorchTensor_Float32(img_resized,
                                                        device_type_);
          }
        }
      }

      // Prepare for training
      {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
        gaussians_->createFromPcd(scene_->cached_point_cloud_,
                                  scene_->cameras_extent_);
        for (const auto& [keyframe_id, keyframe] : historical_keyframes_)
          recent_window_importance_->pushKeyframe(
              keyframe, gaussians_->getXYZ().size(0));
        std::unique_lock<std::mutex> lock(mutex_settings_);
        gaussians_->trainingSetup(opt_params_);
      }

      // Invoke training once
      trainForOneIteration();

      // Finish initial mapping loop
      initial_mapped_ = true;
      break;
    } else if (pSLAM_->isShutDown()) {
      break;
    } else {
      // Initial conditions not satisfied
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // Second loop: Incremental gaussian mapping
  int SLAM_stop_iter = 0;
  while (!isStopped() &&
         !(normalMappingLimit() > 0 && mapping_iter_ >= normalMappingLimit())) {
    if ((ultrafast_rotation_.enabled || real_lod_.enabled) && pSLAM_->isShutDown()) {
      SLAM_stop_iter = getIteration();
      SLAM_ended_ = true;
      markUltrafastOnlineEnd();
      break;
    }
    if (global_gi_) global_gi_cycle_start_ = std::chrono::steady_clock::now();
    mapping_profiler_.begin(device_type_ == torch::kCUDA);
    {
      auto timing = mapping_profiler_.scope(MappingProfiler::Operations);
      // Check conditions for incremental mapping
      if (hasMetIncrementalMappingConditions()) {
        combineMappingOperations();
        if (cull_keyframes_) cullKeyframes();
      }
    }

    // Invoke training once
    trainForOneIteration();

    if (pSLAM_->isShutDown()) {
      SLAM_stop_iter = getIteration();
      SLAM_ended_ = true;
    }

    if (normalMappingLimit() > 0 && mapping_iter_ >= normalMappingLimit()) {
      SLAM_stop_iter = getIteration();
      SLAM_ended_ = true;
    }

    if (SLAM_ended_) break;
  }

  // Third loop: Tail gaussian optimization. A forced mapping budget counts
  // only completed KF mappings, so replay updates never affect the limit.
  // Freeze the online boundary; mode1 tail continues the online ratio cursor.
  if (real_lod_.enabled) real_lod_.endOnline();
  if (pSLAM_->isShutDown()) {
    if (ultrafast_rotation_.enabled) markUltrafastOnlineEnd();
    renderAndRecordAllKeyframes("_online_end");
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    savePly(result_dir_ / (std::to_string(getIteration()) + "_online_end") / "ply");
  }
  if (tail_optimization_schedule_.enabled) {
    if (pSLAM_->isShutDown()) beginForcedTail(getIteration());
    while (!isStopped() && !scene_->keyframes().empty() &&
           tail_optimization_schedule_.active &&
           !tail_optimization_schedule_.complete(mapping_iter_)) {
      trainForOneIteration();
    }
    if (tail_optimization_schedule_.active)
      std::cout << "[TailOptimization] event=end mapping_iter=" << mapping_iter_
                << " completed=" << tail_optimization_schedule_.completed(mapping_iter_)
                << " target=" << tail_optimization_schedule_.target_steps
                << " reason=" << (tail_optimization_schedule_.complete(mapping_iter_)
                    ? "complete" : (isStopped() ? "stopped" : "no_keyframes")) << '\n';
  } else if (!real_lod_.enabled && selector_max_mapping_iterations_ > 0) {
    std::cout << "[Gaussian Mapper]Forced mapping tail: mapping_iter="
              << mapping_iter_ << ", target="
              << selector_max_mapping_iterations_ << ", keyframes="
              << scene_->keyframes().size() << std::endl;
    while (!isStopped() && !scene_->keyframes().empty() &&
           (!ultrafast_rotation_.enabled || opt_params_.iterations_ == -1 ||
            getIteration() + 1 < opt_params_.iterations_) &&
           mapping_iter_ < selector_max_mapping_iterations_) {
      trainForOneIteration();
    }
  } else {
    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (!isStopped() && !scene_->keyframes().empty() &&
           (!ultrafast_rotation_.enabled || opt_params_.iterations_ == -1 ||
            getIteration() + 1 < opt_params_.iterations_) &&
           (getIteration() - SLAM_stop_iter < n_delay_iters ||
            getIteration() % densify_interval < n_delay_iters ||
            isKeepingTraining())) {
      trainForOneIteration();
    }
  }

  // Save and clear
  if(real_lod_.enabled)runRealLodOptimization();
  renderAndRecordAllKeyframes("_shutdown");
  savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
  writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

  std::cout << std::endl;
  std::cout << getIteration() << " iterations completed." << std::endl;
  if (selector_replay_enabled_)
    std::cout << "[ConflictReplay SUMMARY] would_cancel=" << gradient_would_cancel_count_
              << " fallback=" << gradient_fallback_count_ << " enqueued=" << replay_enqueued_count_
              << " attempts=" << replay_attempt_count_ << " committed=" << replay_commit_count_
              << " skipped=" << replay_skip_count_ << " tf_updates=" << replay_tf_updates_
              << " pending=" << conflict_replay_manager_->queueSize()
              << " unfinished_window=" << conflict_replay_manager_->pendingCancellationCount()
              << " shutdown_drain=OFF\n";

  reportUltrafastRotation("training_end");
  mapping_profiler_.report(mapping_iter_);
  signalStop();
}

void GaussianMapper::trainColmap() {
  if(real_lod_.enabled)
    throw std::invalid_argument("RealLoD online step matching requires SLAM; disable Training.real_lod for COLMAP");
  if (ultrafast_rotation_.enabled)
    throw std::invalid_argument("Ultrafast rotation requires online SLAM; COLMAP is offline");
  if (global_gi_config_.streaming())
    throw std::invalid_argument("Streaming GI requires ORB-SLAM3 MapPoint coverage; use recent/disabled for COLMAP training");
  // Prepare multi resolution images for training
  for (auto& kfit : scene_->keyframes()) {
    auto pkf = kfit.second;
    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());
    if (device_type_ == torch::kCUDA) {
      cv::cuda::GpuMat img_gpu;
      img_gpu.upload(pkf->img_undist_);
      pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
      for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        cv::cuda::GpuMat img_resized;
        cv::cuda::resize(img_gpu, img_resized,
                         cv::Size(pkf->gaus_pyramid_width_[l],
                                  pkf->gaus_pyramid_height_[l]));
        pkf->gaus_pyramid_original_image_[l] =
            tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
      }
    } else {
      pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
      for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        cv::Mat img_resized;
        cv::resize(pkf->img_undist_, img_resized,
                   cv::Size(pkf->gaus_pyramid_width_[l],
                            pkf->gaus_pyramid_height_[l]));
        pkf->gaus_pyramid_original_image_[l] =
            tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
      }
    }
  }

  // Prepare for training
  {
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
    gaussians_->createFromPcd(scene_->cached_point_cloud_,
                              scene_->cameras_extent_);
    std::unique_lock<std::mutex> lock(mutex_settings_);
    gaussians_->trainingSetup(opt_params_);
    this->initial_mapped_ = true;
  }

  // Main loop: gaussian splatting training
  while (!isStopped()) {
    // Invoke training once
    trainForOneIteration();

    if (selector_max_mapping_iterations_ > 0) {
      if (mapping_iter_ >= selector_max_mapping_iterations_) break;
    } else if (getIteration() >= opt_params_.iterations_) {
      break;
    }
  }

  // Tail gaussian optimization. In forced-budget mode this is a bounded
  // continuation when the normal loop ended before the requested KF count.
  if (selector_max_mapping_iterations_ > 0) {
    while (!isStopped() && !scene_->keyframes().empty() &&
           mapping_iter_ < selector_max_mapping_iterations_) {
      trainForOneIteration();
    }
  } else {
    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (getIteration() % densify_interval <= n_delay_iters ||
           isKeepingTraining()) {
      trainForOneIteration();
      densify_interval = densifyInterval();
      n_delay_iters = densify_interval * 0.8;
    }
  }

  // Save and clear
  renderAndRecordAllKeyframes("_shutdown");
  savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
  writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

  signalStop();
}

/**
 * @brief The training iteration body
 *
 */
void GaussianMapper::trainForOneIteration() {
  const bool slam_online = pSLAM_ && !pSLAM_->isShutDown();
  if(real_lod_.enabled && pSLAM_ && !slam_online)real_lod_.endOnline();
  if (ultrafast_rotation_.enabled && pSLAM_ && !slam_online)
    markUltrafastOnlineEnd();
  if (pSLAM_ && !slam_online && tail_optimization_schedule_.enabled &&
      !tail_optimization_schedule_.active)
    beginForcedTail(getIteration());
  if (tail_optimization_schedule_.active && tail_optimization_schedule_.complete(mapping_iter_)) {
    mapping_profiler_.cancel();
    global_gi_cycle_start_.reset();
    return;
  }
  // Preserve the online optimizer cap while waiting for actual SLAM shutdown.
  // Forced tail alone overrides it, so its real updates cannot become no-ops.
  if (ultrafast_rotation_.enabled && !tail_optimization_schedule_.active &&
      opt_params_.iterations_ != -1 && getIteration() + 1 >= opt_params_.iterations_) {
    mapping_profiler_.cancel();
    global_gi_cycle_start_.reset();
    if (slam_online) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return;
  }
  const auto mapping_start = global_gi_cycle_start_.value_or(std::chrono::steady_clock::now());
  mapping_profiler_.begin(device_type_ == torch::kCUDA);
  increaseIteration(1);
  std::shared_ptr<GaussianKeyframe> viewpoint_cam =
      useOneRandomSlidingWindowKeyframe();
  if (!viewpoint_cam) {
    increaseIteration(-1);
    mapping_profiler_.cancel();
    global_gi_cycle_start_.reset();
    return;
  }

  const bool mode1_tail = ultrafast_rotation_.enabled && ultrafast_rotation_.online_ended;
  mapping_profiler_.setPhase((!slam_online && pSLAM_ != nullptr) || mode1_tail);
  // A finite online optimization cap may have left uncommitted gradients.
  // Clear them before Replay/full/LOD backward; preserve all Adam state/clocks.
  if (tail_optimization_schedule_.active &&
      mapping_iter_ == tail_optimization_schedule_.start_mapping) {
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    gaussians_->optimizer_->zero_grad(true);
    if (selector_optimizer_) selector_optimizer_->zero_grad(true);
    if (transform_optimizer_) transform_optimizer_->zero_grad(true);
  }

  if (!mapping_budget_warning_printed_ &&
      normalMappingLimit() > 0 && mapping_iter_ < normalMappingLimit() &&
      opt_params_.iterations_ != -1 &&
      getIteration() >= opt_params_.iterations_) {
    std::cerr << "[Gaussian Mapper][WARNING] Forced mapping continues after "
                 "Optimization.iterations: iteration="
              << getIteration() << ", optimization_limit="
              << opt_params_.iterations_ << ", mapping_iter=" << mapping_iter_
              << ", mapping_target=" << selector_max_mapping_iterations_
              << ". Gaussian/Selector/TF normal optimizer steps and Gaussian "
                 "zero_grad are skipped; mapping_iter and periodic AABB checks "
                 "still advance, and Gaussian gradients may accumulate. "
                 "Stopping policy is unchanged. This warning is printed once."
              << std::endl;
    mapping_budget_warning_printed_ = true;
  }

  // Auxiliary work retains the real Mapping clock, independent of ratio slots.
  const bool enhanced = true;
  if (!ultrafast_rotation_.recovering() && shouldRunSelectorReplay()) replay_task_.pending = true;
  if (!ultrafast_rotation_.recovering() && replay_task_.take(enhanced) && shouldRunSelectorReplay(true)) {
    processConflictReplay(viewpoint_cam);
  }

  trainMappingIteration(viewpoint_cam, enhanced, false);
  if(real_lod_.enabled)real_lod_.completeFull();
  ++mapping_iter_;
  if (ultrafast_rotation_.enabled && pSLAM_ && pSLAM_->isShutDown()) markUltrafastOnlineEnd();
  if (ultrafast_rotation_.enabled && mapping_iter_ % 1000 == 0)
    reportUltrafastRotation("mapping");
  if (global_gi_) {
    gi_refresh_task_.observe(mapping_iter_, global_gi_config_.refresh_interval);
    if (gi_refresh_task_.take(enhanced)) {
      auto timing = mapping_profiler_.scope(MappingProfiler::GI);
      std::unique_lock<std::mutex> lock_render(mutex_render_);
      const bool gi_refreshed = global_gi_->refreshIfDue(mapping_iter_, [this](uint64_t id) {
        return renderGlobalGIContribution(id);
      }, true);
      if (gi_refreshed && geometry_marginal_) {
        geometry_marginal_->refresh(gaussians_->getXYZ(), gaussians_->getScalingActivation(),
            global_gi_->importance(), global_gi_->validMask(), mapping_iter_,
            global_gi_->version(), global_gi_->topologyVersion(), result_dir_,
            global_gi_config_.mode == "streaming_marginal_geo" || global_gi_config_.marginal_shadow,
            global_gi_config_.marginal_shadow, global_gi_config_.marginal_diagnostics);
      }
    }
  }
  if (dynamic_aabb_config_.enabled)
    aabb_task_.observe(mapping_iter_, dynamic_aabb_config_.check_interval);
  if (aabb_task_.take(enhanced)) {
    auto timing = mapping_profiler_.scope(MappingProfiler::AABB);
    evaluateDynamicAABB(true);
  }
  if (drift_stability_config_.enabled)
    historical_task_.observe(mapping_iter_, drift_stability_config_.probe_interval);
  if (historical_task_.take(enhanced)) {
    auto timing = mapping_profiler_.scope(MappingProfiler::Historical);
    evaluateDriftStability(true);
  }
  mapping_profiler_.end(mapping_iter_);
  if (global_gi_) {
    // Statistics read back only at reporting cadence, never every LOD forward.
    if (global_gi_config_.performance_logging && (mapping_iter_ == 1 || mapping_iter_ % 1000 == 0)) {
      std::unique_lock<std::mutex> lock_render(mutex_render_);
      global_gi_->recordGuidance(global_gi_->observedMask(), selector_enabled_ && selector_gi_loss_weight_ != 0.0f &&
          hasNormalOptimizationBudget());
    }
    global_gi_->recordMapping(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - mapping_start).count(), mapping_iter_);
  }
  global_gi_cycle_start_.reset();
}

void GaussianMapper::trainMappingIteration(
    std::shared_ptr<GaussianKeyframe> viewpoint_cam,
    bool enhanced, bool resumed_after_slam) {
  renderer_workspace::Scope workspace_scope(reuse_renderer_workspace_);
  ssim_backend::Scope ssim_scope(optimized_ssim_backward_);
  ssim_backend::CacheScope ssim_cache_scope(cached_ssim_statistics_);
  mapping_cuda::L1FusionScope l1_fusion_scope(fused_l1_loss_);
  auto iter_start_timing = std::chrono::steady_clock::now();
  const bool rotation = ultrafast_rotation_.enabled;
  const bool mode2 = ultra_light_mode_ == 2;
  const bool post_slam = resumed_after_slam || (pSLAM_ && pSLAM_->isShutDown()) ||
      (rotation && ultrafast_rotation_.online_ended);
  const bool mode1_tail = rotation && post_slam;
  const bool serial_full_lod = rotation && ultrafast_rotation_.serial_full_lod;
  const bool detach_lod_master = serial_full_lod && ultrafast_rotation_.serial_lod_detach_master;
  const bool two_stage = serial_full_lod || (mode1_tail && !ultrafast_rotation_.rotate_tail);
  if (mode1_tail) markUltrafastOnlineEnd();
  auto& mode2_counts = post_slam ? mode2_tail_ : mode2_online_;
  ++mode2_counts.mappings;
  // Keep the first N Mapping updates Full-only, including short sequences entering tail.
  const bool recovery_full_step = rotation &&
      (mapping_iter_<lod_start_iteration_ || ultrafast_rotation_.recovering());
  const bool full_step = recovery_full_step || (mode1_tail && !ultrafast_rotation_.rotate_tail) || !rotation || ultrafast_rotation_.full();
  const auto render_total_gaussians = gaussians_->getXYZ().size(0);
  const auto phase = mode1_tail && two_stage ? ultrafast_rotation_.tailRatioIndex() : ultrafast_rotation_.phase();
  if (rotation) {
    if (!ultrafast_training_timer_)
      ultrafast_training_timer_ = std::make_unique<UltrafastTrainingTimer>(device_type_ == torch::kCUDA, mapping_iteration_logging_);
    ultrafast_training_timer_->begin();
  }
  auto* rotation_timer = rotation ? ultrafast_training_timer_.get() : nullptr;
  bool densify_stats_updated = false, densify_triggered = false;
  bool selector_committed = false, tf_committed = false;
  bool tail_full_committed = false, tail_lod_committed = false, tail_lod_attempted = false;
  LodGradientResult lod;

  writeKeyframeUsedTimes(result_dir_ / "used_times");

  // if (isdoingInactiveGeoDensify() &&
  // !viewpoint_cam->done_inactive_geo_densify_)
  //     increasePcdByKeyframeInactiveGeoDensify(viewpoint_cam);

  int training_level = num_gaus_pyramid_sub_levels_;
  int image_height, image_width;
  torch::Tensor gt_image, mask;
  if (isdoingGausPyramidTraining() && !(mode1_tail && !full_step))
    training_level = viewpoint_cam->getCurrentGausPyramidLevel();
  if (training_level == num_gaus_pyramid_sub_levels_) {
    image_height = viewpoint_cam->image_height_;
    image_width = viewpoint_cam->image_width_;
    gt_image = viewpoint_cam->original_image_.cuda();
    mask = undistort_mask_[viewpoint_cam->camera_id_];
  } else {
    image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
    image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
    gt_image =
        viewpoint_cam->gaus_pyramid_original_image_[training_level].cuda();
    mask = scene_->cameras_.at(viewpoint_cam->camera_id_)
               .gaus_pyramid_undistort_mask_[training_level];
  }

  // Mutex lock for usage of the gaussian model
  std::unique_lock<std::mutex> lock_render(mutex_render_);

  // Every 1000 its we increase the levels of SH up to a maximum degree
  if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
    default_sh_ += 1;
  gaussians_->setShDegree(default_sh_);
  if (global_gi_) {
    global_gi_->setMappingIteration(mapping_iter_ + 1);
    if (enhanced && !resumed_after_slam) {
      auto timing = mapping_profiler_.scope(MappingProfiler::GI);
      updateGlobalGIBank();
    }
  }
  // An undefined gradient makes shared Adam skip parameters and their moments.
  // Zero-valued gradients would still update S/TF using previous momentum.
  if (rotation || mode2) gaussians_->optimizer_->zero_grad(true);
  if (!enhanced || (rotation && full_step) || mode2) {
    if (selector_optimizer_) selector_optimizer_->zero_grad(true);
    if (transform_optimizer_) transform_optimizer_->zero_grad(true);
  }

  // No outstanding backward/optimizer work; render lock also excludes CD.
  // Append/clone/split mark the bank pending; no full scaffold rebuild per step.
  if (selector_anchor_adaptive_ && selector_network_)
    selector_network_->anchors()->refine(gaussians_->getXYZ(),
        {gaussians_->optimizer_.get(), selector_optimizer_.get()});

  if (protection_config_.performance_logging) gaussians_->protection_.profiler->beginIteration();
  gaussians_->protection_.mapping_iteration = static_cast<int64_t>(mapping_iter_ + 1);

  // Update learning rate
  gaussians_->updateLearningRate(gradient_constraint_config_.enabled
      ? static_cast<int>(normal_update_clock_.gaussianLRStep(getIteration()))
      : getIteration());
  gaussians_->setFeatureLearningRate(featureLearningRate());
  gaussians_->setOpacityLearningRate(opacityLearningRate());
  gaussians_->setScalingLearningRate(scalingLearningRate());
  gaussians_->setRotationLearningRate(rotationLearningRate());

  // Render
  if (getIteration() % 1000 == 0)
    std::cout << "[Gaussian Mapper] Rendering iter=" << getIteration()
              << " gaussians=" << gaussians_->getXYZ().sizes()[0] << std::endl;
  const bool collect_importance = full_step && enhanced && recent_window_importance_ &&
      recent_window_config_.collect_importance &&
      recent_window_importance_->enabled() &&
      recent_window_importance_->contains(viewpoint_cam->fid_) &&
      training_level == num_gaus_pyramid_sub_levels_;
  const bool collect_contribution=full_step && training_level==num_gaus_pyramid_sub_levels_ &&
      gaussians_->protection_.captureContribution();
  torch::Tensor full_contribution;int64_t contribution_pixels=0;
  const bool collect_anchor_si = full_step && anchor_si_.captureNextFull();
  RenderPackage render_pkg;
  torch::Tensor Ll1, full_loss;
  if (full_step) {
    ++mode2_counts.full_renders;
    {
      auto timing = mapping_profiler_.scope(MappingProfiler::FullRender);
      if (rotation_timer) rotation_timer->beginRender();
      anchor_si::CaptureScope anchor_capture(collect_anchor_si);
      auto& state=gaussians_->protection_;
      rendering_contribution::CaptureScope contribution_capture(collect_contribution,
          protection_config_.contribution_pixel_stride,
          collect_contribution?(state.contribution_full_calls-1)/protection_config_.contribution_every_full:0);
      render_pkg = GaussianRenderer::render(
          viewpoint_cam, image_height, image_width, gaussians_, pipe_params_,
          background_, override_color_, 1.0f, false, torch::Tensor(), false,
          collect_importance);
      if(collect_contribution) {
        full_contribution=contribution_capture.sums;
        const int stride=contribution_capture.stride;
        contribution_pixels=int64_t((image_width-1-contribution_capture.offset_x)/stride+1)*
            ((image_height-1-contribution_capture.offset_y)/stride+1);
      }
      if (rotation_timer) rotation_timer->endRender();
    }
    if (selector_explicit_anchor_blocks_)
      gaussians_->addAnchorPruningStats(render_pkg.visibility_indices);
    if(collect_anchor_si)anchor_si_.observe(selector_network_,render_pkg.frame_importance,
        gaussians_->getOpacityActivation().detach(),int64_t(image_height)*image_width,
        topology_version_,selector_target_ratios_);
    if (collect_importance && recent_window_importance_->enabled() &&
        render_pkg.frame_importance.defined() &&
        render_pkg.contribution_count.defined() &&
        render_pkg.frame_importance.numel() > 0) {
      auto timing = mapping_profiler_.scope(MappingProfiler::GI);
      refreshRecentWindowContribution(viewpoint_cam,
                                      render_pkg.frame_importance,
                                      render_pkg.contribution_count);
    }
    {
      auto timing = mapping_profiler_.scope(MappingProfiler::FullLoss);
      auto l1_loss = opt_params_.smooth_l1_ ? loss_utils::smooth_l1_loss : loss_utils::l1_loss;
      Ll1 = l1_loss(render_pkg.image, gt_image, 1.0f);
      auto Lssim = loss_utils::fast_ssim(render_pkg.image, gt_image);
      const float lambda_dssim = lambdaDssim();
      full_loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - Lssim);
      if (opt_params_.opacity_reg_)
        full_loss += opt_params_.opacity_reg_ * gaussians_->getOpacityActivation().abs().mean();
    }
    auto timing = mapping_profiler_.scope(MappingProfiler::FullBackward);
    full_loss.backward();
    ++mode2_counts.full_backwards;
    // Topology callbacks remap this evidence before the mode2 Gaussian commit.
    if (mode2) gaussians_->protection_.markTouched(render_pkg.visibility_indices);
  }
  bool has_selector_update = false;
  if (!real_lod_.enabled && !mode2 && enhanced && (!rotation || !full_step)) {
    const auto ratio_index = rotation ? phase :
        (selector_target_ratios_.empty() ? 0 : selector_step_ % selector_target_ratios_.size());
    if (selector_network_ && selector_optimizer_ && hasNormalOptimizationBudget() &&
        !selector_target_ratios_.empty()) {
      ++mode2_counts.lod_attempts;
      ++mode2_counts.ratios.at(ratio_index);
      tail_lod_attempted = mode1_tail;
    }
    auto timing = mapping_profiler_.scope(MappingProfiler::LOD);
    lod = accumulateLodGradients(
        viewpoint_cam, image_height, image_width, gt_image,
        ratio_index,
        rotation ? false : selector_detach_gaussian_parameters_, true, rotation_timer, false, mode1_tail || (rotation && selector_lod_render_scale_==1.f));
    has_selector_update = lod.trained;
    mode2_counts.lod_renders += lod.rendered;
  }

  {
    torch::NoGradGuard no_grad;
    bool gaussian_committed = false;
    auto commit_optimizer = [&] {
      if (mode2) {
        if (hasNormalOptimizationBudget()) {
          auto timing = mapping_profiler_.scope(MappingProfiler::GaussianOptimizer);
          ++mode2_counts.gaussian_step_calls;
          gaussian_committed = commitMode2Stage(true).gaussian;
          mode2_counts.gaussian_updates += gaussian_committed;
        }
        if (gaussian_committed)
          observeFullMaturity(render_pkg.radii,viewpoint_cam->fid_,full_contribution,contribution_pixels);
        else
          gaussians_->protection_.touched_in_cycle.zero_();
        gaussians_->optimizer_->zero_grad(true);
        return;
      }
      if (!hasNormalOptimizationBudget()) return;
      ++mode2_counts.gaussian_step_calls;
      bool gaussian_had_gradient = false;
      for (std::size_t g = 0; g < 6; ++g) {
        const auto& group = gaussians_->optimizer_->param_groups()[g];
        if (static_cast<const torch::optim::AdamOptions&>(group.options()).lr() == 0.0) continue;
        for (const auto& p : group.params()) gaussian_had_gradient |= p.numel() > 0 && p.grad().defined();
      }
      auto timing = mapping_profiler_.scope(two_stage ? MappingProfiler::GaussianOptimizer : MappingProfiler::Optimizer);
      bool has_gaussian_gradient = !rotation;
      if (rotation)
        for (size_t group_index = 0; group_index < 6; ++group_index) {
          const auto& group = gaussians_->optimizer_->param_groups()[group_index];
          if (static_cast<const torch::optim::AdamOptions&>(group.options()).lr() == 0.0) continue;
          for (const auto& parameter : group.params()) has_gaussian_gradient |= parameter.grad().defined();
        }
      const auto tf_before = normal_update_clock_.tf_updates;
      const auto rejected_before = normal_update_clock_.rejected_updates;
      bool selector_had_gradient = false;
      if (shared_selector_optimizer_configured_) {
        const auto& group = gaussians_->optimizer_->param_groups()[shared_selector_group_index_];
        if (static_cast<const torch::optim::AdamOptions&>(group.options()).lr() != 0.0)
          for (const auto& parameter : group.params()) selector_had_gradient |= parameter.grad().defined();
      }
      if (enhanced) gaussian_committed = stepWithHistoricalGuard(viewpoint_cam, !two_stage);
      else {
        gaussians_->optimizer_->step();
        normal_update_clock_.record(true, false);
        gaussian_committed = true;
      }
      gaussian_committed &= has_gaussian_gradient;
      if (two_stage) tail_full_committed = gaussian_committed;
      if (rotation && gaussian_committed) {
        gaussians_->protection_.markTouched(full_step ? render_pkg.visibility_indices : lod.supervised_indices);
        // Observe after the actual Gaussian commit, before Full topology changes.
        if (full_step && !protection_config_.fixed_maturity_one)
          observeFullMaturity(render_pkg.radii,viewpoint_cam->fid_,full_contribution,contribution_pixels);
      }
      selector_committed = selector_had_gradient &&
          normal_update_clock_.rejected_updates == rejected_before;
      tf_committed = normal_update_clock_.tf_updates > tf_before;
      mode2_counts.gaussian_updates += gaussian_committed && gaussian_had_gradient;
      mode2_counts.selector_commits += selector_committed;
      mode2_counts.tf_commits += tf_committed;
      if (rotation ? selector_committed : has_selector_update) ++selector_step_;
      gaussians_->optimizer_->zero_grad(true);
    };
    // Topology rebuilds Gaussian parameters without copying current gradients.
    // Rotation must commit Full calibration first; other modes keep old order.
    if (rotation) commit_optimizer();
    if (full_step) {
      const float full_loss_value = full_loss.item().toFloat();
      kfs_loss_[viewpoint_cam->fid_] = full_loss_value;
      ema_loss_for_log_ = 0.4f * full_loss_value + 0.6 * ema_loss_for_log_;
    }

    if (full_step && keyframe_record_interval_ &&
        getIteration() % keyframe_record_interval_ == 0)
      recordKeyframeRendered(render_pkg.image, gt_image, viewpoint_cam->fid_,
                             result_dir_, result_dir_, result_dir_);

    // Full-render GI and Selector work for the current topology is complete.
    // Every append/prune is mirrored by the active GI cache under this lock.
    // Densification
    const bool topology_window = getIteration() < opt_params_.densify_until_iter_ ||
        opt_params_.densify_until_iter_ == -1;
    if (rotation && topology_window) {
      if (getIteration() > opt_params_.densify_from_iter_)
        ultrafast_densify_task_.observe(getIteration(), densifyInterval());
      ultrafast_opacity_task_.observe(getIteration(), opacityResetInterval());
      if (model_params_.white_background_ && getIteration() == opt_params_.densify_from_iter_)
        ultrafast_opacity_task_.pending = true;
    }
    if (full_step && topology_window) {
      auto timing = mapping_profiler_.scope(MappingProfiler::Densify);
      // Keep track of max radii in image-space for pruning
      gaussians_->max_radii2D_.index_put_(
          {render_pkg.visibility_indices},
          torch::max(gaussians_->max_radii2D_.index({render_pkg.visibility_indices}),
                     render_pkg.radii.index({render_pkg.visibility_indices})));

      gaussians_->addDensificationStats(render_pkg.viewspace_points,
                                        render_pkg.visibility_indices);
      if(selector_explicit_anchor_blocks_ && getIteration()%densifyInterval()<5)
        std::cout<<"[Explicit Full Gradient] iteration="<<getIteration()<<" width="<<image_width
                 <<" height="<<image_height<<" pyramid_level="<<training_level
                 <<" visible="<<render_pkg.visibility_indices.numel()<<'\n';
      densify_stats_updated = true;
      ++mode2_counts.stats;

      if (rotation ? ultrafast_densify_task_.take(true) :
          ((getIteration() > opt_params_.densify_from_iter_) &&
           (getIteration() % densifyInterval() == 0))) {
        int size_threshold = (getIteration() < prune_big_point_after_iter_ ||
                              prune_big_point_after_iter_ == -1)
                                 ? 0
                                 : 20;
        gaussians_->densifyAndPrune(densifyGradThreshold(),
                                    densify_min_opacity_,  // 0.005,//
                                    scene_->cameras_extent_, size_threshold, densifyInterval());
        densify_triggered = true;
        if (rotation) ultrafast_rotation_.beginRecovery();
        ++mode2_counts.topology;
        if (recent_window_importance_ && recent_window_importance_->enabled())
          recent_window_importance_->assertAligned(gaussians_->getXYZ());
        if (global_gi_) global_gi_->assertAligned(gaussians_->getXYZ());
      }

      if (rotation ? ultrafast_opacity_task_.take(true) : (opacityResetInterval() &&
          (getIteration() % opacityResetInterval() == 0 ||
           (model_params_.white_background_ &&
             getIteration() == opt_params_.densify_from_iter_)))) {
        gaussians_->resetOpacity();
        ++mode2_counts.opacity_resets;
      }
    }

    auto iter_end_timing = std::chrono::steady_clock::now();
    auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                         iter_end_timing - iter_start_timing)
                         .count();

    // Log and save
    if (full_step && training_report_interval_ &&
        (getIteration() % training_report_interval_ == 0))
      trainingReport(getIteration(), opt_params_.iterations_, Ll1, full_loss,
                     ema_loss_for_log_, iter_time, *gaussians_, *scene_,
                     pipe_params_, background_);
    if ((!rotation && all_keyframes_record_interval_ &&
         getIteration() % all_keyframes_record_interval_ == 0)) {
      renderAndRecordAllKeyframes();
      savePly(result_dir_ / std::to_string(getIteration()) / "ply");
    }

    if (loop_closure_iteration_) loop_closure_iteration_ = false;

    // Non-rotation modes preserve the original optimizer/topology ordering.
    if (!rotation) commit_optimizer();
    if (full_step) prepareElasticEntryState();
    if (!mode2 && !two_stage && protection_config_.performance_logging) gaussians_->protection_.profiler->endIteration();
    if (rotation && !two_stage) {
      rotation_timer->end();
      if (mapping_iteration_logging_) {
        std::ostringstream line;
        line << std::fixed << std::setprecision(4) << "[UltrafastRotation] normal_training_step="
             << ultrafast_rotation_.step << " mapping_iteration=" << mapping_iter_ + 1
             << " phase=" << (recovery_full_step ? "FULL_RECOVERY" : ultrafast_rotation_.phaseName(phase))
             << " ratio=" << (recovery_full_step ? 1.0f : ultrafast_rotation_.ratio()) << " keyframe_id=" << viewpoint_cam->fid_
             << " is_lod_step=" << !full_step << " is_full_step=" << full_step
             << " post_slam=" << post_slam
             << " full_height=" << (full_step ? image_height : 0)
             << " full_width=" << (full_step ? image_width : 0)
             << " lod_height=" << lod.render_height << " lod_width=" << lod.render_width
             << " gaussian_optimizer_step=" << gaussian_committed
             << " selector_optimizer_step=" << selector_committed << " tf_optimizer_step=" << tf_committed
             << " kfs_loss_updated=" << full_step << " densify_stats_updated=" << densify_stats_updated
             << " densify_triggered=" << densify_triggered << " prune_triggered=" << densify_triggered
             << " active_gaussians=" << (full_step ? render_total_gaussians : lod.active_gaussians)
             << " tf_transformed_gaussians=" << lod.transformed_gaussians
             << " selector_compensated_gaussians=" << lod.compensated_gaussians
             << " total_gaussians=" << render_total_gaussians
             << " post_topology_gaussians=" << gaussians_->getXYZ().size(0)
             << " loss=" << (full_step ? full_loss.item<double>() : lod.loss)
             << " training_ms=" << rotation_timer->wall_ms << " render_ms=" << rotation_timer->render_ms
             << " cuda_elapsed_ms=" << rotation_timer->cuda_elapsed_ms;
        std::cout << line.str() << '\n';
      }
      if (recovery_full_step)
        ultrafast_rotation_.completeRecovery(rotation_timer->wall_ms,mapping_iteration_logging_);
      else if (mode1_tail)
        ultrafast_rotation_.completeRotatingTail(rotation_timer->wall_ms, rotation_timer->render_ms,
                                  full_step || lod.rendered, tail_lod_attempted,
                                  selector_committed, tf_committed, mapping_iteration_logging_);
      else ultrafast_rotation_.complete(rotation_timer->wall_ms, rotation_timer->render_ms,
                                  full_step || lod.rendered, selector_committed, tf_committed,
                                  mapping_iteration_logging_);
      if (all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0) {
        renderAndRecordAllKeyframes();
        savePly(result_dir_ / std::to_string(getIteration()) / "ply");
      }
    }
  }
  // Full's NoGrad scope is over, but the same render mutex remains held.
  // Re-read all attributes/indices after topology and Full evidence finish.
  if (two_stage) {
    tail_lod_attempted = !recovery_full_step &&
        !(densify_triggered && ultrafast_rotation_.post_topology_full_steps) &&
        selector_enabled_ && selector_network_ && selector_optimizer_ &&
        hasNormalOptimizationBudget() && !selector_target_ratios_.empty() &&
        gaussians_->getXYZ().size(0) > 0;
    if (tail_lod_attempted) {
      if (selector_anchor_adaptive_ && selector_network_)
        selector_network_->anchors()->refine(gaussians_->getXYZ(),
            {gaussians_->optimizer_.get(), selector_optimizer_.get()});
      ++mode2_counts.lod_attempts;
      ++mode2_counts.ratios.at(phase);
      {
        auto timing = mapping_profiler_.scope(MappingProfiler::LOD);
        lod = accumulateLodGradients(viewpoint_cam, image_height, image_width, gt_image,
                                    phase, detach_lod_master, true, rotation_timer, false, mode1_tail || selector_lod_render_scale_==1.f);
      }
      mode2_counts.lod_renders += lod.rendered;
      if (lod.finite_loss) {
        torch::NoGradGuard no_grad;
        bool gaussian_had_gradient = false, selector_had_gradient = false;
        for (std::size_t g = 0; g < gaussians_->optimizer_->param_groups().size(); ++g) {
          const auto& group = gaussians_->optimizer_->param_groups()[g];
          if (static_cast<const torch::optim::AdamOptions&>(group.options()).lr() == 0.0) continue;
          for (const auto& p : group.params()) if (p.numel() && p.grad().defined()) {
            gaussian_had_gradient |= g < 6;
            selector_had_gradient |= shared_selector_optimizer_configured_ && g == shared_selector_group_index_;
          }
        }
        TORCH_CHECK(!detach_lod_master || !gaussian_had_gradient,
                    "Serial LoD leaked gradient into Master Gaussian");
        const auto tf_before = normal_update_clock_.tf_updates;
        auto timing = mapping_profiler_.scope(MappingProfiler::LodOptimizer);
        bool committed = false;
        tail_lod_committed = stepWithHistoricalGuard(viewpoint_cam, false, &committed) && gaussian_had_gradient;
        selector_committed = committed && selector_had_gradient;
        tf_committed = normal_update_clock_.tf_updates > tf_before;
        if (tail_lod_committed) gaussians_->protection_.markTouched(lod.supervised_indices);
        selector_step_ += selector_committed;
        mode2_counts.gaussian_updates += tail_lod_committed;
        if (!detach_lod_master) ++mode2_counts.gaussian_step_calls;
        mode2_counts.selector_commits += selector_committed;
        mode2_counts.tf_commits += tf_committed;
      } else {
        ++mode2_counts.lod_nonfinite;
      }
      gaussians_->optimizer_->zero_grad(true);
    }
    // Both stages share one Mapping LR decision; a rejected LoD cannot undo Full.
    if (hasNormalOptimizationBudget())
      normal_update_clock_.record(tail_full_committed || tail_lod_committed, false);
    if (protection_config_.performance_logging) gaussians_->protection_.profiler->endIteration();
    rotation_timer->end();
    if (mapping_iteration_logging_)
      std::cout << "[UltrafastRotation] normal_training_step=" << ultrafast_rotation_.step
                << " mapping_iteration=" << mapping_iter_ + 1 << " phase=FULL_LOD"
                << " ratio=" << selector_target_ratios_.at(phase) << " keyframe_id=" << viewpoint_cam->fid_
                << " is_full_step=1 is_lod_step=" << lod.rendered
                << " full_gaussian_optimizer_step=" << tail_full_committed
                << " lod_gaussian_optimizer_step=" << tail_lod_committed
                << " selector_optimizer_step=" << selector_committed << " tf_optimizer_step=" << tf_committed
                << " full_height=" << image_height << " full_width=" << image_width
                << " lod_height=" << lod.render_height << " lod_width=" << lod.render_width
                << " full_loss=" << full_loss.item<double>() << " lod_loss=" << lod.loss
                << " training_ms=" << rotation_timer->wall_ms << " render_ms=" << rotation_timer->render_ms << '\n';
    if (mode1_tail)
      ultrafast_rotation_.completeTail(rotation_timer->wall_ms, tail_lod_attempted, lod.rendered,
                                      selector_committed, tf_committed, mapping_iteration_logging_);
    else
      ultrafast_rotation_.completeSerial(rotation_timer->wall_ms, rotation_timer->render_ms, lod.rendered,
                                        selector_committed, tf_committed, mapping_iteration_logging_);
    if (two_stage) {
      std::ofstream trace(result_dir_ / "serial_full_lod_trace.csv",std::ios::app);
      if (trace.tellp()==0)
        trace<<"mapping,post_slam,full,lod,full_gaussian_update,lod_gaussian_update,selector_update,tf_update,ratio_index,gaussians,topology,full_height,full_width,lod_height,lod_width\n";
      trace<<mapping_iter_+1<<','<<post_slam<<",1,"<<lod.rendered<<','<<tail_full_committed<<','<<tail_lod_committed<<','
           <<selector_committed<<','<<tf_committed<<','<<phase<<','<<gaussians_->getXYZ().size(0)<<','<<densify_triggered<<','
           <<image_height<<','<<image_width<<','<<lod.render_height<<','<<lod.render_width<<'\n';
    }
    if (all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0) {
      renderAndRecordAllKeyframes();
      savePly(result_dir_ / std::to_string(getIteration()) / "ply");
    }
  }
  if (mode2) {
    const bool lod_ready = selector_enabled_ && selector_network_ &&
        hasNormalOptimizationBudget() && !selector_target_ratios_.empty() &&
        gaussians_->getXYZ().size(0) > 0;
    if (lod_ready) {
      const auto ratio_index = normal_lod_branch_count_ % selector_target_ratios_.size();
      ++mode2_counts.lod_attempts;
      ++mode2_counts.ratios.at(ratio_index);
      {
        auto timing = mapping_profiler_.scope(MappingProfiler::LOD);
        lod = accumulateLodGradients(viewpoint_cam, image_height, image_width, gt_image,
                                    ratio_index, true, true, nullptr, true);
      }
      mode2_counts.lod_renders += lod.rendered;
      if (lod.finite_loss) {
        auto timing = mapping_profiler_.scope(MappingProfiler::LodOptimizer);
        const auto commit = commitMode2Stage(false);
        selector_committed = commit.selector;
        tf_committed = commit.tf;
        mode2_counts.lod_nonfinite += !commit.finite;
        selector_step_ += selector_committed;
        normal_update_clock_.tf_updates += tf_committed;
        mode2_counts.selector_commits += selector_committed;
        mode2_counts.tf_commits += tf_committed;
      } else {
        ++mode2_counts.lod_nonfinite;
      }
      gaussians_->optimizer_->zero_grad(true);
      ++normal_lod_branch_count_;
    }
    if (protection_config_.performance_logging) gaussians_->protection_.profiler->endIteration();
    if (getIteration() % 1000 == 0) reportMode2Mapping("mapping");
  }
  if (!mode2 && getIteration() % 1000 == 0) reportMode2Mapping("mapping");
  if (rotation && ultrafast_rotation_.post_topology_full_steps) {
    if (recovery_full_step) {
      ++ultrafast_rotation_.recovery_steps;
      // A new topology pass restarts a full window after this Mapping.
      if (!densify_triggered && ultrafast_rotation_.recovery_remaining>0) --ultrafast_rotation_.recovery_remaining;
    }
    std::ofstream trace(result_dir_ / "post_topology_full_trace.csv",std::ios::app);
    if (trace.tellp()==0)
      trace<<"mapping,topology_triggered,recovery,full,lod,selector_update,tf_update,remaining,ratio_index,gaussians\n";
    trace<<mapping_iter_+1<<','<<densify_triggered<<','<<recovery_full_step<<','<<full_step<<','<<lod.rendered<<','
         <<selector_committed<<','<<tf_committed<<','<<ultrafast_rotation_.recovery_remaining<<','<<phase<<','
         <<gaussians_->getXYZ().size(0)<<'\n';
    if (densify_triggered || (recovery_full_step && !ultrafast_rotation_.recovering()))
      std::cout<<"[Post Topology Full] mapping="<<mapping_iter_+1<<" triggered="<<densify_triggered
               <<" remaining="<<ultrafast_rotation_.recovery_remaining<<" completed="<<ultrafast_rotation_.recovery_steps<<'\n';
  }
}

GaussianMapper::MappingStageCommit GaussianMapper::commitMode2Stage(bool gaussian_stage) {
  torch::NoGradGuard no_grad;
  MappingStageCommit result;
  auto& optimizer = *gaussians_->optimizer_;
  std::vector<torch::Tensor> finite_checks;
  for (std::size_t g = 0; g < optimizer.param_groups().size(); ++g) {
    const auto& group = optimizer.param_groups()[g];
    const bool gaussian = g < 6;
    for (const auto& p : group.params()) {
      if (!p.grad().defined()) continue;
      TORCH_CHECK(gaussian == gaussian_stage, "Mode2 cross-stage gradient in group ", g);
      if (gaussian) {
        result.gaussian |= p.numel() > 0 && static_cast<const torch::optim::AdamOptions&>(group.options()).lr() != 0.0;
      } else {
        finite_checks.push_back(torch::isfinite(p.grad()).all());
        result.selector |= shared_selector_optimizer_configured_ && g == shared_selector_group_index_;
        result.tf |= shared_transform_optimizer_configured_ &&
            (g == shared_transform_mlp_group_index_ || g == shared_transform_grid_group_index_);
      }
    }
  }
  if (gaussian_stage) {
    // Preserve original before-step topology, including undefined new grads.
    optimizer.step();
  } else {
    if (!finite_checks.empty() && !torch::stack(finite_checks).all().item<bool>()) {
      result.finite = false; result.selector = result.tf = false;
      return result;
    }
    auto cuda_groups = result.tf && transform_field_ && transform_field_->config().backend == "cuda" &&
        shared_transform_optimizer_configured_
        ? std::vector<std::size_t>{shared_transform_mlp_group_index_, shared_transform_grid_group_index_}
        : std::vector<std::size_t>{};
    if (result.selector && selector_backend_ == "cuda" && shared_selector_optimizer_configured_)
      cuda_groups.push_back(shared_selector_group_index_);
    tf_cuda::tfCudaAdamMixedStep(optimizer, cuda_groups, gaussians_->protection_.timing());
  }
  return result;
}

void GaussianMapper::reportMode2Mapping(const char* reason) const {
  for (int phase = 0; phase < 2; ++phase) {
    const auto& c = phase ? mode2_tail_ : mode2_online_;
    std::cout << "[Mapping STAGES] mode=" << ultra_light_mode_ << " reason=" << reason << " phase=" << (phase ? "tail" : "online")
              << " mappings=" << c.mappings << " full_renders=" << c.full_renders
              << " full_backwards=" << c.full_backwards << " gaussian_step_calls=" << c.gaussian_step_calls
              << " gaussian_updates=" << c.gaussian_updates << " stats=" << c.stats
              << " topology=" << c.topology << " opacity_resets=" << c.opacity_resets
              << " lod_attempts=" << c.lod_attempts << " lod_renders=" << c.lod_renders
              << " selector_commits=" << c.selector_commits << " tf_commits=" << c.tf_commits
              << " lod_nonfinite=" << c.lod_nonfinite << " ratio_counts=";
    for (std::size_t i = 0; i < c.ratios.size(); ++i) std::cout << (i ? "," : "") << c.ratios[i];
    std::cout << " normal_lod_branch_count=" << normal_lod_branch_count_ << '\n';
  }
}

bool GaussianMapper::shouldRunSelectorReplay(bool deferred) const {
  return selector_replay_enabled_ && selector_network_ && selector_optimizer_ &&
         conflict_replay_manager_ && conflict_replay_manager_->queueSize() > 0 &&
         selector_replay_interval_ > 0 && selector_replay_num_frames_ > 0 &&
         mapping_iter_ > 0 &&
          (deferred || mapping_iter_ %
                  static_cast<std::uint64_t>(selector_replay_interval_) ==
              0);
}

void GaussianMapper::markUltrafastOnlineEnd() {
  if (!ultrafast_rotation_.enabled || ultrafast_rotation_.online_ended) return;
  ultrafast_rotation_.endOnline(mapping_iter_);
  std::cout << "[UltrafastRotation ONLINE_END] online_end_mapping_iteration="
            << ultrafast_rotation_.online_end_mapping_iteration
            << " online_end_rotation_step=" << ultrafast_rotation_.online_end_rotation_step
            << " tail_training_steps=" << ultrafast_rotation_.tailSteps() << '\n';
}

void GaussianMapper::reportUltrafastRotation(const char* event) const {
  if (!ultrafast_rotation_.enabled) return;
  const auto& r = ultrafast_rotation_;
  TORCH_CHECK(r.balanced() &&
              (r.serial_full_lod ? r.full_render_count == r.step :
               r.full_render_count + r.lod_render_count == r.step + (r.rotate_tail ? 0 : r.tail_lod_renders)),
              "Ultrafast rotation invariant failed: balance/render count");
  std::ostringstream line;
  line << std::fixed << std::setprecision(4) << "[UltrafastRotation SUMMARY] event=" << event;
  for (std::size_t i = 0; i < r.counts.size(); ++i) {
    line << " ultrafast_" << r.countKey(i) << "_steps=" << r.counts[i];
    if (mapping_iteration_logging_)
      line << ' ' << r.countKey(i) << "_mean_ms=" << (r.counts[i] ? r.elapsed_sum[i] / r.counts[i] : 0.0)
           << ' ' << r.countKey(i) << "_render_mean_ms=" << (r.counts[i] ? r.render_sum[i] / r.counts[i] : 0.0);
  }
  if (mapping_iteration_logging_)
    line << " mean_mapping_training_ms=" << r.mean() << " p95_mapping_training_ms=" << r.p95();
  line << " ultrafast_total_normal_steps=" << r.step << " balanced=" << r.balanced()
       << " recovery_steps=" << r.recovery_steps << " recovery_remaining=" << r.recovery_remaining
       << " full_render_count=" << r.full_render_count << " lod_render_count=" << r.lod_render_count
       << " online_ended=" << r.online_ended
       << " online_end_mapping_iteration=" << r.online_end_mapping_iteration
       << " online_end_rotation_step=" << r.online_end_rotation_step
       << " tail_training_steps=" << r.tailSteps()
       << " tail_full_render_count=" << r.tail_full_renders << " tail_lod_render_count=" << r.tail_lod_renders
       << " tail_lod_attempts=" << r.tail_lod_attempts
       << " selector_online_lod_update_count=" << (r.online_ended ? r.online_end_selector_updates : r.selector_updates)
       << " tf_online_lod_update_count=" << (r.online_ended ? r.online_end_tf_updates : r.tf_updates)
       << " selector_tail_lod_update_count=" << (r.online_ended ? r.selector_updates - r.online_end_selector_updates : 0)
       << " tf_tail_lod_update_count=" << (r.online_ended ? r.tf_updates - r.online_end_tf_updates : 0)
       << " tf_first_initialized_mapping_iteration=" << ultrafast_tf_first_initialized_
       << " tf_first_applied_lod_iteration=" << ultrafast_tf_first_applied_
       << " timing=" << (!mapping_iteration_logging_ ? "disabled" :
          (device_type_ == torch::kCUDA ? "synchronized_wall_and_cuda_event_render" : "cpu_wall"));
  std::cout << line.str() << '\n';
  if (r.online_ended) {
    std::cout << "[UltrafastRotation TAIL] policy=" << (r.serial_full_lod || !r.rotate_tail ? "full_then_lod" : "ratio_rotation") << " ratio_counts=";
    for (auto n : r.tail_ratio_counts) std::cout << n << ',';
    std::cout << " phase_counts=";
    for (auto n : r.tail_phase_counts) std::cout << n << ',';
    std::cout << '\n';
  }
}

void GaussianMapper::beginForcedTail(std::uint64_t iteration) {
  if (!tail_optimization_schedule_.begin(iteration, mapping_iter_)) return;
  std::cout << "[TailOptimization] event=begin slam_stop_iter=" << iteration
            << " start_mapping=" << mapping_iter_
            << " target=" << tail_optimization_schedule_.target_steps
            << " ignored_max_mapping_iterations=" << selector_max_mapping_iterations_ << '\n';
}

std::uint64_t GaussianMapper::normalMappingLimit() const {
  if(real_lod_.enabled)return 0;
  return tail_optimization_schedule_.mappingLimit(selector_max_mapping_iterations_, pSLAM_ != nullptr);
}

bool GaussianMapper::hasNormalOptimizationBudget() {
  if(real_lod_.enabled)return real_lod_.phase!=RealLodSchedule::Phase::Complete;
  return tail_optimization_schedule_.allowsOptimization(getIteration(), opt_params_.iterations_);
}

void GaussianMapper::enqueueConflictReplay(
    const std::shared_ptr<GaussianKeyframe>& current) {
  std::vector<ConflictReplayView> views;
  for (const auto& entry : scene_->getAllKeyframes()) {
    const auto& camera = entry.second;
    if (!camera || !camera->original_image_.defined() ||
        camera->original_image_.numel() == 0 ||
        !camera->getPosef().matrix().allFinite()) continue;
    const auto pose = camera->getPosef().inverse();
    ConflictReplayView view;
    view.keyframe_id = camera->fid_;
    for (int i = 0; i < 3; ++i) {
      view.center[i] = pose.translation()[i];
      for (int j = 0; j < 3; ++j)
        view.rotation[i * 3 + j] = pose.rotationMatrix()(i, j);
    }
    views.push_back(view);
  }
  auto excluded = recent_window_importance_ ? recent_window_importance_->keyframeIds()
                                           : std::vector<std::size_t>{};
  if (current) excluded.push_back(current->fid_);
  const auto batch = conflict_replay_manager_->enqueue(views, excluded);
  replay_enqueued_count_ += batch.enqueued.size();
  std::ostringstream line;
  line << std::fixed << std::setprecision(4) << "[ConflictReplay SELECT] iter=" << mapping_iter_
       << " window=" << batch.window_id << " would_cancel=" << batch.would_cancel_count;
  for (std::size_t s = 0; s < batch.sources.size(); ++s) {
    line << " source" << s << "=KF" << batch.sources[s].first << ':' << batch.sources[s].second
         << " views=";
    for (auto id : batch.selected_per_source[s]) line << id << ',';
    line << " distance(min/max)=" << batch.distance_ranges[s].first << '/'
         << batch.distance_ranges[s].second << " selected=" << batch.selected_per_source[s].size()
         << " reason=" << (batch.selected_per_source[s].size() < static_cast<std::size_t>(conflict_replay_config_.views_per_source)
                                ? "insufficient_remote_views" : "complete");
  }
  line << " enqueued=" << batch.enqueued.size() << " duplicates=" << batch.duplicates
       << " queue_full=" << batch.queue_full << " pending=" << conflict_replay_manager_->queueSize()
       << " missing_sources=" << conflict_replay_config_.source_view_count - batch.sources.size();
  std::cout << line.str() << '\n';
}

void GaussianMapper::processConflictReplay(
    const std::shared_ptr<GaussianKeyframe>& current) {
  const auto available = conflict_replay_manager_->queueSize();
  int attempted = 0;
  const auto recent = recent_window_importance_ ? recent_window_importance_->keyframeIds()
                                               : std::vector<std::size_t>{};
  for (std::size_t checked = 0; checked < available && attempted < selector_replay_num_frames_; ++checked) {
    const auto task = conflict_replay_manager_->pop();
    if (!task) break;
    auto camera = scene_->getKeyframe(task->keyframe_id);
    const bool invalid = !camera || !camera->original_image_.defined() ||
        camera->original_image_.numel() == 0 || !camera->getPosef().matrix().allFinite() ||
        (current && current->fid_ == task->keyframe_id) ||
        std::find(recent.begin(), recent.end(), task->keyframe_id) != recent.end();
    if (invalid) {
      ++replay_attempt_count_;
      ++replay_skip_count_;
      std::cout << "[ConflictReplay SKIP] iter=" << mapping_iter_ << " KF=" << task->keyframe_id
                << " reason=view_ineligible\n";
      continue;
    }
    ++attempted;
    trainJointReplay(*task, camera);
  }
}

void GaussianMapper::observeFullMaturity(const torch::Tensor& radii,int64_t keyframe,
    const torch::Tensor& contribution,int64_t sample_pixels) {
  auto& p=gaussians_->protection_;
  if(!p.config.reliability_v2) {
    p.observeFull(gaussians_->getXYZ(),gaussians_->scaling_,gaussians_->getRotationActivation());return;
  }
  std::vector<torch::Tensor> gradients{gaussians_->xyz_.grad(),gaussians_->features_dc_.grad(),gaussians_->features_rest_.grad(),
       gaussians_->opacity_.grad(),gaussians_->scaling_.grad(),gaussians_->rotation_.grad()};
  for(size_t g=0;g<6;++g)
    if(static_cast<const torch::optim::AdamOptions&>(gaussians_->optimizer_->param_groups()[g].options()).lr()==0)
      gradients[g]=torch::Tensor{};
  p.observeReliabilityFull(gaussians_->getXYZ(),gaussians_->scaling_,gaussians_->getRotationActivation(),radii,gradients,keyframe,contribution,sample_pixels);
}

void GaussianMapper::prepareElasticEntryState() {
  auto& protection = gaussians_->protection_;
  const auto& config = transform_field_config_;
  if (config.elastic_entry_maturity > 0) protection.indicesAbove(config.elastic_entry_maturity);
  if (config.start_mode == "maturity_population" &&
      (!transform_field_ || continuous_tf_statistics_.first_ready_population_iteration < 0)) {
    const auto count = protection.indicesAbove(config.ready_maturity).numel();
    if (protection.first_valid_maturity_cycle_iteration >= 0 && count >= config.min_ready_gaussians &&
        continuous_tf_statistics_.first_ready_population_iteration < 0) {
      continuous_tf_statistics_.first_ready_population_iteration = protection.mapping_iteration;
      std::cout << "[Elastic READY] mapping_iteration=" << protection.mapping_iteration
                << " ready_count=" << count << " ready_maturity=" << config.ready_maturity << '\n';
    }
  }
}

void GaussianMapper::initializeTransformFieldIfNeeded() {
  if (transform_field_ || !transform_field_config_.enabled || !selector_network_) return;
  const auto ready_count = transform_field_config_.start_mode == "maturity_population"
      ? gaussians_->protection_.indicesAbove(transform_field_config_.ready_maturity).numel() : -1;
  if (!transform_field_config_.ready(mapping_iter_,
          gaussians_->protection_.first_valid_maturity_cycle_iteration, ready_count)) return;
  const auto xyz = gaussians_->getXYZ().detach();
  if (xyz.numel() == 0) return;

  torch::NoGradGuard no_grad;
  const float domain_threshold = transform_field_config_.start_mode == "maturity_population"
      ? std::max(transform_field_config_.elastic_entry_maturity, transform_field_config_.ready_maturity)
      : transform_field_config_.elastic_entry_maturity;
  const auto mature_xyz = xyz.index({elasticDomainMask(xyz, gaussians_->protection_.maturity, domain_threshold)});
  if (mature_xyz.numel() == 0) return;
  if (domain_threshold > 0 && mature_xyz.size(0) < 2) return;
  auto aabb_min = std::get<0>(mature_xyz.min(/*dim=*/0));
  auto aabb_max = std::get<0>(mature_xyz.max(/*dim=*/0));
  auto center = (aabb_min + aabb_max) * 0.5;
  auto half_extent = (aabb_max - aabb_min) *
                     (0.5 * std::max(transform_field_config_.aabb_scale,
                                     1.0f));
  half_extent = torch::clamp_min(half_extent, 1e-3);
  aabb_min = (center - half_extent).detach();
  aabb_max = (center + half_extent).detach();

  {
    // Equal initial TF parameters across start modes; do not advance training RNG.
    ScopedTorchRNG cpu_rng(torch::Device(torch::kCPU), initialization_seed_);
    ScopedTorchRNG device_rng(torch::Device(device_type_), initialization_seed_);
    transform_field_ = TransformField(aabb_min, aabb_max, transform_field_config_);
  }
  std::cout << "[Continuous TF INIT] mapping_iteration=" << mapping_iter_
            << " start_mode=" << transform_field_config_.start_mode
            << " first_valid_maturity_cycle_iteration=" << gaussians_->protection_.first_valid_maturity_cycle_iteration
            << " initialization_seed=" << initialization_seed_ << '\n';
  transform_field_->to(torch::Device(device_type_));
  torch::optim::AdamOptions mlp_options(
      transform_field_config_.mlp_lr_init);
  mlp_options.eps() = 1e-15;
  transform_optimizer_ = std::make_shared<torch::optim::Adam>(
      transform_field_->mlpParameters(), mlp_options);
  transform_optimizer_->add_param_group(
      transform_field_->gridParameters());
  transform_optimizer_->param_groups()[1].options().set_lr(
      transform_field_config_.grid_lr_init);
  normal_update_clock_.tf_updates = 0;
  continuous_tf_statistics_.ready_count_when_initialized = ready_count >= 0 ? ready_count
      : gaussians_->protection_.indicesAbove(transform_field_config_.ready_maturity).numel();
  continuous_tf_statistics_.initial_aabb_population = mature_xyz.size(0);
  continuous_tf_statistics_.tf_initialization_iteration = mapping_iter_;
  std::cout << "[Elastic TF INIT] tf_initialization_iteration=" << mapping_iter_
            << " ready_count_when_initialized=" << continuous_tf_statistics_.ready_count_when_initialized
            << " initial_aabb_population=" << mature_xyz.size(0)
            << " initial_aabb_threshold=" << domain_threshold << '\n';
  if (ultrafast_rotation_.enabled) {
    ultrafast_tf_first_initialized_ = mapping_iter_;
    std::cout << "[UltrafastRotation TF] event=first_initialized mapping_iteration="
              << mapping_iter_ << '\n';
  }
  if (drift_stability_manager_ && drift_stability_config_.enabled)
    drift_stability_manager_->refreshReference(
        selector_network_, transform_field_, mapping_iter_, change_version_,
        0, false, true, "tf_initialized");
  std::cout << "[Gaussian Mapper] Transform Field initialized at mapping_iter "
            << mapping_iter_ << ", current-position AABB initialized"
            << std::endl;
}

void GaussianMapper::ensureSharedMappingOptimizer() {
  if (!gaussians_ || !gaussians_->optimizer_) return;

  auto& optimizer = *gaussians_->optimizer_;
  if (selector_network_ && !shared_selector_optimizer_configured_) {
    shared_selector_group_index_ = optimizer.param_groups().size();
    optimizer.add_param_group(selector_network_->parameters());
    optimizer.param_groups()[shared_selector_group_index_].options().set_lr(
        selector_learning_rate_);
    shared_selector_optimizer_configured_ = true;
    std::cout << "[Gaussian Mapper] Normal LOD Selector parameters attached to "
                 "the shared Gaussian optimizer (group "
              << shared_selector_group_index_ << ")" << std::endl;
  }

  if (transform_field_ && !shared_transform_optimizer_configured_) {
    shared_transform_mlp_group_index_ = optimizer.param_groups().size();
    optimizer.add_param_group(transform_field_->mlpParameters());
    optimizer.param_groups()[shared_transform_mlp_group_index_].options().set_lr(
        transform_field_config_.mlp_lr_init);
    shared_transform_grid_group_index_ = optimizer.param_groups().size();
    optimizer.add_param_group(transform_field_->gridParameters());
    optimizer.param_groups()[shared_transform_grid_group_index_].options().set_lr(
        transform_field_config_.grid_lr_init);
    shared_transform_optimizer_configured_ = true;
    std::cout << "[Gaussian Mapper] Normal LOD Transform Field parameters "
                 "attached to the shared Gaussian optimizer (groups "
              << shared_transform_mlp_group_index_ << ", "
              << shared_transform_grid_group_index_ << ")" << std::endl;
  }
}

bool GaussianMapper::stepWithHistoricalGuard(const std::shared_ptr<GaussianKeyframe>& current,
                                            bool count_rejection, bool* stage_committed) {
  if (stage_committed) *stage_committed = false;
  auto cuda_tf_groups = transform_field_ && transform_field_->config().backend == "cuda" &&
      shared_transform_optimizer_configured_
      ? std::vector<std::size_t>{shared_transform_mlp_group_index_, shared_transform_grid_group_index_}
      : std::vector<std::size_t>{};
  bool tf_had_gradient = false;
  if (shared_transform_optimizer_configured_) {
    for (auto index : {shared_transform_mlp_group_index_, shared_transform_grid_group_index_}) {
      const auto& group = gaussians_->optimizer_->param_groups()[index];
      if (static_cast<const torch::optim::AdamOptions&>(group.options()).lr() == 0.0) continue;
      for (const auto& p : group.params()) tf_had_gradient |= p.grad().defined();
    }
  }
  if (!tf_had_gradient) cuda_tf_groups.clear();
  if (selector_backend_ == "cuda" && shared_selector_optimizer_configured_)
    cuda_tf_groups.push_back(shared_selector_group_index_);
  if (!gradient_constraint_config_.enabled) {
    if (!fused_dense_adam_ || !mapping_cuda::tryFusedDenseAdam(*gaussians_->optimizer_))
      tf_cuda::tfCudaAdamMixedStep(*gaussians_->optimizer_, cuda_tf_groups, gaussians_->protection_.timing());
    if (stage_committed) *stage_committed = true;
    normal_update_clock_.record(true, tf_had_gradient, count_rejection); return true;
  }
  const auto start = std::chrono::steady_clock::now();
  ++gradient_guard_attempt_;
  GradientConstraintReport report;
  GuardEvaluation evaluation;
  auto log = [&] {
    if (report.status == "ACCEPT") ++gradient_accept_count_;
    else if (report.status == "SHRINK") ++gradient_shrink_count_;
    else if (report.status == "CANCEL") ++gradient_cancel_count_;
    else if (report.status == "NOOP") ++gradient_noop_count_;
    else if (report.status == "FALLBACK") ++gradient_fallback_count_;
    if (report.failure_kind == GuardFailureKind::NonFinite || report.failure_kind == GuardFailureKind::InvalidState)
      std::cerr << "[GradientGuard WARNING] iter=" << mapping_iter_ << " reason=" << report.reason
                << " committed=" << report.committed << '\n';
    const bool changed = gradient_guard_last_status_ != report.status + ":" + report.reason;
    gradient_guard_last_status_ = report.status + ":" + report.reason;
    if (!changed && gradient_guard_attempt_ % gradient_constraint_config_.log_interval != 0) return;
    std::ostringstream line;
    line << std::fixed << std::setprecision(4) << "[GradientGuard] iter=" << mapping_iter_
         << " source_kf=" << (current ? std::to_string(current->fid_) : "NA")
         << " attempt=" << gradient_guard_attempt_ << " status=" << report.status
         << " committed=" << report.committed
         << " tf_updates=" << normal_update_clock_.tf_updates
         << '/' << gradient_constraint_config_.tf_warmup_steps
         << " gaussian_lr_step=" << normal_update_clock_.gaussianLRStep(getIteration())
         << " alpha=" << report.alpha << " conflicts=" << report.conflicts
         << " candidates=" << report.candidates << " fallback_checks=" << report.fallback_checks
         << " would_cancel=" << report.would_cancel << " reason=" << report.reason;
    for (std::size_t j = 0; j < report.before.size(); ++j) {
      const auto& objective = evaluation.objectives[j];
      line << ' ' << objective.name << "(MSE_x1e6 before/raw/last_checked/committed)="
           << report.before[j] * 1e6;
      for (const auto* losses : {&report.raw_after, &report.last_checked_after, &report.committed_after}) {
        line << '/';
        if (j < losses->size() && std::isfinite((*losses)[j])) line << (*losses)[j] * 1e6;
        else line << "NA";
      }
    }
    line << " update_ratio_by_group=";
    for (std::size_t g = 0; g < report.update_ratio_by_group.size(); ++g)
      line << (g ? "," : "") << g << ':' << report.update_ratio_by_group[g];
    report.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    line << " ms=" << report.milliseconds << " stage_ms(preview/baseline/history/project/validate)="
         << report.preview_ms << '/' << report.baseline_ms << '/' << report.history_ms << '/'
         << report.projection_ms << '/' << report.validation_ms
         << " projection_converged=" << report.projection_converged
         << " projection_violation=" << report.projection_violation
         << " counts(accept/shrink/fallback/cancel/noop/would_cancel)=" << gradient_accept_count_
         << '/' << gradient_shrink_count_ << '/' << gradient_fallback_count_ << '/'
         << gradient_cancel_count_ << '/' << gradient_noop_count_ << '/' << gradient_would_cancel_count_;
    std::cout << line.str() << '\n';
  };
  auto unprotected = [&](const char* reason, const char* status = "UNPROTECTED") {
    tf_cuda::tfCudaAdamMixedStep(*gaussians_->optimizer_, cuda_tf_groups);
    if (stage_committed) *stage_committed = true;
    normal_update_clock_.record(true, tf_had_gradient, count_rejection);
    report.status = status; report.reason = reason; log();
  };
  if (!normal_update_clock_.guardReady(static_cast<bool>(transform_field_),
                                      gradient_constraint_config_.tf_warmup_steps)) {
    unprotected(transform_field_ ? "tf_warmup" : "tf_not_initialized", "WARMUP"); return true;
  }
  if ((gradient_guard_attempt_ - 1) % gradient_constraint_config_.check_interval != 0) {
    unprotected("check_interval"); return true;
  }
  if (!selector_network_ || selector_target_ratios_.empty() || gaussians_->xyz_.size(0) == 0) {
    unprotected("elastic_not_initialized"); return true;
  }
  std::vector<std::shared_ptr<GaussianKeyframe>> all_keyframes;
  for (const auto& entry : scene_->keyframes())
    if (entry.second && entry.second->original_image_.defined()) all_keyframes.push_back(entry.second);
  auto recent_ids = recent_window_importance_ ? recent_window_importance_->keyframeIds()
                                             : std::vector<std::size_t>{};
  if (current) recent_ids.push_back(current->fid_);
  const auto pool = historical_outlier_selector_->select(all_keyframes, recent_ids);
  if (pool.empty()) { unprotected("no_historical_view"); return true; }

  // No cross-iteration gradient/geometry cache. Preserve BOTH RNG streams, even
  // when lazily constructing the reusable candidate networks below.
  ScopedTorchRNG cpu_rng(torch::Device(torch::kCPU));
  std::unique_ptr<ScopedTorchRNG> device_rng;
  if (gaussians_->xyz_.device().is_cuda())
    device_rng = std::make_unique<ScopedTorchRNG>(gaussians_->xyz_.device());
  std::vector<GuardObjective> objectives;
  const int view_count = std::min<int>(gradient_constraint_config_.memory_views_per_check, pool.size());
  const int ratio_count = std::min<int>(gradient_constraint_config_.memory_ratios_per_check, selector_target_ratios_.size());
  for (int v = 0; v < view_count; ++v) {
    const auto camera = pool[(gradient_history_cursor_ + v) % pool.size()].keyframe;
    objectives.push_back({"full_KF" + std::to_string(camera->fid_), camera->fid_, true, 1.0f});
    for (int r = 0; r < ratio_count; ++r) {
      const float ratio = selector_target_ratios_[(gradient_ratio_cursor_ + r) % selector_target_ratios_.size()];
      std::ostringstream name;
      name << std::fixed << std::setprecision(4) << "lod_KF" << camera->fid_ << "_r" << ratio;
      objectives.push_back({name.str(), camera->fid_, false, ratio});
    }
  }
  gradient_history_cursor_ += view_count;
  gradient_ratio_cursor_ += ratio_count;
  evaluation = makeHistoricalEvaluation(objectives);
  last_guard_objectives_ = evaluation.objectives;
  report = GradientConstraintManager(gradient_constraint_config_).step(*gaussians_->optimizer_, evaluation, cuda_tf_groups);
  if (report.would_cancel) {
    ++gradient_would_cancel_count_;
    if (selector_replay_enabled_ && current &&
        conflict_replay_manager_->recordCancellation(current->fid_))
      enqueueConflictReplay(current);
  }
  const bool tf_updated = tf_had_gradient && report.committed &&
      (report.update_ratio_by_group[shared_transform_mlp_group_index_] > 0.0 ||
       report.update_ratio_by_group[shared_transform_grid_group_index_] > 0.0);
  normal_update_clock_.record(report.committed, tf_updated, count_rejection);
  if (stage_committed) *stage_committed = report.committed;
  log();
  if (!report.committed) return false;
  for (size_t group = 0; group < std::min<size_t>(6, report.update_ratio_by_group.size()); ++group)
    if (report.update_ratio_by_group[group] > 0.0) return true;
  return false;
}

GuardEvaluation GaussianMapper::makeHistoricalEvaluation(
    const std::vector<GuardObjective>& objectives) {
  struct ViewLoss { std::shared_ptr<GaussianKeyframe> camera; float ratio; bool full; torch::Tensor gt; };
  struct State {
    GuardGaussianParameters raw;
    torch::Tensor maturity, selection_maturity, uniform;
    std::vector<ViewLoss> views;
    GaussianRenderInput candidate_full;
    std::map<float, GaussianRenderInput> candidate_lod;
    std::shared_ptr<improvements::selector::GumbelNetworkImpl> selector;
    std::shared_ptr<TransformFieldImpl> field;
  };
  auto state = std::make_shared<State>();
  state->raw = {gaussians_->xyz_, gaussians_->features_dc_, gaussians_->features_rest_,
      gaussians_->opacity_, gaussians_->scaling_, gaussians_->rotation_,
      gaussians_->active_sh_degree_, gaussians_->max_sh_degree_};
  state->selector = selector_network_.ptr();
  if (transform_field_) state->field = transform_field_.ptr();
  state->maturity = gaussians_->protection_.maturity.detach();
  state->selection_maturity = gaussians_->protection_.selectionMaturity(selector_protection_enabled_);
  state->uniform = gaussians_->protection_.cached_uniform;
  const float scale = pSLAM_ && !pSLAM_->isShutDown() ? selector_lod_render_scale_ : 1.0f;
  GuardEvaluation result;
  for (const auto& objective : objectives) {
    const auto camera = scene_->getKeyframe(objective.keyframe_id);
    if (!camera || !camera->original_image_.defined() || camera->original_image_.numel() == 0 ||
        !camera->getPosef().matrix().allFinite()) continue;
    const int h = std::max(1, static_cast<int>(camera->image_height_ * scale));
    const int w = std::max(1, static_cast<int>(camera->image_width_ * scale));
    const auto gt = camera->resized_image_cache_.get(camera->original_image_.to(gaussians_->xyz_.device()), h, w);
    state->views.push_back({camera, objective.ratio, objective.full, gt});
    result.objectives.push_back(objective);
  }
  result.prepare_candidate = [this, state](const AdamStepPreview& preview, const std::vector<torch::Tensor>& candidate) {
    gradient_candidate_modules_.prepare(preview, candidate, state->selector, state->field);
    state->candidate_full = state->raw.input(preview, &candidate);
    state->candidate_lod.clear();
  };
  result.evaluate = [this, state](std::size_t j, const AdamStepPreview& preview,
                                const std::vector<torch::Tensor>* candidate) {
    // Fresh live graphs for every historical gradient; only detached candidate
    // activations may be reused within the same parameter revision.
    const auto full = candidate ? state->candidate_full : state->raw.input(preview);
    const auto selector = candidate ? gradient_candidate_modules_.selector() : state->selector;
    const auto field = candidate ? gradient_candidate_modules_.field() : state->field;
    const auto& view = state->views[j];
    auto input = full;
    if (!view.full) {
      if (candidate && state->candidate_lod.count(view.ratio)) input = state->candidate_lod.at(view.ratio);
      else {
        input = historicalLoDInput(full, state->maturity,
            state->uniform, selector, field, view.ratio, selector_temperature_, state->selection_maturity,
            transform_field_config_.elastic_entry_maturity);
        if (candidate) state->candidate_lod.emplace(view.ratio, input);
      }
    }
    const auto finite = torch::stack({torch::isfinite(input.xyz).all(), torch::isfinite(input.opacity).all(),
        torch::isfinite(input.scaling).all(), torch::isfinite(input.rotation).all(),
        torch::isfinite(input.features_dc).all(), torch::isfinite(input.features_rest).all()}).all();
    if (!finite.item<bool>()) return torch::full({}, std::numeric_limits<float>::quiet_NaN(),
                                                full.xyz.options().requires_grad(false));
    const auto rendered = GaussianRenderer::render(view.camera, view.gt.size(1), view.gt.size(2), input,
        pipe_params_, background_, override_color_, 1.0f, false, false,
        torch::GradMode::is_enabled() ? RasterizeMode::Training : RasterizeMode::Inference);
    return (rendered.image - view.gt).square().mean();
  };
  return result;
}

void GaussianMapper::updateTransformLearningRate() {
  if (!transform_optimizer_ || !gaussians_ || !gaussians_->optimizer_) return;
  const auto max_steps = std::max(transform_field_config_.lr_max_steps, 1);
  const auto t = std::clamp(
      static_cast<float>(normal_update_clock_.tf_updates) / static_cast<float>(max_steps),
      0.0f, 1.0f);
  const auto delay_mult = std::clamp(transform_field_config_.lr_delay_mult,
                                     0.0f, 1.0f);
  const auto delay_rate =
      delay_mult + (1.0f - delay_mult) * std::sin(1.5707963267948966f * t);
  const auto interpolate = [t](float initial, float final) {
    if (initial <= 0.0f || final <= 0.0f) return 0.0f;
    return std::exp(std::log(initial) * (1.0f - t) + std::log(final) * t);
  };
  const auto mlp_lr =
      delay_rate * interpolate(transform_field_config_.mlp_lr_init,
                               transform_field_config_.mlp_lr_final);
  const auto grid_lr =
      delay_rate * interpolate(transform_field_config_.grid_lr_init,
                               transform_field_config_.grid_lr_final);
  transform_optimizer_->param_groups()[0].options().set_lr(mlp_lr);
  transform_optimizer_->param_groups()[1].options().set_lr(grid_lr);
  if (shared_transform_optimizer_configured_) {
    gaussians_->optimizer_->param_groups()[shared_transform_mlp_group_index_]
        .options()
        .set_lr(mlp_lr);
    gaussians_->optimizer_->param_groups()[shared_transform_grid_group_index_]
        .options()
        .set_lr(grid_lr);
  }
}

ChangeCapture GaussianMapper::captureGeometryChange(ChangeFlag flag) {
  if(selector_explicit_anchor_blocks_ && flag!=ChangeFlag::FIELD_DOMAIN) {
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    selector_network_->anchors()->beginBlockGeometry();
  }
  ChangeCapture capture;
  capture.flags = toBits(flag);
  const bool field_only = flag == ChangeFlag::FIELD_DOMAIN;
  const bool needs_cd = change_distill_config_.enabled &&
                        (selector_network_ || transform_field_);
  const bool needs_aabb = transform_field_ && dynamic_aabb_config_.enabled;
  // Preserve the operation flag for pose-cache invalidation without
  // cloning Gaussian state when neither Change Distill nor AABB needs it.
  if (!change_coordinator_ || (!needs_cd && !needs_aabb) ||
      (field_only && !transform_field_))
    return capture;
  std::unique_lock<std::mutex> lock_render(mutex_render_);
  const auto xyz = gaussians_->getXYZ().detach();
  if (!xyz.defined() || xyz.numel() == 0) return capture;
  const auto mature_mask = selector_protection_enabled_
                               ? gaussians_->getSelectorMatureMask()
                               : torch::ones(
                                     {xyz.size(0)},
                                     torch::TensorOptions()
                                         .dtype(torch::kBool)
                                          .device(xyz.device()));
  AABB3D old_aabb;
  if (transform_field_) {
    old_aabb.min_xyz = transform_field_->aabbMin().detach().clone();
    old_aabb.max_xyz = transform_field_->aabbMax().detach().clone();
  } else if(selector_compensation_) {
    old_aabb.min_xyz=std::get<0>(xyz.min(0)).clone();
    old_aabb.max_xyz=std::get<0>(xyz.max(0)).clone();
  }
  // CD keeps its compatibility population; an empty CD population must not
  // prevent the independent TF domain from following valid geometry.
  if (!needs_aabb && !mature_mask.any().item<bool>()) return capture;
  capture = change_coordinator_->captureBeforeGeometryChange(
      flag, mature_mask, {xyz, gaussians_->scaling_.detach(),
                         gaussians_->getRotationActivation().detach()},
      old_aabb, topology_version_, correspondence_version_, mapping_iter_);
  if (change_distill_config_.enabled && !field_only && selector_network_ &&
      (change_distill_config_.distill_selector_on_geometry_change ||
       (selector_compensation_ && change_distill_config_.distill_tf_on_geometry_change))) {
    capture.teacher_selector = improvements::selector::GumbelNetwork(
        selector_network_->cloneFrozen());
    if (selector_network_->anchorGaussianAttributes()) {
      capture.old_selector_sh_dc = gaussians_->features_dc_.detach().clone();
      capture.old_selector_opacity = gaussians_->getOpacityActivation().detach().clone();
      if(selector_network_->compensationOutputWidth()==14)
        capture.old_selector_opacity_logits=gaussians_->opacity_.detach().clone();
    }
  }
  if (change_distill_config_.enabled && transform_field_ &&
      ((!field_only && change_distill_config_.distill_tf_on_geometry_change) ||
       change_distill_config_.distill_tf_on_aabb_change))
    capture.teacher_tf = TransformField(transform_field_->cloneFrozen());
  return capture;
}

void GaussianMapper::finishGeometryChange(ChangeCapture&& capture,
                                           const AABBUpdatePlan* prepared_plan) {
  if (!change_coordinator_) return;
  std::unique_lock<std::mutex> lock_render(mutex_render_);
  const bool geometry_operation =
      (capture.flags & (toBits(ChangeFlag::LOCAL_GEOMETRY) |
                        toBits(ChangeFlag::GLOBAL_GEOMETRY))) != 0;
  const bool geometry_changed = capture.has_geometry_mutation ||
                                !capture.corrections.empty();
  const bool local_geometry = (capture.flags & toBits(ChangeFlag::LOCAL_GEOMETRY)) != 0;
  if(selector_explicit_anchor_blocks_ && geometry_operation) gaussians_->finishAnchorGeometry();
  if (geometry_changed && selector_network_ && selector_network_->anchorConditioned() && !selector_explicit_anchor_blocks_)
    selector_network_->anchors()->reassociate(gaussians_->getXYZ(),
        {gaussians_->optimizer_.get(), selector_optimizer_.get()});
  if (geometry_changed && selector_anchor_adaptive_ && selector_network_)
    selector_network_->anchors()->refine(gaussians_->getXYZ(),
        {gaussians_->optimizer_.get(), selector_optimizer_.get()});
  // KF pose changes also invalidate GI, even in an empty CD event.
  if (geometry_operation) {
    if(anchor_si_config_.enabled)anchor_si_.invalidate();
    if (recent_window_importance_)
      recent_window_importance_->invalidateContributions();
  }
  if (capture.snapshot_count == 0) {
    if(local_geometry || (geometry_operation && gaussians_->fixedTenAnchors())) gaussians_->migrateLocalAnchors();
    if (capture.flags != toBits(ChangeFlag::NONE)) {
      std::cout << "[ChangeCoordinator] skip type="
                << changeEventTypeName(capture.flags)
                << " reason=no_snapshot" << std::endl;
      if (geometry_changed) {
        ++change_version_;
        if (drift_stability_manager_ && drift_stability_config_.enabled)
          drift_stability_manager_->refreshReference(
              selector_network_, transform_field_, mapping_iter_, change_version_,
              mapping_iter_ + 1, true, true, "geometry_change");
      }
    }
    return;
  }
  ChangeTrainingContext context;
  context.current = {gaussians_->getXYZ().detach(), gaussians_->scaling_.detach(),
                     gaussians_->getRotationActivation().detach()};
  if (selector_network_ && selector_network_->anchorGaussianAttributes()) {
    context.selector_sh_dc = gaussians_->features_dc_.detach();
    context.selector_opacity = gaussians_->getOpacityActivation().detach();
    if(selector_network_->compensationOutputWidth()==14)
      context.selector_opacity_logits=gaussians_->opacity_.detach();
  }
  context.selector = selector_network_;
  context.selector_compensation_active=selector_compensation_ && getIteration()>=selector_compensation_start_iter_;
  context.tf = transform_field_;
  context.normal_selector_optimizer = selector_optimizer_.get();
  context.normal_tf_optimizer = transform_optimizer_.get();
  context.shared_mapping_optimizer = gaussians_->optimizer_.get();
  context.ratios = selector_target_ratios_;
  context.temperature = selector_temperature_;
  AABBUpdatePlan domain_plan;
  if (!prepared_plan && transform_field_ && dynamic_aabb_manager_ && dynamic_aabb_config_.enabled) {
    const auto xyz = context.current.xyz.index({elasticDomainMask(context.current.xyz,
        gaussians_->protection_.maturity, transform_field_config_.elastic_entry_maturity)});
    AABB3D domain{transform_field_->aabbMin(), transform_field_->aabbMax()};
    if (transform_field_config_.elastic_entry_maturity == 0 || xyz.size(0) >= 2)
      domain_plan = dynamic_aabb_manager_->makeExpansionPlan(xyz, domain,
          dynamic_aabb_manager_->evaluate(xyz, domain));
    prepared_plan = &domain_plan;
  }
  auto report = change_coordinator_->finalizeGeometryChange(
      std::move(capture), context, correspondence_version_, prepared_plan);
  // Complete any explicitly enabled CD before deleting rows; its teacher keeps
  // the pre-event Gaussian correspondence until execute() has returned.
  if(local_geometry || (geometry_operation && gaussians_->fixedTenAnchors())) gaussians_->migrateLocalAnchors();
  cd_selector_steps_ += report.selector_steps;
  cd_tf_steps_ += report.tf_steps;
  cd_compensation_steps_ += report.compensation_steps;
  if (geometry_changed || report.aabb_changed) {
    ++change_version_;
    if (drift_stability_manager_ && drift_stability_config_.enabled)
      drift_stability_manager_->refreshReference(
          selector_network_, transform_field_, mapping_iter_, change_version_,
          geometry_operation ? mapping_iter_ + 1 : mapping_iter_,
          geometry_changed, true,
          geometry_changed ? "geometry_change" : "aabb_change");
  }
}

void GaussianMapper::evaluateDynamicAABB(bool force_check) {
  if (!dynamic_aabb_manager_ || !transform_field_ ||
      (!force_check &&
       !dynamic_aabb_manager_->shouldPeriodicCheck(mapping_iter_)))
    return;
  torch::Tensor mature_xyz;
  AABB3D current_aabb;
  AABBPressure pressure;
  {
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    const auto xyz = gaussians_->getXYZ().detach();
    mature_xyz = xyz.index({elasticDomainMask(xyz, gaussians_->protection_.maturity,
        transform_field_config_.elastic_entry_maturity)});
    if (transform_field_config_.elastic_entry_maturity > 0 && mature_xyz.size(0) < 2) return;
    current_aabb.min_xyz = transform_field_->aabbMin().detach();
    current_aabb.max_xyz = transform_field_->aabbMax().detach();
    pressure = dynamic_aabb_manager_->evaluate(mature_xyz, current_aabb);
  }
  std::ostringstream check_log;
  check_log << std::fixed << std::setprecision(4)
            << "[Gaussian Mapper]Dynamic AABB check: mapping_iter="
            << mapping_iter_ << ", population=" << pressure.mature_count
            << ", outside=" << pressure.outside_total
            << ", outside_ratio=" << pressure.outside_ratio
            << ", x_low=" << pressure.x_low_count
            << ", x_high=" << pressure.x_high_count
            << ", y_low=" << pressure.y_low_count
            << ", y_high=" << pressure.y_high_count
            << ", z_low=" << pressure.z_low_count
            << ", z_high=" << pressure.z_high_count;
  std::cout << check_log.str() << std::endl;
  const auto plan = dynamic_aabb_manager_->makeExpansionPlan(
      mature_xyz, current_aabb, pressure);
  if (plan.should_expand) {
    std::ostringstream expansion_log;
    expansion_log << std::fixed << std::setprecision(4)
              << "[Gaussian Mapper]Dynamic AABB expansion plan: old=["
              << plan.old_aabb.min_xyz[0].item<float>() << ", "
              << plan.old_aabb.max_xyz[0].item<float>() << "; "
              << plan.old_aabb.min_xyz[1].item<float>() << ", "
              << plan.old_aabb.max_xyz[1].item<float>() << "; "
              << plan.old_aabb.min_xyz[2].item<float>() << ", "
              << plan.old_aabb.max_xyz[2].item<float>() << "] new=["
              << plan.new_aabb.min_xyz[0].item<float>() << ", "
              << plan.new_aabb.max_xyz[0].item<float>() << "; "
              << plan.new_aabb.min_xyz[1].item<float>() << ", "
              << plan.new_aabb.max_xyz[1].item<float>() << "; "
              << plan.new_aabb.min_xyz[2].item<float>() << ", "
              << plan.new_aabb.max_xyz[2].item<float>() << "]"
              << ", affected_planes={xy:" << plan.affected_planes.xy
              << ", xz:" << plan.affected_planes.xz
              << ", yz:" << plan.affected_planes.yz
              << ", xe:" << plan.affected_planes.xe
              << ", ye:" << plan.affected_planes.ye
              << ", ze:" << plan.affected_planes.ze << "}";
    std::cout << expansion_log.str() << std::endl;
    if (change_coordinator_) {
      auto capture = captureGeometryChange(ChangeFlag::FIELD_DOMAIN);
      finishGeometryChange(std::move(capture), &plan);
    }
  }
}

void GaussianMapper::evaluateDriftStability(bool deferred) {
  if (!drift_stability_manager_ ||
       (!deferred && !drift_stability_manager_->shouldProbe(mapping_iter_)))
    return;

  std::unique_lock<std::mutex> lock_render(mutex_render_);
  // Establish/validate references BEFORE constructing teacher render evidence.
  if (!drift_stability_manager_->prepareProbe(
          selector_network_, transform_field_, mapping_iter_, change_version_, deferred))
    return;
  // Freeze geometric inputs first. SH/opacity are copied only if QUALITY is
  // actually needed, under this same mutex (DSD never changes master Gaussians).
  const auto xyz = gaussians_->getXYZ().detach().clone();
  if (!xyz.defined() || xyz.numel() == 0) return;
  const auto mature_mask = (selector_protection_enabled_
                               ? gaussians_->getSelectorMatureMask()
                               : torch::ones(
                                     {xyz.size(0)},
                                     torch::TensorOptions()
                                          .dtype(torch::kBool)
                                          .device(xyz.device())))
                               .detach()
                               .clone();
  const auto mature_indices = torch::nonzero(mature_mask).reshape({-1});
  const auto probe_maturity = gaussians_->protection_.selectionMaturity(selector_protection_enabled_).clone();
  const auto probe_tf_maturity = gaussians_->protection_.maturity.detach().clone();
  if (mature_indices.numel() == 0) {
    std::cout << "[HistoricalProbe SKIP] iter=" << mapping_iter_
              << " reason=no_mature_gaussians" << std::endl;
    drift_stability_manager_->maybeProbeAndDistill(
        {}, selector_network_, transform_field_, selector_target_ratios_,
        selector_temperature_, {}, {}, mapping_iter_, change_version_, {}, nullptr, deferred);
    return;
  }
  AABB3D aabb;
  if (transform_field_) {
    aabb.min_xyz = transform_field_->aabbMin().detach();
    aabb.max_xyz = transform_field_->aabbMax().detach();
  } else {
    aabb.min_xyz = std::get<0>(xyz.min(/*dim=*/0));
    aabb.max_xyz = std::get<0>(xyz.max(/*dim=*/0));
  }
  const auto probe_log_scaling = gaussians_->scaling_.detach().clone();
  const auto probe_rotation = gaussians_->getRotationActivation().detach().clone();
  const auto probe_scaling = probe_log_scaling.exp();
  torch::Tensor probe_opacity, probe_features_dc, probe_features_rest;
  const GaussianAttrState current{xyz, probe_log_scaling, probe_rotation};
  HistoricalProbeCache probe_cache(current, selector_temperature_);
  torch::Tensor quality_uniform;
  quality_uniform = gaussians_->protection_.cached_uniform.clone();
  // One probe uses one resolution, including recovery and EMA candidate checks.
  const float quality_scale = pSLAM_ && !pSLAM_->isShutDown() ? 0.5f : 1.0f;
  if (!gaussians_->last_elastic_supervised_iter_.defined() ||
      gaussians_->last_elastic_supervised_iter_.size(0) != xyz.size(0)) {
    gaussians_->last_elastic_supervised_iter_ =
        torch::full({xyz.size(0)}, static_cast<int64_t>(mapping_iter_),
                    xyz.options().dtype(torch::kInt64));
  }
  std::vector<std::shared_ptr<GaussianKeyframe>> all_keyframes;
  all_keyframes.reserve(scene_->keyframes().size());
  for (const auto& [id, keyframe] : scene_->keyframes())
    if (keyframe) all_keyframes.push_back(keyframe);
  const auto recent_ids = recent_window_importance_
                              ? recent_window_importance_->keyframeIds()
                              : std::vector<std::size_t>{};
  const auto outliers = historical_outlier_selector_
                            ? historical_outlier_selector_->select(
                                  all_keyframes, recent_ids)
                            : std::vector<HistoricalOutlierView>{};
  std::vector<HistoricalViewObservation> views;
  views.reserve(outliers.size() + 1);
  const auto protected_count = xyz.size(0) - mature_indices.numel();
  using RenderKey = std::tuple<uintptr_t, uintptr_t, float>;
  std::map<RenderKey, GaussianRenderInput> render_inputs;
  using ModuleKey = std::pair<uintptr_t, float>;
  std::map<ModuleKey, std::pair<torch::Tensor, torch::Tensor>> selector_inputs;
  std::map<ModuleKey, ElasticGaussianAttributes> field_inputs;
  uint64_t selector_revision = 0, field_revision = 0;
  auto render_probe_quality = [&](const std::shared_ptr<GaussianKeyframe>& keyframe,
                                  const improvements::selector::GumbelNetwork& selector,
                                  const TransformField& field,
                                  float ratio) {
    torch::NoGradGuard no_grad;
    return probe_cache.quality(keyframe->fid_, ratio, selector, field, [&] {
      if (!probe_opacity.defined()) {
        probe_opacity = gaussians_->getOpacityActivation().detach().clone();
        probe_features_dc = gaussians_->features_dc_.detach().clone();
        probe_features_rest = gaussians_->features_rest_.detach().clone();
      }
      const bool selector_changed = selector_revision != probe_cache.selectorRevision();
      const bool field_changed = field_revision != probe_cache.tfRevision();
      if (selector_changed || field_changed) {
        const auto student_selector_id = selector_network_
            ? reinterpret_cast<uintptr_t>(selector_network_.ptr().get()) : 0;
        const auto student_field_id = transform_field_
            ? reinterpret_cast<uintptr_t>(transform_field_.ptr().get()) : 0;
        for (auto it = render_inputs.begin(); it != render_inputs.end();)
          if ((selector_changed && std::get<0>(it->first) == student_selector_id) ||
              (field_changed && std::get<1>(it->first) == student_field_id))
            it = render_inputs.erase(it);
          else ++it;
        for (auto it = selector_inputs.begin(); it != selector_inputs.end();)
          if (selector_changed && it->first.first == student_selector_id)
            it = selector_inputs.erase(it);
          else ++it;
        for (auto it = field_inputs.begin(); it != field_inputs.end();)
          if (field_changed && it->first.first == student_field_id)
            it = field_inputs.erase(it);
          else ++it;
        selector_revision = probe_cache.selectorRevision();
        field_revision = probe_cache.tfRevision();
      }
      const auto selector_id = selector ? reinterpret_cast<uintptr_t>(selector.ptr().get()) : 0;
      const auto field_id = field ? reinterpret_cast<uintptr_t>(field.ptr().get()) : 0;
      const RenderKey key{selector_id, field_id, ratio};
      auto input = render_inputs.find(key);
      if (input == render_inputs.end()) {
        auto selected = selector_inputs.find({selector_id, ratio});
        if (selected == selector_inputs.end()) {
          const float entry = transform_field_config_.elastic_entry_maturity;
          const auto entry_indices = entry > 0 ? torch::nonzero(probe_tf_maturity >= entry).reshape({-1}) : torch::Tensor{};
          auto probability = selector ? elasticSelectorProbability(probe_tf_maturity, probe_maturity,
              entry, ratio, entry_indices, [&](const torch::Tensor& indices) {
                return selector.ptr()->forwardProbability(indices.defined() ? xyz.index_select(0, indices) : xyz,
                    indices.defined() ? probe_rotation.index_select(0, indices) : probe_rotation,
                    indices.defined() ? probe_scaling.index_select(0, indices) : probe_scaling, ratio, selector_temperature_);
              }) : mixSelectorProbability(torch::zeros({xyz.size(0)}, xyz.options()), probe_maturity, ratio);
          if (selector) {
            probability = selector->maskInactiveSlots(probability,probe_opacity);
            ++probe_cache.selector_forwards;
            probe_cache.selector_rows += xyz.size(0);
          }
          const auto keep = quality_uniform < probability;
          const auto mask = keep.to(torch::kFloat32);
          const auto active = keep.sum();
          selected = selector_inputs.emplace(ModuleKey{selector_id, ratio},
              std::make_pair(mask.unsqueeze(1) * probe_opacity,
                  torch::stack({active.to(torch::kDouble), probability.sum().to(torch::kDouble)}))).first;
        }
        auto elastic = field_inputs.find({field_id, ratio});
        if (elastic == field_inputs.end()) {
          ElasticGaussianAttributes attrs{xyz, probe_scaling, probe_rotation};
          if (field) {
            auto transform_mask = field.ptr()->candidateMask(xyz, probe_tf_maturity);
            const auto indices = torch::nonzero(transform_mask).reshape({-1});
            if (indices.numel() > 0) {
              ++probe_cache.tf_forwards;
              probe_cache.tf_rows += indices.numel();
              const auto state = field.ptr()->applyResidual(xyz.index_select(0, indices),
                  probe_scaling.index_select(0, indices), probe_rotation.index_select(0, indices),
                  ratio, elasticMaturity(probe_tf_maturity.index_select(0, indices), field.ptr()->config().elastic_entry_maturity));
              attrs.xyz = xyz.index_copy(0, indices, state.xyz);
              attrs.scaling = probe_scaling.index_copy(0, indices, state.scaling);
              attrs.rotation = probe_rotation.index_copy(0, indices, state.rotation);
            }
          }
          elastic = field_inputs.emplace(ModuleKey{field_id, ratio}, attrs).first;
        }
        input = render_inputs.emplace(key, GaussianRenderInput{
            elastic->second.xyz, selected->second.first,
            elastic->second.scaling, elastic->second.rotation, probe_features_dc,
            probe_features_rest, gaussians_->active_sh_degree_, gaussians_->max_sh_degree_}).first;
      }
      const int height = std::max(1, static_cast<int>(keyframe->image_height_ * quality_scale));
      const int width = std::max(1, static_cast<int>(keyframe->image_width_ * quality_scale));
      auto gt = keyframe->resized_image_cache_.get(
          keyframe->original_image_.to(xyz.device()), height, width);
      auto rendered = GaussianRenderer::render(keyframe, height, width, input->second,
          pipe_params_, background_, override_color_, 1.0f, false, false,
          RasterizeMode::Inference);
      // One device-to-host transfer for PSNR and the exact active row count.
      const auto metrics = torch::stack({loss_utils::psnr(rendered.image, gt).to(torch::kDouble),
          selector_inputs.at({selector_id, ratio}).second[0],
          selector_inputs.at({selector_id, ratio}).second[1]}).to(torch::kCPU);
      return HistoricalRenderQuality{
          metrics[0].item<double>(), 0,
          static_cast<int64_t>(metrics[1].item<double>()),
          double(ratio) * xyz.size(0),
          metrics[2].item<double>(),
          double(ratio) * xyz.size(0), metrics[2].item<double>()};
    });
  };
  for (const auto& view : outliers) {
    HistoricalViewObservation observation;
    observation.keyframe_id = view.keyframe_id;
    observation.outlier_score = view.outlier_score;
    torch::NoGradGuard no_grad;
    const auto visible = GaussianRenderer::render(
        view.keyframe, view.keyframe->image_height_, view.keyframe->image_width_,
        gaussians_, pipe_params_, background_, override_color_, 1.0f, false,
        torch::Tensor(), true, false, RasterizeMode::Visibility);
    const auto& visible_indices = visible.visibility_indices;
    if (visible_indices.numel() > 0) {
      observation.visible_mature_indices = visible_indices.index(
          {mature_mask.index({visible_indices})});
    }
    views.push_back(std::move(observation));
  }
  for (auto id = recent_ids.rbegin(); id != recent_ids.rend(); ++id) {
    if (std::none_of(all_keyframes.begin(), all_keyframes.end(),
                    [&](const auto& keyframe) { return keyframe->fid_ == *id; }))
      continue;  // A culled KF may still be retained by the recent window.
    HistoricalViewObservation guard;
    guard.keyframe_id = *id;
    guard.outlier_score = 1.0;
    guard.guard_only = true;
    views.push_back(std::move(guard));
    break;
  }
  drift_stability_manager_->maybeProbeAndDistill(
      current, selector_network_, transform_field_, selector_target_ratios_,
      selector_temperature_, aabb, views, mapping_iter_, change_version_,
      [&](std::size_t keyframe_id, float ratio,
          const improvements::selector::GumbelNetwork& selector,
          const TransformField& field) {
        const auto view = std::find_if(all_keyframes.begin(), all_keyframes.end(),
            [keyframe_id](const auto& value) { return value->fid_ == keyframe_id; });
        TORCH_CHECK(view != all_keyframes.end(), "Missing historical/recent validation view");
        return render_probe_quality(*view, selector, field, ratio);
      }, &probe_cache, deferred);
  std::ostringstream cost;
  cost << std::fixed << std::setprecision(4)
       << "[HistoricalProbe COST] iter=" << mapping_iter_
       << " quality_scale=" << quality_scale << " immature=" << protected_count
       << " renders=" << probe_cache.quality_renders
       << " render_cache_hits=" << probe_cache.quality_hits
       << " forwards(S/TF)=" << probe_cache.selector_forwards << "/" << probe_cache.tf_forwards
       << " rows(S/TF)=" << probe_cache.selector_rows << "/" << probe_cache.tf_rows;
  std::cout << cost.str() << std::endl;
}

void GaussianMapper::registerRecentKeyframe(
    const std::shared_ptr<GaussianKeyframe>& keyframe) {
  if (!keyframe) return;
  const bool new_keyframe = historical_keyframes_.find(keyframe->fid_) == historical_keyframes_.end();
  historical_keyframes_[keyframe->fid_] = keyframe;
  if (new_keyframe && selector_enabled_) {
    torch::NoGradGuard guard;
    gaussians_->protection_.cached_uniform = torch::rand_like(gaussians_->protection_.cached_uniform);
  }
  if (recent_window_importance_ && gaussians_ && gaussians_->getXYZ().defined())
    recent_window_importance_->pushKeyframe(
        keyframe, gaussians_->getXYZ().size(0));
}

void GaussianMapper::refreshRecentWindowContribution(
    const std::shared_ptr<GaussianKeyframe>& keyframe,
    const torch::Tensor& raw_importance,
    const torch::Tensor& contribution_count) {
  if (!keyframe || !recent_window_importance_ ||
      !recent_window_importance_->enabled()) return;
  recent_window_importance_->updateContribution(
      keyframe->fid_, raw_importance, contribution_count);
}

torch::Tensor GaussianMapper::renderGlobalGIContribution(uint64_t keyframe_id) {
  const auto found = historical_keyframes_.find(keyframe_id);
  TORCH_CHECK(found != historical_keyframes_.end(), "SGI bank contains an unavailable keyframe");
  const auto& camera = found->second;
  torch::NoGradGuard no_grad;
  auto rendered = GaussianRenderer::render(camera, camera->image_height_, camera->image_width_,
      gaussians_, pipe_params_, background_, override_color_, 1.0f, false,
      torch::Tensor(), true, true, RasterizeMode::Inference);
  // Preserve the repository kernel's raw sum(conic_opacity.w*T); no window normalization.
  return rendered.frame_importance;
}

void GaussianMapper::updateGlobalGIBank() {
  const auto snapshot_start = std::chrono::steady_clock::now();
  GlobalGIBatch batch;
  for (;;) {
    const auto lock_start = std::chrono::steady_clock::now();
    std::optional<ORB_SLAM3::GIAssociationState::Read> gi_read(std::in_place);
    const auto lock_acquired = std::chrono::steady_clock::now();
    auto map = pSLAM_->getAtlas()->GetCurrentMapForGI();
    if (!map) {
      gi_read.reset();
      if (isStopped() || pSLAM_->isShutDown()) return;
      // Release the read transaction before retrying a map switch/reset.
      std::this_thread::yield();
      continue;
    }
    std::vector<GlobalGISnapshotCache::Metadata> metadata;
    std::unordered_map<uint64_t, ORB_SLAM3::KeyFrame*> frames;
    for (auto keyframe : map->GetAllKeyFrames()) {
      if (!keyframe || keyframe->GIBad()) continue;
      const auto found = historical_keyframes_.find(keyframe->mnId);
      if (found == historical_keyframes_.end()) continue;
      const auto& camera = found->second;
      if (camera->image_width_ <= 0 || camera->image_height_ <= 0 ||
          !camera->full_proj_transform_.defined()) continue;
      metadata.emplace_back(keyframe->mnId, keyframe->GIAssociationRevision());
      frames.emplace(keyframe->mnId, keyframe);
    }
    batch = global_gi_snapshot_cache_.update(map->GIGeneration(),
        ORB_SLAM3::GIAssociationState::badRevision(), metadata,
        [&frames](uint64_t id) { return frames.at(id)->GIPointIds(); });
    static const bool validate_snapshot = std::getenv("CARTGS_GI_VALIDATE_SNAPSHOT") != nullptr;
    if (validate_snapshot) {
      std::map<uint64_t, std::vector<uint64_t>> authoritative;
      for (const auto& frame : frames) {
        auto& ids = authoritative[frame.first];
        for (auto point : frame.second->GetMapPoints())
          if (point && !point->isBad()) ids.push_back(point->mnId);
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      }
      const auto cached = global_gi_snapshot_cache_.snapshot();
      if (cached.size() != authoritative.size())
        throw std::logic_error("GI snapshot cache candidate count mismatch");
      for (const auto& frame : cached)
        if (authoritative.at(frame.id) != frame.mappoints)
          throw std::logic_error("GI snapshot cache association mismatch");
      global_gi_->recordValidatedSnapshot();
    }
    const auto lock_end = std::chrono::steady_clock::now();
    global_gi_->recordAssociationLocks(
        std::chrono::duration<double, std::milli>(lock_acquired - lock_start).count(),
        std::chrono::duration<double, std::milli>(lock_end - lock_acquired).count(),
        global_gi_snapshot_cache_.rebuiltRows(),
        ORB_SLAM3::GIAssociationState::writerWaitNs() / 1e6,
        ORB_SLAM3::GIAssociationState::trackingWaitNs() / 1e6);
    global_gi_->recordMapLocks(0, 0,
        map->tracking_map_wait_ns.load(std::memory_order_relaxed) / 1e6,
        map->tracking_map_lock_count.load(std::memory_order_relaxed));
    break;
  }
  // Stable IDs cross the transaction boundary; all ORB pointers stay inside it.
  global_gi_->updateBankBatch(std::move(batch), [this](uint64_t id) {
    return renderGlobalGIContribution(id);
  }, std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - snapshot_start).count());
  global_gi_->assertAligned(gaussians_->getXYZ());
}

bool GaussianMapper::trainJointReplay(
    const ConflictReplayTask& task, std::shared_ptr<GaussianKeyframe> camera) {
  ++replay_attempt_count_;
  const auto start = std::chrono::steady_clock::now();
  auto& optimizer = *gaussians_->optimizer_;
  std::unique_lock<std::mutex> lock_render(mutex_render_);
  optimizer.zero_grad(true);
  struct ClearGradients {
    torch::optim::Adam& optimizer;
    ~ClearGradients() { optimizer.zero_grad(true); }
  } clear{optimizer};
  auto skip = [&](const char* reason) {
    ++replay_skip_count_;
    std::ostringstream line;
    line << std::fixed << std::setprecision(4) << "[ConflictReplay SKIP] iter=" << mapping_iter_
         << " window=" << task.window_id << " KF=" << task.keyframe_id << " committed=0 reason=" << reason
         << " ms=" << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cerr << line.str() << '\n';
    return false;
  };
  if (!selector_network_ || !selector_optimizer_ || selector_target_ratios_.empty() ||
      gaussians_->xyz_.size(0) == 0 ||
      !hasNormalOptimizationBudget())
    return skip("training_unavailable");

  // Replay reuses the normal LR schedule without advancing its clocks.
  gaussians_->updateLearningRate(gradient_constraint_config_.enabled
      ? static_cast<int>(normal_update_clock_.gaussianLRStep(getIteration())) : getIteration());
  gaussians_->setFeatureLearningRate(featureLearningRate());
  gaussians_->setOpacityLearningRate(opacityLearningRate());
  gaussians_->setScalingLearningRate(scalingLearningRate());
  gaussians_->setRotationLearningRate(rotationLearningRate());
  const auto ratio_index = replay_ratio_cursor_++ % selector_target_ratios_.size();
  const float ratio = selector_target_ratios_[ratio_index];
  auto gt = camera->original_image_.to(gaussians_->xyz_.device());
  auto l1_loss = opt_params_.smooth_l1_ ? loss_utils::smooth_l1_loss : loss_utils::l1_loss;
  auto rendered = GaussianRenderer::render(camera, camera->image_height_, camera->image_width_,
      gaussians_, pipe_params_, background_, override_color_, 1.0f, false, torch::Tensor(), false, false);
  auto full_loss = (1.0 - lambdaDssim()) * l1_loss(rendered.image, gt, 1.0f) +
                   lambdaDssim() * (1.0 - loss_utils::fast_ssim(rendered.image, gt));
  if (opt_params_.opacity_reg_)
    full_loss += opt_params_.opacity_reg_ * gaussians_->getOpacityActivation().abs().mean();
  const double full_value = full_loss.detach().item<double>();
  if (!std::isfinite(full_value)) return skip("nonfinite_full_loss");
  full_loss.backward();
  const auto lod = accumulateLodGradients(camera, camera->image_height_, camera->image_width_,
                                         gt, ratio_index, false, false);
  if (!lod.finite_loss) return skip("nonfinite_lod_loss");
  const double training_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  const auto preview_start = std::chrono::steady_clock::now();
  auto cuda_groups = transform_field_ && transform_field_->config().backend == "cuda" &&
      shared_transform_optimizer_configured_
      ? std::vector<std::size_t>{shared_transform_mlp_group_index_, shared_transform_grid_group_index_}
      : std::vector<std::size_t>{};
  if (selector_backend_ == "cuda" && shared_selector_optimizer_configured_)
    cuda_groups.push_back(shared_selector_group_index_);
  AdamStepPreview preview(optimizer, cuda_groups);
  if (preview.parameters().empty()) return skip("no_gradients");
  if (!preview.finite()) return skip("nonfinite_adam");
  auto candidate = preview.candidate(preview.delta(), 1.0);
  if (!preview.finiteCandidate(candidate)) return skip("nonfinite_candidate");
  if (!preview.changed(candidate)) return skip("zero_update");
  const double preview_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - preview_start).count();
  const auto quality_start = std::chrono::steady_clock::now();

  // Diagnostics use identical masks/noise before and after. They do not recurse
  // into Guard or impose a new quality veto on the compensating Replay update.
  ScopedTorchRNG cpu_rng(torch::Device(torch::kCPU));
  std::unique_ptr<ScopedTorchRNG> device_rng;
  if (gaussians_->xyz_.device().is_cuda())
    device_rng = std::make_unique<ScopedTorchRNG>(gaussians_->xyz_.device());
  std::vector<GuardObjective> objectives{{"replay_full", camera->fid_, true, 1.0f},
                                       {"replay_lod", camera->fid_, false, ratio}};
  for (const auto& objective : last_guard_objectives_) {
    const bool duplicate = std::any_of(objectives.begin(), objectives.end(), [&](const GuardObjective& existing) {
      return existing.keyframe_id == objective.keyframe_id && existing.full == objective.full &&
             existing.ratio == objective.ratio;
    });
    if (!duplicate) objectives.push_back(objective);
  }
  const auto evaluation = makeHistoricalEvaluation(objectives);
  torch::NoGradGuard no_grad;
  std::vector<double> before, after;
  for (std::size_t j = 0; j < evaluation.objectives.size(); ++j)
    before.push_back(evaluation.evaluate(j, preview, nullptr).item<double>());
  evaluation.prepare_candidate(preview, candidate);
  for (std::size_t j = 0; j < evaluation.objectives.size(); ++j)
    after.push_back(evaluation.evaluate(j, preview, &candidate).item<double>());
  for (const auto* losses : {&before, &after})
    for (double loss : *losses) if (!std::isfinite(loss)) return skip("nonfinite_quality");
  const double quality_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - quality_start).count();
  const auto update_ratios = preview.updateRatios(candidate);
  preview.commit(candidate);
  ++replay_commit_count_;
  const bool tf_updated = shared_transform_optimizer_configured_ &&
      (update_ratios[shared_transform_mlp_group_index_] > 0.0 ||
       update_ratios[shared_transform_grid_group_index_] > 0.0);
  if (tf_updated) ++replay_tf_updates_;
  bool gaussian_updated = false;
  for (std::size_t group = 0; group < 6; ++group) gaussian_updated |= update_ratios[group] > 0.0;
  if (gaussian_updated && ultra_light_mode_ == 1) {
    gaussians_->protection_.markTouched(rendered.visibility_indices);
    gaussians_->protection_.markTouched(lod.supervised_indices);
  }
  const bool selector_updated = shared_selector_optimizer_configured_ &&
      update_ratios[shared_selector_group_index_] > 0.0;
  if (lod.supervised_indices.defined())
    gaussians_->updateElasticSupervision(lod.supervised_indices, static_cast<std::int64_t>(mapping_iter_));
  std::ostringstream line;
  line << std::fixed << std::setprecision(4) << "[ConflictReplay UPDATE] iter=" << mapping_iter_
       << " window=" << task.window_id << " KF=" << task.keyframe_id << " ratio=" << ratio
       << " committed=1 loss(full/lod)=" << full_value << '/' << lod.loss
       << " changed(GS/Selector/TF)=" << gaussian_updated << '/' << selector_updated << '/' << tf_updated;
  for (std::size_t j = 0; j < before.size(); ++j)
    line << ' ' << evaluation.objectives[j].name << "(MSE_x1e6 before/committed)="
         << before[j] * 1e6 << '/' << after[j] * 1e6;
  line << " counts(attempt/commit/skip)=" << replay_attempt_count_ << '/' << replay_commit_count_
       << '/' << replay_skip_count_ << " pending=" << conflict_replay_manager_->queueSize()
       << " quality_renders=" << 2 * evaluation.objectives.size()
       << " stage_host_ms(training/preview/quality)=" << training_ms << '/' << preview_ms << '/' << quality_ms
       << " ms=" << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  std::cout << line.str() << '\n';
  return true;
}

GaussianMapper::LodGradientResult GaussianMapper::accumulateLodGradients(
    std::shared_ptr<GaussianKeyframe> viewpoint_cam,
    int image_height,
    int image_width,
    torch::Tensor& gt_image,
    std::size_t ratio_index,
    bool detach_gaussian_parameters,
    bool record_supervision,
    UltrafastTrainingTimer* rotation_timer,
    bool normal_mode2,
    bool native_lod) {
  if (normal_mode2) detach_gaussian_parameters = true;
  LodGradientResult result;
  if(real_lod_.enabled&&!real_lod_.optimizingLod())return result;
  if (!viewpoint_cam || !selector_network_ || !selector_optimizer_ ||
      selector_target_ratios_.empty())
    return result;
  if (!hasNormalOptimizationBudget())
    return result;

  const auto num_gaussians = gaussians_->getXYZ().size(0);
  if (num_gaussians == 0 && !rotation_timer) return result;
  if (recent_window_importance_ && recent_window_importance_->enabled())
    recent_window_importance_->assertAligned(gaussians_->getXYZ());
  if (global_gi_) global_gi_->assertAligned(gaussians_->getXYZ());

  const bool hierarchical = global_gi_config_.hierarchical_representative_enabled;
  // This experiment explicitly uses effective maturity=1; disabled path stays legacy.
  const auto maturity = hierarchical ? torch::ones_like(gaussians_->protection_.maturity)
                                     : gaussians_->protection_.maturity.detach();
  if (geometry_marginal_) geometry_marginal_->assertAligned(num_gaussians);
  const float entry = hierarchical ? 0.0f : transform_field_config_.elastic_entry_maturity;
  const auto entry_indices = entry > 0 ? gaussians_->protection_.indicesAbove(entry) : torch::Tensor{};
  int64_t selector_candidates = entry > 0 ? entry_indices.numel() : num_gaussians;
  // Log a complete ratio rotation; multiples of 1000 alone land on Full.
  const bool log_budget = mapping_iteration_logging_ && record_supervision && (mapping_iter_ % 1000 < 5 ||
      (training_report_interval_ > 0 && mapping_iter_ % training_report_interval_ < 5));
  ratio_index %= selector_target_ratios_.size();
  const float target_ratio = selector_target_ratios_[ratio_index];

  initializeTransformFieldIfNeeded();
  ensureSharedMappingOptimizer();
  if (transform_field_config_.elastic_entry_maturity > 0 || selector_anchor_ratio_features_ || selector_compensation_method_=="anchor_ratio_latent14" || selector_compensation_method_=="s00r00e08d12" || selector_compensation_method_=="s00r00e50d12" || (selector_compensation_method_=="spatial_physical19_ratio32"||(((selector_compensation_method_=="spatial_offset3_ratio48"||selector_compensation_method_=="spatial_offset3_residual48"||selector_compensation_method_=="spatial_offset3_bandwidth48")||selector_compensation_method_=="spatial_physical19_ratio54")||(selector_compensation_method_=="spatial_anchor128_gate"||(selector_compensation_method_=="spatial_anchor128_offset_gate"||(selector_compensation_method_=="spatial_anchor128_offset_bound2"||selector_compensation_method_=="spatial_anchor128_offset_bound2_shared_selector")))))) selector_optimizer_->zero_grad(true);
  else selector_optimizer_->zero_grad();
  if (transform_optimizer_) {
    updateTransformLearningRate();
    transform_optimizer_->zero_grad(/*set_to_none=*/true);
  }

  auto xyz = detach_gaussian_parameters ? gaussians_->getXYZ().detach()
                                        : gaussians_->getXYZ();
  auto scaling = detach_gaussian_parameters
                     ? gaussians_->getScalingActivation().detach()
                     : gaussians_->getScalingActivation();
  auto log_scaling = detach_gaussian_parameters
                     ? gaussians_->scaling_.detach() : gaussians_->scaling_;
  auto rotation = detach_gaussian_parameters
                      ? gaussians_->getRotationActivation().detach()
                      : gaussians_->getRotationActivation();
  auto opacity = detach_gaussian_parameters
                     ? gaussians_->getOpacityActivation().detach()
                     : gaussians_->getOpacityActivation();
  auto features_dc = detach_gaussian_parameters
                         ? gaussians_->features_dc_.detach()
                         : gaussians_->features_dc_;
  auto features_rest = detach_gaussian_parameters
                           ? gaussians_->features_rest_.detach()
                           : gaussians_->features_rest_;
  auto& selector_profile = continuous_tf_statistics_.ratios[target_ratio].selector_profiler;
  if (protection_config_.performance_logging) selector_profile.beginIteration();
  if (selector_slot_deactivation_) selector_network_->anchors()->refreshSlotActivity(opacity);
  torch::Tensor shared_features, selection_logits;
  torch::Tensor mixed_probability,selector_mask,fused_selected_ids;
  torch::Tensor anchor_candidate_mask;
  torch::Tensor context_slot_eligibility;
  int64_t visible_anchor_count=0;
  const bool anchor_frustum=selector_anchor_frustum_culling_ && selector_explicit_anchor_blocks_ &&
      selector_network_->anchorConditioned();
  const bool fast_flexgs=fused_flexgs_selector_ && selector_selection_mode_=="flexgs" &&
      (!selector_protection_enabled_ || protection_config_.reliability_v2) && !hierarchical && entry==0 && xyz.is_cuda() &&
      xyz.scalar_type()==torch::kFloat32 && selector_network_->anchorConditioned() &&
      selector_anchor_feature_dim_==0 && !selector_network_->selectorSpatialFeatures() && (!selector_network_->selectorSetEncoder() || selector_network_->selectorSetEncoderSchema()==3) && selector_anchor_gaussian_attributes_ && !selector_anchor_global_xyz_ &&
      !selector_anchor_ratio_features_ && (!selector_compensation_ || selector_network_->spatialCompensationEnabled()) &&
      mapping_cuda::SelectorExecutionDispatcher::plan(num_gaussians,xyz.get_device()).supported();
  torch::Tensor maturity_weights;
  if(anchor_frustum) {
    auto timing=mapping_profiler_.scope(MappingProfiler::SelectorForward);
    auto& anchors=selector_network_->anchors();
    torch::Tensor rows,eligible;
    {
      torch::NoGradGuard guard;
      auto visible=anchorNominalFrustumMask(anchors->centers(),anchors->offsetScales(),viewpoint_cam->full_proj_transform_);
      if(anchors->variableMembers())visible=visible & (anchors->memberCounts()>0);
      anchor_candidate_mask=visible.index_select(0,anchors->rows());
      if(entry>0)anchor_candidate_mask=anchor_candidate_mask & (maturity>=entry);
      rows=torch::nonzero(anchor_candidate_mask).reshape({-1});
      eligible=selector_network_->maskInactiveSlots(torch::ones({num_gaussians},xyz.options()),opacity).to(torch::kBool);
      if(cached_anchor_context_ && selector_network_->selectorContextResidual())context_slot_eligibility=eligible;
      eligible=eligible.index_select(0,rows);
      if(log_budget)visible_anchor_count=visible.sum().item<int64_t>();
    }
    selector_candidates=rows.numel();
    mixed_probability=torch::zeros({num_gaussians},xyz.options());
    selector_mask=torch::zeros_like(mixed_probability);
    fused_selected_ids=rows;
    if(selector_candidates) {
      const bool compact_fast_flexgs=fast_flexgs &&
          mapping_cuda::SelectorExecutionDispatcher::plan(selector_candidates,xyz.get_device()).supported();
      const auto positions=xyz.index_select(0,rows), scales=scaling.index_select(0,rows);
      const auto dc=features_dc.index_select(0,rows), op=opacity.index_select(0,rows);
      if(compact_fast_flexgs || selector_compensation_ || selector_network_->selectorSetEncoder() || selector_network_->selectorContextResidual()) {
        auto selection=compact_fast_flexgs?selector_network_->forwardFlexgsCUDA(positions,target_ratio,scales,dc,op,rows,eligible):
            selector_network_->forwardSelectionFeatures(positions,target_ratio,selector_temperature_,rows,scales,dc,op);
        selection_logits=selection.logits;
        if(!selector_network_->spatialCompensationEnabled())
          shared_features=torch::zeros({num_gaussians,selection.shared.size(1)},selection.shared.options()).index_copy(0,rows,selection.shared);
      } else selection_logits=selector_network_->forwardLogits(positions,rotation.index_select(0,rows),scales,
          target_ratio,rows,dc,op);
      torch::Tensor probability,mask,selected;
      if(selector_selection_mode_=="flexgs") {
        auto exponentials=torch::empty_like(selection_logits).exponential_();
        if(selection_logits.is_cuda()) {
          auto sampled=protection_config_.reliability_v2 && reuse_maturity_softmax_ && selector_temperature_==1.f?
              mapping_cuda::flexgsSampleCompactWithWeights(selection_logits,exponentials,eligible,selector_temperature_):
              mapping_cuda::flexgsSampleCompact(selection_logits,exponentials,eligible,selector_temperature_);
          probability=sampled[0];mask=sampled[1];selected=sampled[2];
          if(protection_config_.reliability_v2 && protection_config_.reliability_loss_weight>0)
            maturity_weights=sampled.size()==4?sampled[3].select(1,1):
                ((selection_logits-exponentials.log())/selector_temperature_).softmax(-1).select(1,1);
        } else {
          probability=(selection_logits/selector_temperature_).softmax(-1).select(1,1)*eligible;
          mask=sampleSelectorGumbelST(selection_logits,selector_temperature_,-exponentials.log())*eligible;
          selected=torch::nonzero(mask.detach()>0).reshape({-1});
          if(protection_config_.reliability_v2 && protection_config_.reliability_loss_weight>0)
            maturity_weights=((selection_logits-exponentials.log())/selector_temperature_).softmax(-1).select(1,1);
        }
      } else {
        auto weight=entry>0?elasticMaturity(maturity.index_select(0,rows),entry):
            gaussians_->protection_.selectionMaturity(selector_protection_enabled_).index_select(0,rows);
        probability=mixSelectorProbability((selection_logits/selector_temperature_).softmax(-1).select(1,1),weight,target_ratio)*eligible;
        mask=sampleSelectorST(probability,gaussians_->protection_.cached_uniform.index_select(0,rows))*eligible;
        selected=torch::nonzero(mask.detach()>0).reshape({-1});
      }
      // Restore global rows before budget/photo/GI consumers: sum(mask) targets r*N,
      // never r*N_candidates. No MLP work or straight-through gradients outside rows.
      mixed_probability=mixed_probability.index_copy(0,rows,probability);
      selector_mask=selector_mask.index_copy(0,rows,mask);
      fused_selected_ids=rows.index_select(0,selected);
      if(maturity_weights.defined())
        maturity_weights=torch::zeros({num_gaussians},xyz.options()).index_copy(0,rows,maturity_weights);
    }
    if(protection_config_.performance_logging)selector_profile.endIteration();
  } else if(protection_config_.reliability_v2) {
    auto timing=mapping_profiler_.scope(MappingProfiler::SelectorForward);
    auto selection=fast_flexgs?selector_network_->forwardFlexgsCUDA(xyz,target_ratio,scaling,features_dc,opacity):
        selector_network_->forwardSelectionFeatures(xyz,target_ratio,selector_temperature_,{},scaling,features_dc,opacity);
    selection_logits=selection.logits;shared_features=selection.shared;
    auto eligible=selector_network_->maskInactiveSlots(torch::ones({num_gaussians},xyz.options()),opacity).to(torch::kBool);
    auto exponentials=torch::empty_like(selection_logits).exponential_();
    auto sampled=reuse_maturity_softmax_&&selector_temperature_==1.f?
        mapping_cuda::flexgsSampleCompactWithWeights(selection_logits,exponentials,eligible,selector_temperature_):
        mapping_cuda::flexgsSampleCompact(selection_logits,exponentials,eligible,selector_temperature_);
    mixed_probability=sampled[0];selector_mask=sampled[1];fused_selected_ids=sampled[2];
    // Exactly the same Gumbel draws as the actual hard-forward sampler.
    if(protection_config_.reliability_loss_weight>0)
      maturity_weights=sampled.size()==4?sampled[3].select(1,1):
          ((selection_logits-exponentials.log())/selector_temperature_).softmax(-1).select(1,1);
  } else if(fast_flexgs) {
    auto timing=mapping_profiler_.scope(MappingProfiler::SelectorForward);
    auto selection=selector_network_->forwardFlexgsCUDA(xyz,target_ratio,scaling,features_dc,opacity);
    selection_logits=selection.logits;shared_features=selection.shared;
    auto eligible=selector_network_->maskInactiveSlots(torch::ones({num_gaussians},xyz.options()),opacity).to(torch::kBool);
    auto exponentials=torch::empty_like(selection_logits).exponential_();
    auto sampled=mapping_cuda::flexgsSampleCompact(selection_logits,exponentials,eligible,selector_temperature_);
    mixed_probability=sampled[0];selector_mask=sampled[1];fused_selected_ids=sampled[2];
  } else {
  mixed_probability = elasticSelectorProbability(maturity,
      hierarchical ? maturity : gaussians_->protection_.selectionMaturity(selector_protection_enabled_), entry, target_ratio, entry_indices,
      [&](const torch::Tensor& indices) {
        const auto positions = indices.defined() ? xyz.index_select(0, indices) : xyz;
        const auto rotations = indices.defined() ? rotation.index_select(0, indices) : rotation;
        const auto scales = indices.defined() ? scaling.index_select(0, indices) : scaling;
        ProtectionProfiler::Scope timer(gaussians_->protection_.timing(), ProtectionProfiler::SelectorAllRows, xyz.is_cuda());
        ProtectionProfiler::Scope ratio_timer(protection_config_.performance_logging ? &selector_profile : nullptr,
            ProtectionProfiler::SelectorAllRows, xyz.is_cuda());
        auto selector_timing = mapping_profiler_.scope(MappingProfiler::SelectorForward);
        if(selector_compensation_) {
          auto selection=selector_network_->forwardSelectionFeatures(positions,target_ratio,selector_temperature_,indices,scales,
              indices.defined()?features_dc.index_select(0,indices):features_dc,
              indices.defined()?opacity.index_select(0,indices):opacity);
          TORCH_CHECK(!indices.defined(),"Integrated compensation requires the full fixed-maturity population");
          shared_features=selection.shared;
          selection_logits=selection.logits;
          return selection.probability;
        }
        if(selector_selection_mode_ == "flexgs") {
          selection_logits=selector_network_->forwardLogits(positions,rotations,scales,target_ratio,indices,
              indices.defined()?features_dc.index_select(0,indices):features_dc,
              indices.defined()?opacity.index_select(0,indices):opacity);
          return (selection_logits/selector_temperature_).softmax(-1).select(1,1);
        }
        if (selector_network_->anchorConditioned())
          return selector_network_->forwardAnchorProbability(positions, target_ratio, selector_temperature_, indices,
              scales, indices.defined() ? features_dc.index_select(0, indices) : features_dc,
              indices.defined() ? opacity.index_select(0, indices) : opacity);
        return selector_network_->forwardProbability(positions, rotations, scales, target_ratio, selector_temperature_);
      });
  if (protection_config_.performance_logging) selector_profile.endIteration();
  if(selector_selection_mode_ == "flexgs") {
    // The same noisy straight-through mask drives photo loss and the FlexGS budget.
    mixed_probability=selector_network_->maskInactiveSlots(mixed_probability,opacity);
    selector_mask=selector_network_->maskInactiveSlots(
        sampleSelectorGumbelST(selection_logits,selector_temperature_),opacity);
  }else if(fused_selector_compaction_ && mixed_probability.is_cuda() && mixed_probability.scalar_type()==torch::kFloat32){
    const auto inactive=selector_network_->anchorConditioned()
        ? selector_network_->anchors()->inactiveSlots() : torch::Tensor();
    const auto sampled=mapping_cuda::selectorSampleCompact(mixed_probability,gaussians_->protection_.cached_uniform,opacity,inactive);
    mixed_probability=sampled[0];selector_mask=sampled[1];fused_selected_ids=sampled[2];
  }else{
    mixed_probability = selector_network_->maskInactiveSlots(mixed_probability,opacity);
    selector_mask = sampleSelectorST(mixed_probability, gaussians_->protection_.cached_uniform);
  }
  }
  const auto ratio_loss = selector_selection_mode_ == "flexgs"
      ? selectorGumbelRatioLoss(selector_mask, target_ratio, selector_gumbel_budget_normalize_)
      : selectorRatioLoss(mixed_probability, target_ratio);

  GITeacher gi_teacher;
  {
    auto timing = mapping_profiler_.scope(MappingProfiler::GITeacher);
  if (flexgs_importance_enabled_ && real_lod_.optimizingLod()) {
    TORCH_CHECK(flexgs_importance_.defined()&&flexgs_importance_.numel()==num_gaussians&&
        flexgs_importance_topology_==topology_version_,"FlexGS importance must match the current topology");
    gi_teacher.label=flexgsImportanceTarget(flexgs_importance_,target_ratio);
    gi_teacher.weight=torch::ones_like(flexgs_importance_);
    gi_teacher.supervised_mask=torch::ones_like(flexgs_importance_,flexgs_importance_.options().dtype(torch::kBool));
    gi_teacher.supervised_k=num_gaussians;gi_teacher.target_k=static_cast<int64_t>(target_ratio*num_gaussians);
    gi_teacher.gi_version=flexgs_importance_version_;gi_teacher.topology_version=topology_version_;
  } else if (global_gi_) {
    // Valid cache estimates and actual observation are independent conditions.
    const bool marginal = global_gi_config_.mode == "streaming_marginal_geo" &&
        geometry_marginal_ && geometry_marginal_->ready();
    hierarchical_si::Result hierarchy;
    const auto teacher_score = hierarchical ? hierarchical_si::score(true, xyz, scaling,
        global_gi_->importance(), global_gi_->validMask(), global_gi_->observedMask(), &hierarchy)
        : (marginal ? geometry_marginal_->score() : global_gi_->importance());
    if (hierarchical && global_gi_config_.hierarchical_representative_diagnostics)
      hierarchical_si::writeDiagnostics(hierarchy, result_dir_ / "hierarchical_si",
          "iteration_" + std::to_string(mapping_iter_) + "_ratio_" + std::to_string(target_ratio));
    if (hierarchical && mapping_iter_ % 1000 < 5) {
      std::cout << "[HierarchicalSI] iteration=" << mapping_iter_ << " candidates=" << hierarchy.candidates
                << " nodes=" << hierarchy.nodes.size() << " raw_sum=" << hierarchy.raw_sum
                << " hierarchical_sum=" << hierarchy.output_sum << " mass_relative=" << hierarchy.conservation_relative
                << " build_ms=" << hierarchy.build_ms << " effective_maturity=1\n";
    }
    gi_teacher = gi_teacher_builder_.build(teacher_score,
        marginal ? geometry_marginal_->valid() : global_gi_->validMask(),
        marginal ? geometry_marginal_->valid() : global_gi_->observedMask(), maturity, target_ratio,
        marginal ? geometry_marginal_->version() : global_gi_->version(), global_gi_->topologyVersion(), entry);
    if (global_gi_config_.marginal_diagnostics && mapping_iter_ % 1000 < 5 && gi_teacher.valid()) {
      std::ofstream trace(result_dir_ / "gi_teacher_trace.csv", std::ios::app);
      if (trace.tellp() == 0) trace << "iteration,ratio,supervised,positive,weight_sum,compatibility_mature,tf_ready,zero_maturity,snapshot_age,marginal\n";
      trace << mapping_iter_ << ',' << target_ratio << ',' << gi_teacher.supervised_k << ','
            << gi_teacher.label.sum().item<int64_t>() << ',' << gi_teacher.weight.sum().item<double>() << ','
            << gaussians_->protection_.matureMask().sum().item<int64_t>() << ','
            << (maturity >= transform_field_config_.ready_maturity).sum().item<int64_t>() << ','
            << (maturity == 0).sum().item<int64_t>() << ','
            << (marginal ? mapping_iter_ - geometry_marginal_->publishedIteration() : 0) << ',' << marginal << '\n';
    }
  } else if (recent_window_importance_ && recent_window_config_.collect_importance &&
             recent_window_importance_->enabled()) {
    auto combined_gi = recent_window_importance_->getImportance();
    if (recent_window_importance_->hasValidContribution() &&
        combined_gi.defined() && combined_gi.size(0) == num_gaussians)
      gi_teacher = gi_teacher_builder_.build(
          combined_gi, torch::isfinite(combined_gi),
          recent_window_importance_->observedMask(), maturity, target_ratio,
          recent_window_importance_->version(), recent_window_importance_->topologyVersion(), entry);
  }
  }

  auto l1_loss =
      opt_params_.smooth_l1_ ? loss_utils::smooth_l1_loss : loss_utils::l1_loss;
  torch::Tensor gi_loss;
  if (gi_teacher.valid() && gi_teacher.supervised_k > 0 &&
      selector_gi_loss_weight_ != 0.0f) {
    auto timing = mapping_profiler_.scope(MappingProfiler::GILoss);
    gi_loss = flexgs_importance_enabled_ ? (selector_mask-gi_teacher.label).abs().mean()
                                        : gi_teacher.loss(selector_mask, opt_params_.smooth_l1_);
  }

  // This map belongs only to this LOD forward/backward. Master rows and all
  // global Selector/GI/ratio losses keep their original population.
  const auto compact_to_global =
      fused_selected_ids.defined() ? fused_selected_ids : torch::nonzero(selector_mask.detach() > 0.0f).reshape({-1});
  const int64_t compact_count = compact_to_global.numel();
  if(protection_config_.reliability_v2 && mapping_iter_%250<7) {
    torch::NoGradGuard guard;auto& p=gaussians_->protection_;auto chosen=maturity.index_select(0,compact_to_global);
    auto scalar=[&](double v){return torch::full({},v,maturity.options());};
    auto sum=chosen.sum()/std::max<int64_t>(1,compact_count);
    auto alpha=reliabilityAlpha(maturity,protection_config_.reliability_gate_low,protection_config_.reliability_gate_high);
    auto O=p.optimization_support;auto V=p.view_support;
    auto pool=anchor_candidate_mask.defined()?maturity.index({anchor_candidate_mask}):maturity;
    reliability_trace_.push_back(torch::stack({scalar(mapping_iter_),scalar(target_ratio),scalar(num_gaussians),scalar(compact_count),
        anchor_candidate_mask.defined()?pool.sum()/std::max<int64_t>(1,pool.numel()):maturity.mean(),sum,(maturity<.4).to(torch::kFloat32).mean(),(maturity>=.8).to(torch::kFloat32).mean(),
        O.mean(),V.mean(),alpha.mean(),mixed_probability.mean()}));
  }
  result.active_gaussians = compact_count;
  if(lod_health_guard_.enabled) {
    auto eligible=selector_network_->maskInactiveSlots(torch::ones_like(maturity),opacity).to(torch::kBool);
    if(anchor_candidate_mask.defined())eligible=eligible&anchor_candidate_mask;
    auto reason=lod_health_guard_.selection(ratio_index,mapping_iter_+1,target_ratio,num_gaussians,
        compact_count,mixed_probability,eligible);
    lod_health_guard_.sample["lod_step"]=Json::UInt64(real_lod_.lod_steps+1);
    if(!reason.empty())failLodHealth(reason);
  }
  GaussianRenderInput full_input{
      xyz, opacity, scaling, rotation, features_dc, features_rest,
      gaussians_->active_sh_degree_, gaussians_->max_sh_degree_};
  auto selected_input = full_input.indexSelect(compact_to_global);
  const auto compact_mask = selector_mask.index_select(0, compact_to_global);
  int64_t transform_count = 0;
  bool fused_decode_used = false;
  if(selector_compensation_ && getIteration()>=selector_compensation_start_iter_ && compact_count>0) {
    auto compensation_timing=mapping_profiler_.scope(MappingProfiler::Compensation);
    auto logits=detach_gaussian_parameters?gaussians_->opacity_.detach():gaussians_->opacity_;
    if(selector_network_->spatialCompensationEnabled()) {
      auto& spatial=selector_network_->spatialCompensation();auto& anchors=selector_network_->anchors();
      if(!spatial->ready()) {
        spatial->initialize(xyz,log_scaling,rotation,features_dc,logits,
            anchors->centers().index_select(0,anchors->rows()),anchors->offsetScales().index_select(0,anchors->rows()),true);
        std::cout<<"[SpatialLocalXYZ] fixed normalization initialized at iteration="<<getIteration()<<" population="<<num_gaussians<<'\n';
      }
    }
    torch::Tensor alpha,comp_rows;
    if(protection_config_.reliability_gate||!elide_inactive_maturity_gate_) {
      auto selected_m=maturity.index_select(0,compact_to_global);
      alpha=protection_config_.reliability_gate?reliabilityAlpha(selected_m,
          protection_config_.reliability_gate_low,protection_config_.reliability_gate_high):torch::ones_like(selected_m);
      comp_rows=protection_config_.reliability_gate&&protection_config_.reliability_gate_appearance?
          torch::nonzero(alpha>0).reshape({-1}):torch::arange(compact_count,compact_to_global.options());
    }
    auto comp_ids=protection_config_.reliability_gate?compact_to_global.index_select(0,comp_rows):compact_to_global;
    auto comp_input=protection_config_.reliability_gate?full_input.indexSelect(comp_ids):selected_input;
    torch::Tensor residual;
    const auto decoded_rows=comp_rows.defined()?comp_rows.numel():compact_count;
    if(!decoded_rows)residual=torch::zeros({0,14},xyz.options());
    else
    if(selector_network_->spatialCompensationEnabled()) {
      auto& spatial=selector_network_->spatialCompensation();auto& anchors=selector_network_->anchors();
      auto owners=anchors->rows().index_select(0,comp_ids);
      auto centers=anchors->centers().index_select(0,owners);
      auto references=anchors->offsetScales().index_select(0,owners);
      auto local_offset=anchors->compensationResidualMode()==2?(comp_input.xyz-centers)/references:torch::Tensor{};
      torch::Tensor packed_residual;
      if(fused_residual48_input_&&!torch::GradMode::is_enabled()&&anchors->compensationResidualMode()==1) {
        packed_residual=mapping_cuda::residual48InferencePack(comp_input.xyz,centers,references,spatial->mean,spatial->stddev,
            owners,anchors->compensationFeatureTables(),anchors->compensationResidualWeights(target_ratio));
      }else if(fused_residual48_training_input_&&torch::GradMode::is_enabled()&&anchors->compensationResidualMode()==1) {
        mapping_cuda::FusionRange input_trace("Comp.Residual48TrainingPack");
        auto weights=anchors->compensationResidualWeights(target_ratio);
        auto tables=anchors->compensationFeatureTables();std::vector<torch::Tensor> features;
        for(size_t j=0;j<weights.size();++j)features.push_back(tables[j].index_select(0,owners));
        packed_residual=mapping_cuda::residual48TrainingPack(comp_input.xyz,centers,references,spatial->mean,spatial->stddev,features,weights);
      }
      residual=spatial->forward(comp_input.xyz,log_scaling.index_select(0,comp_ids),comp_input.rotation,comp_input.features_dc,
          logits.index_select(0,comp_ids),centers,references,target_ratio,true,
          packed_residual.defined()?torch::Tensor{}:(spatial->sharedAnchor128()?anchors->spatialFeaturesTable().index_select(0,owners):(spatial->ratioFeatureWidth()?anchors->compensationInput(comp_ids,target_ratio,local_offset):(anchors->spatialFeatures()?anchors->spatialFeaturesTable().index_select(0,owners):torch::Tensor{}))),packed_residual);
    } else residual=selector_network_->compensationResidual(shared_features.index_select(0,comp_ids),comp_ids,target_ratio);
    if(protection_config_.reliability_gate)
      residual=reliabilityGate(residual,alpha.index_select(0,comp_rows),protection_config_.reliability_gate_appearance);
    if(protection_config_.reliability_gate)residual=torch::zeros({compact_count,14},residual.options()).index_copy(0,comp_rows,residual);
    if(fused_compensation_decode_ && residual.size(1)==14 && residual.is_cuda() && residual.scalar_type()==torch::kFloat32) {
      auto logits=detach_gaussian_parameters?gaussians_->opacity_.detach():gaussians_->opacity_;
      const bool physical=selector_network_->spatialCompensationEnabled()&&selector_network_->spatialCompensation()->physicalScaleAdditive();
      const bool softplus=selector_network_->spatialCompensationEnabled()&&selector_network_->spatialCompensation()->softplusScale();
      auto decoded=(softplus?mapping_cuda::decode14SoftplusScale:(physical?mapping_cuda::decode14AdditiveScale:mapping_cuda::decode14LogScale))(selected_input.xyz,log_scaling.index_select(0,compact_to_global),selected_input.rotation,
          logits.index_select(0,compact_to_global),selected_input.features_dc,residual,compact_mask);
      selected_input.xyz=decoded[0];selected_input.scaling=decoded[1];selected_input.rotation=decoded[2];
      selected_input.opacity=decoded[3];selected_input.features_dc=decoded[4];
      fused_decode_used=true;
    } else {
    auto transformed=improvements::selector::GumbelNetworkImpl::applyCompensationResidual(
        {selected_input.xyz,log_scaling.index_select(0,compact_to_global),selected_input.rotation},residual);
    selected_input.xyz=transformed.xyz;
    selected_input.scaling=selector_network_->spatialCompensationEnabled()?
        selector_network_->spatialCompensation()->decodeScale(log_scaling.index_select(0,compact_to_global),residual.slice(1,3,6)):transformed.log_scaling.exp();
    selected_input.rotation=transformed.rotation;
    if(residual.size(1)==14) {
      auto logits=detach_gaussian_parameters?gaussians_->opacity_.detach():gaussians_->opacity_;
      auto appearance=improvements::selector::GumbelNetworkImpl::applyAppearanceResidual(
          selected_input.features_dc,logits.index_select(0,compact_to_global),residual);
      selected_input.features_dc=appearance.first;
      selected_input.opacity=appearance.second.sigmoid();
    }
    }
    result.compensated_gaussians=compact_count;
    compensation_forwards_+=decoded_rows>0;
    compensation_decoded_rows_+=decoded_rows;
    if(mapping_iter_%100==0 || (selector_gumbel_budget_normalize_ && mapping_iter_%100<4))
      std::cout<<"[Selector Compensation] iteration="<<getIteration()<<" population="<<num_gaussians
             <<" candidates="<<selector_candidates
             <<" selected="<<compact_count<<" decoded="<<residual.size(0)<<" ratio="<<target_ratio<<'\n';
  }
  // Gate the compensated opacity once. This preserves the Selector's ST
  // photometric gradient even though selected forward mask values are one.
  if(lod_health_guard_.enabled && compact_count>0 && selector_network_->spatialCompensationEnabled()) {
    auto& anchors=selector_network_->anchors();auto reference=anchors->offsetScales().index_select(0,
        anchors->rows().index_select(0,compact_to_global));
    auto reason=lod_health_guard_.compensation(selected_input.xyz,selected_input.scaling,selected_input.opacity,
        xyz.index_select(0,compact_to_global),scaling.index_select(0,compact_to_global),reference);
    if(!reason.empty())failLodHealth(reason);
  }
  if(lod_health_guard_.enabled)lod_health_guard_.record(result_dir_);
  if(!fused_decode_used)selected_input.opacity=compact_mask.unsqueeze(1)*selected_input.opacity;
  auto inside = torch::zeros({compact_count}, xyz.options().dtype(torch::kBool));
  auto candidate_maturity = maturity.slice(0, 0, 0);
  auto tf_before_epsilon = inside;
  if (transform_field_ && compact_count > 0) {
    inside = transform_field_->insideMask(selected_input.xyz);
    const auto compact_maturity = maturity.index_select(0, compact_to_global);
    tf_before_epsilon = entry > 0 ? inside & (compact_maturity >= entry) : inside;
    const auto transform_mask = transform_field_->candidateMask(selected_input.xyz, compact_maturity);
    const auto tf_local_indices = torch::nonzero(transform_mask).reshape({-1});
    transform_count = tf_local_indices.numel();
    if (transform_count > 0) {
      candidate_maturity = elasticMaturity(compact_maturity.index_select(0, tf_local_indices), entry);
      if (continuous_tf_statistics_.tf_start_mapping_iteration < 0)
        continuous_tf_statistics_.tf_start_mapping_iteration = mapping_iter_ + 1;
      ProtectionProfiler::Scope tf_timer(gaussians_->protection_.timing(), ProtectionProfiler::TFForward, xyz.is_cuda());
      tf_cuda::ScopedTrainingProfile tf_profile(gaussians_->protection_.timing());
      if (rotation_timer && ultrafast_tf_first_applied_ < 0) {
        ultrafast_tf_first_applied_ = mapping_iter_ + 1;
        std::cout << "[UltrafastRotation TF] event=first_applied_lod mapping_iteration="
                  << mapping_iter_ + 1 << " normal_training_step=" << ultrafast_rotation_.step << '\n';
      }
      auto transformed = transform_field_->applyResidual(
          selected_input.xyz.index_select(0, tf_local_indices),
          selected_input.scaling.index_select(0, tf_local_indices),
          selected_input.rotation.index_select(0, tf_local_indices),
          target_ratio, candidate_maturity);
      selected_input.xyz = selected_input.xyz.index_copy(
          0, tf_local_indices, transformed.xyz);
      selected_input.scaling = selected_input.scaling.index_copy(
          0, tf_local_indices, transformed.scaling);
      selected_input.rotation = selected_input.rotation.index_copy(
          0, tf_local_indices, transformed.rotation);
    }
  }

  torch::Tensor selector_loss;
  result.transformed_gaussians = transform_count;
  int64_t visible_count = 0;
  if (compact_count > 0 || rotation_timer || normal_mode2 || native_lod) {
    const bool slam_online = pSLAM_ && !pSLAM_->isShutDown();
    const bool post_slam = pSLAM_ && !slam_online;
    const float lod_render_scale = native_lod ? 1.0f :
        ((normal_mode2 || slam_online) ? selector_lod_render_scale_ : 1.0f);
    // Tail optimization uses native images even if a KF still has pyramid uses.
    const int source_height = (native_lod || normal_mode2 || post_slam) ? viewpoint_cam->image_height_ : image_height;
    const int source_width = (native_lod || normal_mode2 || post_slam) ? viewpoint_cam->image_width_ : image_width;
    const int lod_height = std::max(
        1, static_cast<int>(source_height * lod_render_scale));
    const int lod_width = std::max(
        1, static_cast<int>(source_width * lod_render_scale));
    result.render_height = lod_height;
    result.render_width = lod_width;
    if(native_lod) {
      TORCH_CHECK(lod_height==viewpoint_cam->image_height_&&lod_width==viewpoint_cam->image_width_,"Native LoD dimensions changed");
      if(mapping_iter_%100<5)std::cout<<"[LoD Native Audit] mapping="<<mapping_iter_+1<<" online="<<slam_online<<" height="<<lod_height<<" width="<<lod_width<<'\n';
    }
    auto lod_gt_image = (native_lod || normal_mode2 || post_slam)
        ? viewpoint_cam->original_image_.to(gt_image.device()) : gt_image;
    lod_gt_image = viewpoint_cam->resized_image_cache_.get(
        lod_gt_image, lod_height, lod_width);
    if (rotation_timer) rotation_timer->beginRender();
    RenderPackage selected_render_pkg;
    {
      auto timing = mapping_profiler_.scope(MappingProfiler::LodRender);
      selected_render_pkg = GaussianRenderer::render(
          viewpoint_cam, lod_height, lod_width, selected_input,
          pipe_params_, background_, override_color_, 1.0f, false, false);
      if (rotation_timer) rotation_timer->endRender();
    }
    result.rendered = true;
    if (selected_render_pkg.visibility_indices.defined() &&
        selected_render_pkg.visibility_indices.numel() > 0) {
      // Renderer indices/radii/viewspace points are compact-local. Only the
      // supervision indices return to master space; full-render densification
      // statistics are deliberately untouched.
      const auto supervised_indices = compact_to_global.index_select(
          0, selected_render_pkg.visibility_indices);
      visible_count = supervised_indices.numel();
      result.supervised_indices = supervised_indices;
      if (record_supervision)
        gaussians_->updateElasticSupervision(
            supervised_indices, static_cast<std::int64_t>(mapping_iter_));
    }
    auto timing = mapping_profiler_.scope(MappingProfiler::LodLoss);
    auto selected_l1 = l1_loss(selected_render_pkg.image, lod_gt_image, 1.0f);
    auto selected_ssim = loss_utils::fast_ssim(selected_render_pkg.image, lod_gt_image);
    const float lambda_dssim = lambdaDssim();
    auto selected_photo = (1.0 - lambda_dssim) * selected_l1 +
                          lambda_dssim * (1.0 - selected_ssim);
    selector_loss = selector_render_loss_weight_ * selected_photo +
                    selector_ratio_loss_weight_ * ratio_loss;
  } else {
    selector_loss = selector_ratio_loss_weight_ * ratio_loss;
  }
  if (gi_loss.defined()) {
    const float ratio_weight = importance_ratio_aware_weight_ &&
                                       ratio_index < importance_ratio_weights_.size()
                                   ? importance_ratio_weights_[ratio_index]
                                   : 1.0f;
    selector_loss += selector_gi_loss_weight_ * ratio_weight * gi_loss;
    if(mapping_iter_%1000<7) {
      std::cout<<(flexgs_importance_enabled_?"[FlexGS Teacher] mapping=":"[SGI Teacher] mapping=")<<mapping_iter_<<" ratio="<<target_ratio
          <<" supervised="<<gi_teacher.supervised_k<<" positive="<<gi_teacher.label.sum().item<float>()
          <<" weight_sum="<<gi_teacher.weight.sum().item<float>()<<" loss="<<gi_loss.detach().item<float>()
          <<" lambda="<<selector_gi_loss_weight_<<" gi_version="<<gi_teacher.gi_version
          <<" topology_version="<<gi_teacher.topology_version<<'\n';
    }
  }
  if (transform_count > 0) {
    ProtectionProfiler::Scope tf_timer(gaussians_->protection_.timing(), ProtectionProfiler::TFRegularization, xyz.is_cuda());
    tf_cuda::ScopedTrainingProfile tf_profile(gaussians_->protection_.timing());
    selector_loss += transform_field_->regularizationLoss();
  }
  auto anchor_si_loss=anchor_si_config_.enabled
      ? anchor_si_.loss(selector_network_,mixed_probability,opacity,ratio_index,topology_version_)
      : torch::Tensor();
  if(anchor_si_loss.defined())selector_loss=selector_loss+anchor_si_loss.detach();
  {mapping_cuda::FusionRange trace("Mapping.MetadataStatistics");
  continuous_tf_statistics_.record(target_ratio, compact_count, transform_count,
      inside, candidate_maturity, gi_teacher.supervised_k, gi_teacher.weight,
      selector_candidates, num_gaussians, tf_before_epsilon,fused_empty_tf_statistics_&&!transform_field_);}

  if (!record_supervision || rotation_timer || normal_mode2 || native_lod) {
    result.loss = selector_loss.detach().item<double>();
    result.finite_loss = std::isfinite(result.loss);
    if (!result.finite_loss) return result;
  }
  if (selector_loss.requires_grad()) {
    auto timing = mapping_profiler_.scope(MappingProfiler::LodBackward);
    if(anchor_si_loss.defined())anchor_si_.accumulateHeadGradients(anchor_si_loss,selector_network_);
    if(maturity_weights.defined()) {
      mapping_cuda::FusionRange trace("Mapping.MaturityPrivateBackward");
      auto eligible=context_slot_eligibility.defined()?context_slot_eligibility:
          selector_network_->maskInactiveSlots(torch::ones_like(maturity),opacity).to(torch::kBool);
      if(anchor_candidate_mask.defined())eligible=eligible & anchor_candidate_mask;
      auto mloss=fused_maturity_loss_&&maturity_weights.is_cuda()&&maturity_weights.scalar_type()==torch::kFloat32?
          mapping_cuda::maturityLossFused(maturity_weights,gaussians_->protection_.selectorReliability(),eligible,protection_config_.reliability_delta):
          reliabilitySelectionLoss(maturity_weights,gaussians_->protection_.selectorReliability(),eligible,protection_config_.reliability_delta);
      std::vector<torch::Tensor> parameters;
      for(const auto& p:selector_network_->named_parameters())
        if(p.key().rfind("local_emd.",0)==0 || p.key().rfind("soft_net.",0)==0 ||
           p.key().rfind("selector_set_encoder.",0)==0 ||
           p.key().rfind("selector_context_residual.",0)==0 ||
           p.key().rfind("selector_spatial_projection.",0)==0)parameters.push_back(p.value());
      auto gradients=torch::autograd::grad({mloss*protection_config_.reliability_loss_weight},parameters,{},true,false,true);
      torch::NoGradGuard guard;
      for(size_t j=0;j<parameters.size();++j)if(gradients[j].defined()) {
        if(parameters[j].grad().defined())parameters[j].mutable_grad().add_(gradients[j]);
        // Compact populations can use ATen Linear's view-valued VJP. Detach
        // only the wrapper: older LibTorch zero_grad calls detach_ on .grad.
        else parameters[j].mutable_grad()=gradients[j].is_view()?gradients[j].detach():gradients[j];
      }
    }
    {mapping_cuda::FusionRange trace("Mapping.PhotoBudgetBackward");selector_loss.backward();}
  }
  if (log_budget) {
    torch::NoGradGuard no_grad;
    const double target_active = static_cast<double>(target_ratio) * num_gaussians;
    std::cout << "[LOD Compact] mapping_iter=" << gaussians_->protection_.mapping_iteration
              << " ratio=" << target_ratio << " active_gaussians=" << compact_count
              << " target_active=" << target_active
              << " expected_active=" << mixed_probability.detach().sum().item<float>()
              << " actual_active=" << compact_count
              << " selector_candidates=" << selector_candidates
              << " anchor_frustum=" << anchor_frustum
              << " visible_anchors=" << visible_anchor_count
              << " budget_shortfall=" << std::max(0.0,target_active-compact_count)
              << " actual_vs_target=" << compact_count / std::max(1.0, target_active)
              << " tf=" << transform_count << " visible=" << visible_count << '\n';
    if (selector_network_->anchorConditioned()) {
      // Diagnostic only: export remains exactly the cached Bernoulli forward.
      const auto k = std::min<int64_t>(num_gaussians, static_cast<int64_t>(std::floor(target_active)));
      const auto order = std::get<1>(mixed_probability.detach().topk(k));
      const auto overlap = k ? (selector_mask.detach().index_select(0, order) > 0).sum().item<int64_t>() : 0;
      std::cout << "[Anchor Selection] policy=cached_bernoulli export_policy=cached_bernoulli"
                << " anchors=" << selector_network_->anchors()->size()
                << " diagnostic_probability_topk=" << k << " overlap=" << overlap
                << " jaccard=" << overlap / std::max(1.0, double(k + compact_count - overlap)) << '\n';
    }
    gaussians_->protection_.report(std::cout, selector_protection_enabled_);
  }
  result.trained = num_gaussians > 0;
  return result;
}

bool GaussianMapper::isStopped() {
  std::unique_lock<std::mutex> lock_status(this->mutex_status_);
  return this->stopped_;
}

void GaussianMapper::signalStop(const bool going_to_stop) {
  std::unique_lock<std::mutex> lock_status(this->mutex_status_);
  this->stopped_ = going_to_stop;
}

bool GaussianMapper::hasMetInitialMappingConditions() {
  if (!pSLAM_->isShutDown() &&
      pSLAM_->GetNumKeyframes() >= min_num_initial_map_kfs_ &&
      pSLAM_->getAtlas()->hasMappingOperation())
    return true;

  bool conditions_met = false;
  return conditions_met;
}

bool GaussianMapper::hasMetIncrementalMappingConditions() {
  if (!pSLAM_->isShutDown() && pSLAM_->getAtlas()->hasMappingOperation())
    return true;

  bool conditions_met = false;
  return conditions_met;
}

void GaussianMapper::combineMappingOperations() {
  // Get Mapping Operations
  while (pSLAM_->getAtlas()->hasMappingOperation()) {
    ORB_SLAM3::MappingOperation opr =
        pSLAM_->getAtlas()->getAndPopMappingOperation();

    switch (opr.meOperationType) {
      case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA: {
        // std::cout << "[Gaussian Mapper]Local BA Detected."
        //           << std::endl;

        // Get new keyframes
        auto& associated_kfs = opr.associatedKeyFrames();

        // Add keyframes to the scene
        for (auto& kf : associated_kfs) {
          // Keyframe Id
          auto kfid = std::get<0>(kf);
          std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
          // If the keyframe is already in the scene, only update the pose.
          // Otherwise create a new one
          if (pkf) {
            auto& pose = std::get<2>(kf);
            pkf->setPose(pose.unit_quaternion().cast<double>(),
                         pose.translation().cast<double>());
            pkf->computeTransformTensors();

            // Give local BA keyframes times of use
            increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
          } else {
            handleNewKeyframe(kf);
          }
        }

        // Get new points
        auto& associated_points = opr.associatedMapPoints();
        auto& points = std::get<0>(associated_points);
        auto& colors = std::get<1>(associated_points);

        // Add new points to the model
        if (initial_mapped_ && points.size() >= (selector_explicit_anchor_blocks_ ? 3 : 30)) {
          torch::NoGradGuard no_grad;
          std::unique_lock<std::mutex> lock_render(mutex_render_);
          gaussians_->increasePcd(points, colors, getIteration());
        }
      } break;

      case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA: {
        std::cout << "[Gaussian Mapper]Loop Closure Detected." << std::endl;
        auto change = captureGeometryChange(ChangeFlag::LOCAL_GEOMETRY);

        // Get the loop keyframe scale modification factor
        float loop_kf_scale = opr.mfScale;

        // Get new keyframes (scaled transformation applied in ORB-SLAM3)
        auto& associated_kfs = opr.associatedKeyFrames();
        // Mark the transformed points to avoid transforming more than once
        torch::Tensor point_not_transformed_flags = torch::full(
            {gaussians_->xyz_.size(0)}, true,
            torch::TensorOptions().device(device_type_).dtype(torch::kBool));
        if (record_loop_ply_)
          savePly(result_dir_ / (std::to_string(getIteration()) +
                                 "_0_before_loop_correction"));
        int num_transformed = 0;
        // Add keyframes to the scene
        for (auto& kf : associated_kfs) {
          // Keyframe Id
          auto kfid = std::get<0>(kf);
          std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
          // In case new points are added in handleNewKeyframe()
          int64_t num_new_points =
              gaussians_->xyz_.size(0) - point_not_transformed_flags.size(0);
          if (num_new_points > 0)
            point_not_transformed_flags =
                torch::cat({point_not_transformed_flags,
                            torch::full({num_new_points}, true,
                                        point_not_transformed_flags.options())},
                           /*dim=*/0);
          // If kf is already in the scene, evaluate the change in pose,
          // if too large we perform loop correction on its visible model
          // points. If not in the scene, create a new one.
          if (pkf) {
            auto& pose = std::get<2>(kf);
            Sophus::SE3f original_pose =
                pkf->getPosef();  // original_pose = old, inv_pose = new
            Sophus::SE3f inv_pose = pose.inverse();
            Sophus::SE3f diff_pose = inv_pose * original_pose;
            bool large_rot = !diff_pose.rotationMatrix().isApprox(
                Eigen::Matrix3f::Identity(), large_rot_th_);
            bool large_trans = !diff_pose.translation().isMuchSmallerThan(
                1.0, large_trans_th_);
            if (large_rot || large_trans) {
              std::cout << "[Gaussian Mapper]Large loop correction detected, "
                           "transforming visible points of kf "
                        << kfid << std::endl;
              diff_pose.translation() -=
                  inv_pose
                      .translation();  // t = (R_new * t_old + t_new) - t_new
              diff_pose.translation() *=
                  loop_kf_scale;  // t = s * (R_new * t_old)
              diff_pose.translation() +=
                  inv_pose.translation();  // t = (s * R_new * t_old) + t_new
              {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                auto correction =
                    gaussians_->scaledTransformVisiblePointsOfKeyframe(
                    point_not_transformed_flags, diff_pose,
                    pkf->world_view_transform_, pkf->full_proj_transform_,
                    pkf->creation_iter_, stableNumIterExistence(),
                    num_transformed,
                    loop_kf_scale);  // selected xyz *= s
                change.has_geometry_mutation = change.has_geometry_mutation ||
                    (correction.changed_attributes != 0 &&
                     correction.indices.defined() && correction.indices.numel() > 0);
                if (change.snapshot_count > 0)
                  change_coordinator_->addCorrection(change, correction, kfid);
              }
              // Give loop keyframes times of use
              increaseKeyframeTimesOfUse(pkf,
                                         loop_closure_increased_times_of_use_);
            }
            // }
            pkf->setPose(pose.unit_quaternion().cast<double>(),
                         pose.translation().cast<double>());
            pkf->computeTransformTensors();
          } else {
            handleNewKeyframe(kf);
          }
        }
        if (record_loop_ply_)
          savePly(result_dir_ / (std::to_string(getIteration()) +
                                 "_1_after_loop_correction"));
        // Get new points (scaled transformation applied in ORB-SLAM3, so this
        // step is performed at last to avoid scaling twice)
        auto& associated_points = opr.associatedMapPoints();
        auto& points = std::get<0>(associated_points);
        auto& colors = std::get<1>(associated_points);

        // Add new points to the model
        if (initial_mapped_ && points.size() >= (selector_explicit_anchor_blocks_ ? 3 : 30)) {
          torch::NoGradGuard no_grad;
          std::unique_lock<std::mutex> lock_render(mutex_render_);
          gaussians_->increasePcd(points, colors, getIteration());
        }

        finishGeometryChange(std::move(change));

        // Mark this iteration
        loop_closure_iteration_ = true;
      } break;

      case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement: {
        std::cout << "[Gaussian Mapper]Scale refinement Detected. Transforming "
                     "all kfs and points..."
                  << std::endl;
        auto change = initial_mapped_
                          ? captureGeometryChange(ChangeFlag::GLOBAL_GEOMETRY)
                          : ChangeCapture{};

        float s = opr.mfScale;
        Sophus::SE3f& T = opr.mT;
        if (initial_mapped_) {
          // Apply the scaled transformation on gaussian model points
          {
            std::unique_lock<std::mutex> lock_render(mutex_render_);
            auto correction = gaussians_->applyScaledTransformation(s, T);
            change.has_geometry_mutation = correction.changed_attributes != 0 &&
                correction.population_count > 0;
            if (change.snapshot_count > 0)
              change_coordinator_->addCorrection(change, correction);
          }
          // Apply the scaled transformation to the scene
          scene_->applyScaledTransformation(s, T);
        } else {  // TODO: the workflow should not come here, delete this branch
          // Apply the scaled transformation to the cached points
          for (auto& pt : scene_->cached_point_cloud_) {
            // pt <- (s * Ryw * pt + tyw)
            auto& pt_xyz = pt.second.xyz_;
            pt_xyz *= s;
            pt_xyz = T.cast<double>() * pt_xyz;
          }

          // Apply the scaled transformation on gaussian keyframes
          for (auto& kfit : scene_->keyframes()) {
            std::shared_ptr<GaussianKeyframe> pkf = kfit.second;
            Sophus::SE3f Twc = pkf->getPosef().inverse();
            Twc.translation() *= s;
            Sophus::SE3f Tyc = T * Twc;
            Sophus::SE3f Tcy = Tyc.inverse();
            pkf->setPose(Tcy.unit_quaternion().cast<double>(),
                         Tcy.translation().cast<double>());
            pkf->computeTransformTensors();
          }
        }
        finishGeometryChange(std::move(change));
      } break;

      default: {
        throw std::runtime_error("MappingOperation type not supported!");
      } break;
    }
  }
}

void GaussianMapper::handleNewKeyframe(std::tuple<unsigned long /*Id*/,
                                                  unsigned long /*CameraId*/,
                                                  Sophus::SE3f /*pose*/,
                                                  cv::Mat /*image*/,
                                                  bool /*isLoopClosure*/,
                                                  cv::Mat /*auxiliaryImage*/,
                                                  std::vector<float>,
                                                  std::vector<float>,
                                                  std::string>& kf) {
  std::shared_ptr<GaussianKeyframe> pkf =
      std::make_shared<GaussianKeyframe>(std::get<0>(kf), getIteration());
  pkf->zfar_ = z_far_;
  pkf->znear_ = z_near_;
  // Pose
  auto& pose = std::get<2>(kf);
  pkf->setPose(pose.unit_quaternion().cast<double>(),
               pose.translation().cast<double>());
  cv::Mat imgRGB_undistorted, imgAux_undistorted;
  try {
    // Camera
    Camera& camera = scene_->cameras_.at(std::get<1>(kf));
    pkf->setCameraParams(camera);

    imgRGB_undistorted = std::get<3>(kf);
    imgAux_undistorted = std::get<5>(kf);

    pkf->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(
        imgRGB_undistorted, device_type_);
    pkf->img_filename_ = std::get<8>(kf);
    pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
    pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
    pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
  } catch (std::out_of_range) {
    throw std::runtime_error(
        "[GaussianMapper::combineMappingOperations]KeyFrame Camera not found!");
  }
  // Add the new keyframe to the scene
  pkf->computeTransformTensors();
  scene_->addKeyframe(pkf, &kfid_shuffled_);
  registerRecentKeyframe(pkf);

  // Give new keyframes times of use and add it to the training sliding window
  increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

  // Get dense point cloud from the new keyframe to accelerate training
  pkf->img_undist_ = imgRGB_undistorted;
  pkf->img_auxiliary_undist_ = imgAux_undistorted;
  pkf->kps_pixel_ = std::move(std::get<6>(kf));
  pkf->kps_point_local_ = std::move(std::get<7>(kf));
  if (isdoingInactiveGeoDensify()) increasePcdByKeyframeInactiveGeoDensify(pkf);

  // Prepare multi resolution images for training
  if (device_type_ == torch::kCUDA) {
    cv::cuda::GpuMat img_gpu;
    img_gpu.upload(pkf->img_undist_);
    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
      cv::cuda::GpuMat img_resized;
      cv::cuda::resize(
          img_gpu, img_resized,
          cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
      pkf->gaus_pyramid_original_image_[l] =
          tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
    }
  } else {
    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
      cv::Mat img_resized;
      cv::resize(
          pkf->img_undist_, img_resized,
          cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
      pkf->gaus_pyramid_original_image_[l] =
          tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
    }
  }
}

void GaussianMapper::generateKfidRandomShuffle() {
  if (scene_->keyframes().empty()) return;

  std::size_t nkfs = scene_->keyframes().size();
  kfid_shuffle_.resize(nkfs);
  std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
  std::mt19937 g(rd_());
  std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

  kfid_shuffled_ = true;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomSlidingWindowKeyframe() {
  // auto t1 = std::chrono::steady_clock::now();
  if (scene_->keyframes().empty()) return nullptr;

  if (!kfid_shuffled_) generateKfidRandomShuffle();

  std::shared_ptr<GaussianKeyframe> viewpoint_cam = nullptr;
  int random_cam_idx;

  if (kfid_shuffled_) {
    int start_shuffle_idx = kfid_shuffle_idx_;
    do {
      // Next shuffled idx
      ++kfid_shuffle_idx_;
      if (kfid_shuffle_idx_ >= kfid_shuffle_.size()) kfid_shuffle_idx_ = 0;
      // Add 1 time of use to all kfs if they are all unavalible
      if (kfid_shuffle_idx_ == start_shuffle_idx) {
        for (auto& kfit : scene_->keyframes()) {
          increaseKeyframeTimesOfUse(kfit.second, 1);
        }
        if (opt_params_.auto_distribute_ &&
            (!ultrafast_rotation_.enabled || !kfs_loss_.empty())) {
          std::vector<std::pair<std::size_t, float>> vec(kfs_loss_.begin(),
                                                         kfs_loss_.end());
          // std::vector<std::pair<std::size_t, int>>
          // vec(kfs_used_times_.begin(), kfs_used_times_.end());
          int k = std::max(
              1, static_cast<int>(vec.size() / opt_params_.auto_distribute_));
          std::nth_element(vec.begin(), vec.begin() + k, vec.end(),
                           [](const std::pair<std::size_t, float>& a,
                              const std::pair<std::size_t, float>& b) {
                             return a.second > b.second;
                           });
          for (int i = 0; i < k; ++i) {
            increaseKeyframeTimesOfUse(scene_->keyframes()[vec[i].first], 1);
          }
        }
      }
      // Get viewpoint kf
      random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
      auto random_cam_it = scene_->keyframes().begin();
      for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
        ++random_cam_it;
      viewpoint_cam = (*random_cam_it).second;
    } while (viewpoint_cam->remaining_times_of_use_ <= 0);
  }

  // Count used times
  auto viewpoint_fid = viewpoint_cam->fid_;
  if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
    kfs_used_times_[viewpoint_fid] = 1;
  else
    ++kfs_used_times_[viewpoint_fid];

  // Handle times of use
  --(viewpoint_cam->remaining_times_of_use_);

  // auto t2 = std::chrono::steady_clock::now();
  // auto t21 =
  // std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
  // std::cout<<t21 <<" ns"<<std::endl;
  return viewpoint_cam;
}

std::shared_ptr<GaussianKeyframe> GaussianMapper::useOneRandomKeyframe() {
  if (scene_->keyframes().empty()) return nullptr;

  // Get randomly
  int nkfs = static_cast<int>(scene_->keyframes().size());
  int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / nkfs);
  auto random_cam_it = scene_->keyframes().begin();
  for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx) ++random_cam_it;
  std::shared_ptr<GaussianKeyframe> viewpoint_cam = (*random_cam_it).second;

  // Count used times
  auto viewpoint_fid = viewpoint_cam->fid_;
  if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
    kfs_used_times_[viewpoint_fid] = 1;
  else
    ++kfs_used_times_[viewpoint_fid];

  return viewpoint_cam;
}

void GaussianMapper::increaseKeyframeTimesOfUse(
    std::shared_ptr<GaussianKeyframe> pkf,
    int times) {
  pkf->remaining_times_of_use_ += times;
}

void GaussianMapper::cullKeyframes() {
  std::unordered_set<unsigned long> kfids =
      pSLAM_->getAtlas()->GetCurrentKeyFrameIds();
  std::vector<unsigned long> kfids_to_erase;
  std::size_t nkfs = scene_->keyframes().size();
  kfids_to_erase.reserve(nkfs);
  for (auto& kfit : scene_->keyframes()) {
    unsigned long kfid = kfit.first;
    if (kfids.find(kfid) == kfids.end()) {
      kfids_to_erase.emplace_back(kfid);
    }
  }

  for (auto& kfid : kfids_to_erase) {
    scene_->keyframes().erase(kfid);
  }
}

void GaussianMapper::increasePcdByKeyframeInactiveGeoDensify(
    std::shared_ptr<GaussianKeyframe> pkf) {
  // auto start_timing = std::chrono::steady_clock::now();
  torch::NoGradGuard no_grad;

  Sophus::SE3f Twc = pkf->getPosef().inverse();

  switch (this->sensor_type_) {
    case MONOCULAR: {
      // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
      // std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
      assert(pkf->kps_pixel_.size() % 2 == 0);
      int N = pkf->kps_pixel_.size() / 2;
      torch::Tensor kps_pixel_tensor =
          torch::from_blob(pkf->kps_pixel_.data(), {N, 2},
                           torch::TensorOptions().dtype(torch::kFloat32))
              .to(device_type_);
      torch::Tensor kps_point_local_tensor =
          torch::from_blob(pkf->kps_point_local_.data(), {N, 3},
                           torch::TensorOptions().dtype(torch::kFloat32))
              .to(device_type_);
      torch::Tensor kps_has3D_tensor = torch::where(
          kps_point_local_tensor.index({torch::indexing::Slice(), 2}) > 0.0f,
          true, false);

      cv::cuda::GpuMat rgb_gpu;
      rgb_gpu.upload(pkf->img_undist_);
      torch::Tensor colors =
          tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_gpu);
      colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

      auto result =
          monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
              kps_pixel_tensor, kps_has3D_tensor, kps_point_local_tensor,
              colors, monocular_inactive_geo_densify_max_pixel_dist_,
              pkf->intr_, pkf->image_width_, pkf->image_height_);
      torch::Tensor& points3D_valid = std::get<0>(result);
      torch::Tensor& colors_valid = std::get<1>(result);
      // Transform points to the world coordinate
      torch::Tensor Twc_tensor =
          tensor_utils::EigenMatrix2TorchTensor(Twc.matrix(), device_type_)
              .transpose(0, 1);
      transformPoints(points3D_valid, Twc_tensor);
      // Add new points to the cache
      if (depth_cached_ == 0) {
        depth_cache_points_ = points3D_valid;
        depth_cache_colors_ = colors_valid;
      } else {
        depth_cache_points_ =
            torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
        depth_cache_colors_ =
            torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
      }
      // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
      // std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    } break;
    case STEREO: {
      // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
      // std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
      cv::cuda::GpuMat rgb_left_gpu, rgb_right_gpu;
      cv::cuda::GpuMat gray_left_gpu, gray_right_gpu;

      rgb_left_gpu.upload(pkf->img_undist_);
      rgb_right_gpu.upload(pkf->img_auxiliary_undist_);

      // From CV_32FC3 to CV_32FC1
      cv::cuda::cvtColor(rgb_left_gpu, gray_left_gpu, cv::COLOR_RGB2GRAY);
      cv::cuda::cvtColor(rgb_right_gpu, gray_right_gpu, cv::COLOR_RGB2GRAY);

      // From CV_32FC1 to CV_8UC1
      gray_left_gpu.convertTo(gray_left_gpu, CV_8UC1, 255.0);
      gray_right_gpu.convertTo(gray_right_gpu, CV_8UC1, 255.0);

      // Compute disparity
      cv::cuda::GpuMat cv_disp;
      stereo_cv_sgm_->compute(gray_left_gpu, gray_right_gpu, cv_disp);
      cv_disp.convertTo(cv_disp, CV_32F, 1.0 / 16.0);

      // Reproject to get 3D points
      cv::cuda::GpuMat cv_points3D;
      cv::cuda::reprojectImageTo3D(cv_disp, cv_points3D, stereo_Q_, 3);

      // From cv::cuda::GpuMat to torch::Tensor
      torch::Tensor disp = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_disp);
      disp = disp.flatten(0, 1).contiguous();
      torch::Tensor points3D =
          tensor_utils::cvGpuMat2TorchTensor_Float32(cv_points3D);
      points3D = points3D.permute({1, 2, 0}).flatten(0, 1).contiguous();
      torch::Tensor colors =
          tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_left_gpu);
      colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

      // Clear undisired and unreliable stereo points
      torch::Tensor point_valid_flags = torch::full(
          {disp.size(0)}, false,
          torch::TensorOptions().dtype(torch::kBool).device(device_type_));
      int nkps_twice = pkf->kps_pixel_.size();
      int width = pkf->image_width_;
      for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
        int idx = static_cast<int>(/*u*/ pkf->kps_pixel_[kpidx]) +
                  static_cast<int>(/*v*/ pkf->kps_pixel_[kpidx + 1]) * width;
        // int u = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]);
        // if (u < 0.3 * width || u > 0.7 * width)
        point_valid_flags[idx] = true;
      }
      point_valid_flags = torch::logical_and(
          point_valid_flags,
          torch::where(
              disp > static_cast<float>(stereo_cv_sgm_->getMinDisparity()),
              true, false));
      point_valid_flags = torch::logical_and(
          point_valid_flags,
          torch::where(
              disp < static_cast<float>(stereo_cv_sgm_->getNumDisparities()),
              true, false));

      torch::Tensor points3D_valid = points3D.index({point_valid_flags});
      torch::Tensor colors_valid = colors.index({point_valid_flags});

      // Transform points to the world coordinate
      torch::Tensor Twc_tensor =
          tensor_utils::EigenMatrix2TorchTensor(Twc.matrix(), device_type_)
              .transpose(0, 1);
      transformPoints(points3D_valid, Twc_tensor);

      // Add new points to the cache
      if (depth_cached_ == 0) {
        depth_cache_points_ = points3D_valid;
        depth_cache_colors_ = colors_valid;
      } else {
        depth_cache_points_ =
            torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
        depth_cache_colors_ =
            torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
      }
    } break;
    case RGBD: {
      cv::cuda::GpuMat img_rgb_gpu, img_depth_gpu;
      img_rgb_gpu.upload(pkf->img_undist_);
      img_depth_gpu.upload(pkf->img_auxiliary_undist_);

      // From cv::cuda::GpuMat to torch::Tensor
      torch::Tensor rgb =
          tensor_utils::cvGpuMat2TorchTensor_Float32(img_rgb_gpu);
      rgb = rgb.permute({1, 2, 0}).flatten(0, 1).contiguous();
      torch::Tensor depth =
          tensor_utils::cvGpuMat2TorchTensor_Float32(img_depth_gpu);
      depth = depth.flatten(0, 1).contiguous();

      // To clear undisired and unreliable depth
      torch::Tensor point_valid_flags = torch::full(
          {depth.size(0)}, false /*true*/,
          torch::TensorOptions().dtype(torch::kBool).device(device_type_));
      int nkps_twice = pkf->kps_pixel_.size();
      int width = pkf->image_width_;
      for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
        int idx = static_cast<int>(/*u*/ pkf->kps_pixel_[kpidx]) +
                  static_cast<int>(/*v*/ pkf->kps_pixel_[kpidx + 1]) * width;
        point_valid_flags[idx] = true;
      }
      point_valid_flags = torch::logical_and(
          point_valid_flags,
          torch::where(depth > RGBD_min_depth_, true, false));
      point_valid_flags = torch::logical_and(
          point_valid_flags,
          torch::where(depth < RGBD_max_depth_, true, false));

      torch::Tensor colors_valid = rgb.index({point_valid_flags});

      // Reproject to get 3D points
      torch::Tensor points3D_valid;
      Camera& camera = scene_->cameras_.at(pkf->camera_id_);
      switch (camera.model_id_) {
        case Camera::PINHOLE: {
          points3D_valid = reprojectDepthPinhole(depth, point_valid_flags,
                                                 pkf->intr_, pkf->image_width_);
        } break;
        case Camera::FISHEYE: {
          // TODO: support fisheye camera?
          throw std::runtime_error(
              "[Gaussian Mapper]Fisheye cameras are not supported currently!");
        } break;
        default: {
          throw std::runtime_error("[Gaussian Mapper]Invalid camera model!");
        } break;
      }
      points3D_valid = points3D_valid.index({point_valid_flags});

      // Transform points to the world coordinate
      torch::Tensor Twc_tensor =
          tensor_utils::EigenMatrix2TorchTensor(Twc.matrix(), device_type_)
              .transpose(0, 1);
      transformPoints(points3D_valid, Twc_tensor);

      // Add new points to the cache
      if (depth_cached_ == 0) {
        depth_cache_points_ = points3D_valid;
        depth_cache_colors_ = colors_valid;
      } else {
        depth_cache_points_ =
            torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
        depth_cache_colors_ =
            torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
      }
    } break;
    default: {
      throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    } break;
  }

  pkf->done_inactive_geo_densify_ = true;
  ++depth_cached_;

  if (depth_cached_ >= max_depth_cached_) {
    depth_cached_ = 0;
    // Add new points to the model
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    gaussians_->increasePcd(depth_cache_points_, depth_cache_colors_,
                            getIteration());
  }

  // auto end_timing = std::chrono::steady_clock::now();
  // auto completion_time =
  // std::chrono::duration_cast<std::chrono::milliseconds>(
  //                 end_timing - start_timing).count();
  // std::cout << "[Gaussian Mapper]increasePcdByKeyframeInactiveGeoDensify()
  // takes "
  //             << completion_time
  //             << " ms"
  //             << std::endl;
}

void GaussianMapper::recordKeyframeRendered(
    torch::Tensor& rendered,
    torch::Tensor& ground_truth,
    unsigned long kfid,
    std::filesystem::path result_img_dir,
    std::filesystem::path result_gt_dir,
    std::filesystem::path result_loss_dir,
    std::string name_suffix) {
  if (record_rendered_image_) {
    auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered);
    cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
    image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
    cv::imwrite(result_img_dir / (std::to_string(getIteration()) + "_" +
                                  std::to_string(kfid) + name_suffix + ".jpg"),
                image_cv);
  }

  if (record_ground_truth_image_) {
    auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(ground_truth);
    cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
    gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
    cv::imwrite(
        result_gt_dir / (std::to_string(getIteration()) + "_" +
                         std::to_string(kfid) + name_suffix + "_gt.jpg"),
        gt_image_cv);
  }

  if (record_loss_image_) {
    torch::Tensor loss_tensor = torch::abs(rendered - ground_truth);
    auto loss_image_cv = tensor_utils::torchTensor2CvMat_Float32(loss_tensor);
    cv::cvtColor(loss_image_cv, loss_image_cv, CV_RGB2BGR);
    loss_image_cv.convertTo(loss_image_cv, CV_8UC3, 255.0f);
    cv::imwrite(
        result_loss_dir / (std::to_string(getIteration()) + "_" +
                           std::to_string(kfid) + name_suffix + "_loss.jpg"),
        loss_image_cv);
  }
}

cv::Mat GaussianMapper::renderFromPose(const Sophus::SE3f& Tcw,
                                       const int width,
                                       const int height,
                                       const bool main_vision) {
  if (!initial_mapped_ || getIteration() <= 0)
    return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
  std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
  pkf->zfar_ = z_far_;
  pkf->znear_ = z_near_;
  // Pose
  pkf->setPose(Tcw.unit_quaternion().cast<double>(),
               Tcw.translation().cast<double>());
  try {
    // Camera
    Camera& camera = scene_->cameras_.at(viewer_camera_id_);
    pkf->setCameraParams(camera);
    // Transformations
    pkf->computeTransformTensors();
  } catch (std::out_of_range) {
    throw std::runtime_error(
        "[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
  }

  RenderPackage render_pkg;
  {
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    // Render
    render_pkg =
        GaussianRenderer::render(pkf, height, width, gaussians_, pipe_params_,
                                 background_, override_color_);
  }

  // Result
  return tensor_utils::torchTensor2CvMat_Float32(render_pkg.image);
}

void GaussianMapper::renderAndRecordKeyframe(
    std::shared_ptr<GaussianKeyframe> pkf,
    float& dssim,
    float& psnr,
    float& psnr_gs,
    double& render_time,
    std::filesystem::path result_img_dir,
    std::filesystem::path result_gt_dir,
    std::filesystem::path result_loss_dir,
    std::string name_suffix) {
  auto start_timing = std::chrono::steady_clock::now();
  auto render_pkg = GaussianRenderer::render(
      pkf, pkf->image_height_, pkf->image_width_, gaussians_, pipe_params_,
      background_, override_color_);
  auto rendered_image = render_pkg.image;
  torch::cuda::synchronize();
  auto end_timing = std::chrono::steady_clock::now();
  auto render_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            end_timing - start_timing)
                            .count();
  render_time = 1e-6 * render_time_ns;
  auto gt_image = pkf->original_image_;

  dssim = loss_utils::fast_ssim(rendered_image, gt_image).item().toFloat();
  psnr = loss_utils::psnr(rendered_image, gt_image).item().toFloat();
  psnr_gs = loss_utils::psnr_gaussian_splatting(rendered_image, gt_image)
                .item()
                .toFloat();

  recordKeyframeRendered(rendered_image, gt_image, pkf->fid_, result_img_dir,
                         result_gt_dir, result_loss_dir, name_suffix);
}

void GaussianMapper::renderAndRecordAllKeyframes(std::string name_suffix) {
  std::filesystem::path result_dir =
      result_dir_ / (std::to_string(getIteration()) + name_suffix);
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

  std::filesystem::path image_dir = result_dir / "image";
  if (record_rendered_image_)
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_dir);

  std::filesystem::path image_gt_dir = result_dir / "image_gt";
  if (record_ground_truth_image_)
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_gt_dir);

  std::filesystem::path image_loss_dir = result_dir / "image_loss";
  if (record_loss_image_) {
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_loss_dir);
  }

  std::filesystem::path render_time_path = result_dir / "render_time.txt";
  std::ofstream out_time(render_time_path);
  out_time << "##[Gaussian Mapper]Render time statistics: keyframe id, "
              "time(milliseconds)"
           << std::endl;

  std::filesystem::path dssim_path = result_dir / "dssim.txt";
  std::ofstream out_dssim(dssim_path);
  out_dssim << "##[Gaussian Mapper]keyframe id, dssim" << std::endl;

  std::filesystem::path psnr_path = result_dir / "psnr.txt";
  std::ofstream out_psnr(psnr_path);
  out_psnr << "##[Gaussian Mapper]keyframe id, psnr" << std::endl;

  std::filesystem::path psnr_gs_path =
      result_dir / "psnr_gaussian_splatting.txt";
  std::ofstream out_psnr_gs(psnr_gs_path);
  out_psnr_gs << "##[Gaussian Mapper]keyframe id, psnr_gaussian_splatting"
              << std::endl;

  std::size_t nkfs = scene_->keyframes().size();
  auto kfit = scene_->keyframes().begin();
  float dssim, psnr, psnr_gs;
  double render_time;
  for (std::size_t i = 0; i < nkfs; ++i) {
    renderAndRecordKeyframe((*kfit).second, dssim, psnr, psnr_gs, render_time,
                            image_dir, image_gt_dir, image_loss_dir);
    out_time << (*kfit).first << " " << std::fixed << std::setprecision(8)
             << render_time << std::endl;

    out_dssim << (*kfit).first << " " << std::fixed << std::setprecision(10)
              << dssim << std::endl;
    out_psnr << (*kfit).first << " " << std::fixed << std::setprecision(10)
             << psnr << std::endl;
    out_psnr_gs << (*kfit).first << " " << std::fixed << std::setprecision(10)
                << psnr_gs << std::endl;

    ++kfit;
  }
}

void GaussianMapper::savePly(std::filesystem::path result_dir) {
  // Mapping exports are graph-free (and may already hold mutex_render_).
  if (selector_anchor_adaptive_ && selector_network_)
    selector_network_->anchors()->refine(gaussians_->getXYZ(),
        {gaussians_->optimizer_.get(), selector_optimizer_.get()});
  reportMode2Mapping("save");
  reportUltrafastRotation("save");
  continuous_tf_statistics_.report(std::cout, gaussians_->protection_.first_valid_maturity_cycle_iteration);
  gaussians_->protection_.report(std::cout, selector_protection_enabled_);
  if (global_gi_) {
    global_gi_->recordGuidance(global_gi_->observedMask(), selector_enabled_ && selector_gi_loss_weight_ != 0.0f &&
        hasNormalOptimizationBudget());
    global_gi_->report();
  }
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
  saveViewBankOrbMap(result_dir);
  if(real_lod_.enabled)saveRealLodState(result_dir);
  gaussians_->protection_.dynamics.save(result_dir, gaussians_->protection_.maturity,
      protection_config_.maturity_rise_rate, protection_config_.maturity_fall_rate);
  saveSelectorConfig(result_dir);
  keyframesToJson(result_dir);
  saveModelParams(result_dir);
  {
    Json::Value state;
    state["ultra_light_mode"] = ultra_light_mode_;
    state["serial_full_lod"] = ultrafast_rotation_.serial_full_lod;
    state["serial_lod_detach_master"] = ultrafast_rotation_.serial_lod_detach_master;
    state["normal_lod_branch_count"] = Json::UInt64(normal_lod_branch_count_);
    state["selector_step"] = Json::UInt64(selector_step_);
    state["tf_updates"] = Json::UInt64(normal_update_clock_.tf_updates);
    for (int phase = 0; phase < 2; ++phase) {
      const auto& c = phase ? mode2_tail_ : mode2_online_;
      auto& s = state[phase ? "tail" : "online"];
      s["mappings"] = Json::UInt64(c.mappings);
      s["full_renders"] = Json::UInt64(c.full_renders);
      s["full_backwards"] = Json::UInt64(c.full_backwards);
      s["gaussian_step_calls"] = Json::UInt64(c.gaussian_step_calls);
      s["gaussian_updates"] = Json::UInt64(c.gaussian_updates);
      s["stats"] = Json::UInt64(c.stats);
      s["topology"] = Json::UInt64(c.topology);
      s["opacity_resets"] = Json::UInt64(c.opacity_resets);
      s["lod_attempts"] = Json::UInt64(c.lod_attempts);
      s["lod_renders"] = Json::UInt64(c.lod_renders);
      s["selector_commits"] = Json::UInt64(c.selector_commits);
      s["tf_commits"] = Json::UInt64(c.tf_commits);
      s["lod_nonfinite"] = Json::UInt64(c.lod_nonfinite);
      for (auto count : c.ratios) s["ratio_counts"].append(Json::UInt64(count));
    }
    std::ofstream output(result_dir / (ultra_light_mode_ == 2 ? "ultralight_mode2_state.json" : "mapping_stage_state.json"));
    output << state;
  }
  if (ultrafast_rotation_.enabled) {
    const auto& r = ultrafast_rotation_;
    Json::Value state;
    for (std::size_t i = 0; i < r.counts.size(); ++i) {
      state["ultrafast_" + r.countKey(i) + "_steps"] = Json::UInt64(r.counts[i]);
      if (r.counts[i] && mapping_iteration_logging_) {
        state[r.countKey(i) + "_mean_ms"] = r.elapsed_sum[i] / r.counts[i];
        state[r.countKey(i) + "_render_mean_ms"] = r.render_sum[i] / r.counts[i];
      }
    }
    state["ultrafast_total_normal_steps"] = Json::UInt64(r.step);
    state["balanced"] = r.balanced();
    state["iteration_logging"] = mapping_iteration_logging_;
    state["mean_mapping_training_ms"] = mapping_iteration_logging_ ? Json::Value(r.mean()) : Json::Value(Json::nullValue);
    state["p95_mapping_training_ms"] = mapping_iteration_logging_ ? Json::Value(r.p95()) : Json::Value(Json::nullValue);
    state["full_render_count"] = Json::UInt64(r.full_render_count);
    state["serial_full_lod"] = r.serial_full_lod;
    state["serial_lod_detach_master"] = r.serial_lod_detach_master;
    state["post_topology_full_steps"] = r.post_topology_full_steps;
    state["recovery_steps"] = Json::UInt64(r.recovery_steps);
    state["online_recovery_steps"] = Json::UInt64(r.online_recovery_steps);
    state["recovery_windows"] = Json::UInt64(r.recovery_windows);
    state["recovery_remaining"] = Json::UInt64(r.recovery_remaining);
    state["lod_render_count"] = Json::UInt64(r.lod_render_count);
    state["online_ended"] = r.online_ended;
    state["online_end_mapping_iteration"] = Json::UInt64(r.online_end_mapping_iteration);
    state["online_end_rotation_step"] = Json::UInt64(r.online_end_rotation_step);
    state["tail_training_steps"] = Json::UInt64(r.tailSteps());
    state["tail_ratio_rotation"] = r.rotate_tail;
    for(auto ratio:r.active_ratios)state["target_ratios"].append(ratio);
    state["tail_policy"] = !r.serial_full_lod ? (r.rotate_tail ? "ratio_rotation" : "full_then_joint_lod") :
        (r.serial_lod_detach_master ? "full_master_then_detached_lod" : "full_then_joint_lod");
    state["tail_full_render_count"] = Json::UInt64(r.tail_full_renders);
    state["tail_recovery_steps"] = Json::UInt64(r.tail_recovery_steps);
    for (auto n : r.tail_phase_counts) state["tail_phase_counts"].append(Json::UInt64(n));
    state["tail_lod_render_count"] = Json::UInt64(r.tail_lod_renders);
    state["tail_lod_attempts"] = Json::UInt64(r.tail_lod_attempts);
    for (auto n : r.tail_ratio_counts) state["tail_ratio_counts"].append(Json::UInt64(n));
    state["selector_online_lod_update_count"] = Json::UInt64(r.online_ended ? r.online_end_selector_updates : r.selector_updates);
    state["tf_online_lod_update_count"] = Json::UInt64(r.online_ended ? r.online_end_tf_updates : r.tf_updates);
    state["selector_tail_lod_update_count"] = Json::UInt64(r.online_ended ? r.selector_updates - r.online_end_selector_updates : 0);
    state["tf_tail_lod_update_count"] = Json::UInt64(r.online_ended ? r.tf_updates - r.online_end_tf_updates : 0);
    state["tf_first_initialized_mapping_iteration"] = Json::Int64(ultrafast_tf_first_initialized_);
    state["tf_first_applied_lod_iteration"] = Json::Int64(ultrafast_tf_first_applied_);
    state["timing"] = !mapping_iteration_logging_ ? "disabled" :
        (device_type_ == torch::kCUDA ? "synchronized_wall_and_cuda_event_render" : "cpu_wall");
    std::ofstream output(result_dir / "ultrafast_rotation_state.json");
    output << state;
  }

  std::filesystem::path ply_dir = result_dir / "point_cloud";
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

  ply_dir = ply_dir / ("iteration_" + std::to_string(getIteration()));
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

  gaussians_->savePly(ply_dir / "point_cloud.ply");
  gaussians_->saveSelectorMetadataPly(ply_dir / "selector_metadata.ply");
  if(global_gi_) {
    Json::Value state;state["mode"]=global_gi_config_.mode;state["loss_weight"]=selector_gi_loss_weight_;
    state["version"]=Json::UInt64(global_gi_->version());state["topology_version"]=Json::UInt64(global_gi_->topologyVersion());
    state["maturity_fixed_one"]=protection_config_.fixed_maturity_one;
    for(auto id:global_gi_->bank().members())state["bank_fids"].append(Json::UInt64(id));
    std::ofstream out(ply_dir/"sgi_state.json");out<<state;
  }
  if(protection_config_.reliability_v2) {
    auto& p=gaussians_->protection_;
    Json::Value gate;gate["enabled"]=protection_config_.reliability_gate;
    gate["appearance"]=protection_config_.reliability_gate_appearance;
    gate["low"]=protection_config_.reliability_gate_low;gate["high"]=protection_config_.reliability_gate_high;
    std::ofstream gate_out(ply_dir/"reliability_gate.json");gate_out<<gate;
    if(!reliability_trace_.empty()) {
      auto trace=torch::stack(reliability_trace_).cpu().contiguous();auto t=trace.accessor<float,2>();
      std::ofstream csv(ply_dir/"reliability_training.csv");
      csv<<"mapping,ratio,N,K,maturity_pool,maturity_selected,below04,above08,O_mean,V_mean,alpha_mean,p_mean\n";
      for(int64_t i=0;i<trace.size(0);++i){for(int64_t j=0;j<trace.size(1);++j)csv<<(j?",":"")<<t[i][j];csv<<'\n';}
    }
    auto state=torch::stack({p.maturity,p.optimization_support,p.view_support,p.drift_ema},1).cpu().contiguous();
    auto v=state.accessor<float,2>();std::ofstream out(ply_dir/"reliability_maturity.csv");
    out.precision(10);out<<"row,maturity,optimization_support,view_support,drift_ema\n";
    for(int64_t i=0;i<state.size(0);++i)out<<i<<','<<v[i][0]<<','<<v[i][1]<<','<<v[i][2]<<','<<v[i][3]<<'\n';
  }
  Json::Value protection_state;
  protection_state["schema"] = "continuous_cycle_geometric";
  protection_state["maturity_mode"] = protection_config_.fixed_maturity_one ? "fixed_one" : "continuous_cycle_geometric";
  protection_state["maturity_dynamics"] = "temporal_trust";
  protection_state["maturity_rise_rate"] = protection_config_.maturity_rise_rate;
  protection_state["maturity_fall_rate"] = protection_config_.maturity_fall_rate;
  protection_state["protection_enabled"] = selector_protection_enabled_;
  protection_state["selector_enabled"] = selector_enabled_;
  protection_state["mapping_iteration"] = Json::Int64(gaussians_->protection_.mapping_iteration);
  protection_state["metadata_memory_mb"] = gaussians_->protection_.metadataMB();
  for (const auto& entry : gaussians_->protection_.statistics(selector_protection_enabled_)) protection_state[entry.first] = entry.second;
  std::ofstream protection_output(ply_dir / "protection_state.json");
  protection_output << protection_state;

  if (selector_network_) {
    if (selector_slot_deactivation_) selector_network_->anchors()->refreshSlotActivity(gaussians_->getOpacityActivation());
    // Avoid torch::save's generic operator<< path; LibTorch serializes the
    // registered module parameters through Module::save(OutputArchive&).
    torch::serialize::OutputArchive selector_archive;
    selector_network_->save(selector_archive);
    selector_archive.save_to((result_dir / "selector.pt").string());
    if(anchor_si_config_.enabled) {
      anchor_si_.sync(selector_network_,topology_version_);
      anchor_si_.save(result_dir / "anchor_si_state.pt");
    }
    if (selector_network_->anchorConditioned()) {
      Json::Value anchor_state;
      anchor_state["schema"] = selector_anchor_ratio_features_ ? "explicit_anchor_selector_v7_ratio_features" : (selector_variable_anchor_members_ ? "explicit_anchor_selector_v6_variable_members" : (selector_explicit_anchor_blocks_ ? "explicit_anchor_selector_v5_ten_slots" : (selector_anchor_global_xyz_ ? "anchor_conditioned_selector_v4_global_xyz_cell_local" :
          (selector_anchor_gaussian_attributes_ ? "anchor_conditioned_selector_v3_attributes" : "anchor_conditioned_selector_v2"))));
      anchor_state["gaussian_attributes"] = selector_anchor_gaussian_attributes_;
      if(selector_anchor_feature_dim_==32)anchor_state["schema"]="explicit_anchor_selector_v8_feature32";
      if(selector_anchor_feature_dim_==0)anchor_state["schema"]="explicit_anchor_selector_v10_s00r00";
      if(selector_anchor_ratio_features_ && selector_anchor_ratio_shared_dim_==16)anchor_state["schema"]="explicit_anchor_selector_v9_shared16_ratio8";
      anchor_state["selector_compensation"] = selector_compensation_;
      anchor_state["anchor_frustum_culling"] = selector_anchor_frustum_culling_;
      anchor_state["compensation_forwards"] = Json::UInt64(compensation_forwards_);
      anchor_state["compensation_decoded_rows"] = Json::UInt64(compensation_decoded_rows_);
      anchor_state["cd_compensation_steps"] = Json::UInt64(cd_compensation_steps_);
      anchor_state["compensation_start_iter"] = selector_compensation_start_iter_;
      anchor_state["compensation_output_dim"] = selector_network_->compensationOutputWidth();
      anchor_state["compensation_hidden_layers"] = selector_network_->compensationHiddenLayers();
      anchor_state["compensation_method"] = selector_compensation_method_;
      if(selector_network_->privateCompensation()) {
        const auto& bank=selector_network_->privateCompensation();
        TORCH_CHECK(bank->size()==gaussians_->getXYZ().size(0),"Private compensation lost Gaussian rows");
        anchor_state["private_compensation_rows"]=Json::Int64(bank->size());
        anchor_state["private_compensation_tables"]=int(bank->tables().size());
        anchor_state["private_compensation_interpolation"]="linear raw ratio; clamp outside trained knots";
        for(const auto& table:bank->tables()) {
          Json::Value row;
          row["rows"]=Json::Int64(table.size(0));row["width"]=14;
          row["finite"]=torch::isfinite(table).all().item<bool>();
          row["rms"]=table.numel()?table.square().mean().sqrt().item<double>():0.;
          row["max_abs"]=table.numel()?table.abs().max().item<double>():0.;
          auto found=gaussians_->optimizer_->state().find(table.unsafeGetTensorImpl());
          row["adam_step"]=Json::Int64(found==gaussians_->optimizer_->state().end()?0:
              static_cast<torch::optim::AdamParamState&>(*found->second).step());
          anchor_state["private_compensation_state"].append(row);
        }
      }
      anchor_state["compensation_layout"] = selector_anchor_ratio_features_
          ? "shared 26->32; select 32->32->2; selected compensation 32->32->14 (XYZ3,logscale3,quaternion4,opacity_logit1,SH_DC3)"
          : "shared 27->32; select 32->32->2; selected compensation 32->32->14 (XYZ3,logscale3,quaternion4,opacity_logit1,SH_DC3)";
      anchor_state["input_dim"] = selector_anchor_ratio_features_ ? selector_anchor_ratio_shared_dim_+18+int(selector_anchor_ratio_scalar_input_!="none") : (selector_anchor_feature_dim_+4+(selector_anchor_gaussian_attributes_?7:0)+(selector_anchor_global_xyz_?3:0));
      if(selector_network_->selectorSpatialFeatures()) {
        anchor_state["input_dim"]=139;
        anchor_state["selector_base_input_dim"]=11;
        anchor_state["selector_shared_spatial_feature_dim"]=128;
        anchor_state["selector_parameters"]=5602;
        anchor_state["selector_feature_source"]="same raw anchors.spatial_features table as Compensation; no detach; no duplicate table";
        anchor_state["selector_execution"]="LibTorch MLP, FlexGS Gumbel ST sampler unchanged";
      }
      if(!selector_compensation_)anchor_state["compensation_layout"]="disabled; Selector only; no compensation network or features";
      anchor_state["ratio_scalar_input"] = selector_anchor_ratio_scalar_input_;
      if(selector_anchor_feature_dim_==32)anchor_state["compensation_layout"]="shared 43->32; select 32->32->2; selected compensation 32->32->14 (XYZ3,logscale3,quaternion4,opacity_logit1,SH_DC3)";
      if(selector_network_->spatialCompensationEnabled()) {
        anchor_state["compensation_fourier_bands"]=int(selector_network_->spatialCompensation()->fourierBands());
        anchor_state["compensation_active_fourier_dim"]=int(6*selector_network_->spatialCompensation()->fourierBands());
        anchor_state["compensation_layout"]="Selector11->32->32->2; selected live Spatial55->64->64 with continuous log-ratio FiLM and four independent heads; XYZ in Anchor units";
        anchor_state["compensation_parameters"]=33806;
        if(selector_network_->anchors()->spatialFeatures()) {
          const auto& table=selector_network_->anchors()->spatialFeaturesTable();
          anchor_state["compensation_layout"]="Selector11; selected Spatial(55 physical/Fourier + 16 Anchor latent)->64->64; continuous ratio FiLM; four heads; joint train";
          anchor_state["compensation_parameters"]=34830;
          if(selector_network_->spatialCompensation()->directRatioInput()) {
            anchor_state["compensation_layout"]="Selector11; selected 55 physical/Fourier + Anchor16 + normalized log-ratio1 ->64 ReLU ->64 ReLU; four heads; no FiLM";
            anchor_state["compensation_parameters"]=26382;
          }
          anchor_state["spatial_feature_rows"]=Json::Int64(table.size(0));anchor_state["spatial_feature_dim"]=int(table.size(1));
          anchor_state["spatial_feature_finite"]=torch::isfinite(table).all().item<bool>();
          anchor_state["spatial_feature_rms"]=table.numel()?table.square().mean().sqrt().item<double>():0.;
          auto found=gaussians_->optimizer_->state().find(table.unsafeGetTensorImpl());
          anchor_state["spatial_feature_adam_step"]=Json::Int64(found==gaussians_->optimizer_->state().end()?0:static_cast<torch::optim::AdamParamState&>(*found->second).step());
        }
        anchor_state["spatial_normalization"]="fixed at first compensation from full current map; live selected Gaussian attributes and Master gradients";
        anchor_state["spatial_normalization_ready"]=selector_network_->spatialCompensation()->ready();
        anchor_state["private_compensation_tables"]=0;
      }
      if(selector_network_->privateCompensation())anchor_state["compensation_layout"]="frozen C(h) + Gaussian-identity 14D residual; selected rows only";
      if(selector_network_->anchors()->compensationFeatures()) {
        anchor_state["compensation_layout"]=selector_anchor_ratio_features_
            ? "shared 26->32 (Anchor8 + ratio8; no scalar ratio); select 32->32->2; selected compensation (shared32 + Anchor/ratio latent14)->32->14"
            : "shared 27->32; select 32->32->2; selected compensation (shared32 + Anchor/ratio latent14)->32->14";
        if(selector_anchor_ratio_scalar_input_!="none")anchor_state["compensation_layout"]=
            "shared 27->32 (Anchor8 + ratio8 + "+selector_anchor_ratio_scalar_input_+" ratio scalar); select 32->32->2; selected compensation (shared32 + Anchor/ratio latent14)->32->14";
        if(selector_anchor_ratio_features_ && selector_anchor_ratio_shared_dim_==16)anchor_state["compensation_layout"]=
            "shared "+std::to_string(anchor_state["input_dim"].asInt())+"->32 (Anchor16 + ratio8 + "+selector_anchor_ratio_scalar_input_+" scalar); select 32->32->2; selected compensation (shared32 + latent14)->32->14";
        if(selector_anchor_feature_dim_==0)anchor_state["compensation_layout"]=
            "shared 11->32 (offset3,scale3,SH_DC3,opacity1,log_ratio1); select 32->32->2; selected compensation (shared32 + Anchor/ratio latent"+
            std::to_string(selector_network_->anchors()->compensationFeatureWidth())+")->32->32->14";
        anchor_state["compensation_feature_dim"]=int(selector_network_->anchors()->compensationFeatureWidth());
        anchor_state["compensation_feature_interpolation"]="linear raw ratio latent interpolation before MLP; endpoint clamp";
        for(const auto& table:selector_network_->anchors()->compensationFeatureTables()) {
          Json::Value row;row["rows"]=Json::Int64(table.size(0));row["width"]=int(table.size(1));
          row["finite"]=torch::isfinite(table).all().item<bool>();
          row["rms"]=table.numel()?table.square().mean().sqrt().item<double>():0.;
          auto found=gaussians_->optimizer_->state().find(table.unsafeGetTensorImpl());
          row["adam_step"]=Json::Int64(found==gaussians_->optimizer_->state().end()?0:
              static_cast<torch::optim::AdamParamState&>(*found->second).step());
          anchor_state["compensation_feature_state"].append(row);
        }
      }
      if(selector_network_->spatialCompensationEnabled()&&selector_network_->spatialCompensation()->ratioFeatureWidth()) {
        anchor_state["compensation_layout"]="Selector11; selected physical19 + Anchor/ratio32 ->64 ReLU ->64 ReLU; four heads; no Fourier, scalar ratio or FiLM; XYZ in Anchor units";
        anchor_state["compensation_parameters"]=25038;
        anchor_state["compensation_input_dim"]=51;
        anchor_state["compensation_ratio_scalar_input"]=false;
        if(selector_network_->spatialCompensation()->ratioCodeInput()) {
          anchor_state["compensation_layout"]="Selector11; selected physical19 + Anchor/ratio32 + log-ratio Fourier3 ->64 ReLU ->64 ReLU; four heads; no FiLM or Offset modulation";
          anchor_state["compensation_parameters"]=25230;
          anchor_state["compensation_input_dim"]=54;
          anchor_state["compensation_ratio_code_dim"]=3;
          const auto bounds=selector_network_->spatialCompensation()->ratio_bounds.to(torch::kCPU);
          anchor_state["compensation_ratio_encoding_min"]=bounds[0].item<double>();
          anchor_state["compensation_ratio_encoding_max"]=bounds[1].item<double>();
        }
      }
      if(selector_network_->spatialCompensationEnabled()&&selector_network_->spatialCompensation()->ratioFeatureWidth()==48) {
        anchor_state["compensation_layout"]="Selector11; selected offset3 + per-Anchor per-ratio Feature48 + continuous ratio3 ->64 ReLU ->64 ReLU; four heads; no Feature gate or physical19";
        anchor_state["compensation_parameters"]=25230;
        anchor_state["compensation_input_dim"]=54;
        anchor_state["compensation_feature_tables"]=int(selector_network_->anchors()->compensationFeatureTables().size());
        anchor_state["compensation_feature_dim"]=48;
        anchor_state["compensation_feature_interpolation"]="adjacent Feature48 linear in raw ratio; endpoint clamp; shared initial Feature across ratio knots";
        anchor_state["compensation_ratio_code_dim"]=3;
        const auto mode=selector_network_->anchors()->compensationResidualMode();
        anchor_state["compensation_residual_mode"]=mode;
        if(mode) {
          anchor_state["compensation_layout"]=mode==1?"Selector11; offset3 + base48 and cumulative ratio residual48 + ratio3; 64/64 four heads":"Selector11; offset3 + base48 and increasing-bandwidth cumulative residual48 + ratio3; 64/64 four heads";
          anchor_state["compensation_feature_interpolation"]="raw-ratio partial next residual; prefix receives all higher-ratio gradients; base random and residuals zero at creation";
          if(mode==2)for(double f:{.25,.5,1.,1.5,2.})anchor_state["compensation_spatial_max_cycles_per_anchor"].append(f);
        }
      }
      if(selector_network_->spatialCompensationEnabled()&&selector_network_->spatialCompensation()->sharedAnchor128()) {
        anchor_state["compensation_layout"]="Selector11; selected physical19 + one Anchor128 shared by ratios, gate=1+tanh(3->32->128), ratio3; 150->64->64 four heads; live joint Master gradients";
        anchor_state["compensation_parameters"]=35726;
        anchor_state["compensation_input_dim"]=150;
        if(selector_network_->spatialCompensation()->offsetOnlyInput()) {
          anchor_state["compensation_layout"]="Selector11; selected Anchor-local offset3 + shared Anchor128 with ratio gate + ratio3; 134->64->64 four heads; no other physical inputs";
          anchor_state["compensation_parameters"]=34702;
          anchor_state["compensation_input_dim"]=134;
        }
        if(selector_network_->spatialCompensation()->boundedXYZ()) {
          anchor_state["compensation_xyz_bound_anchor_units"]=2.;
          anchor_state["compensation_xyz_bound_transform"]="u / sqrt(1 + squared_norm(u)/4), then multiply Anchor reference scale";
          anchor_state["compensation_layout"]="Selector11; offset3 + shared gated Anchor128 + ratio3 ->64->64 four heads; XYZ vector norm smoothly bounded by 2x Anchor reference scale";
        }
        anchor_state["compensation_feature_tables"]=1;
        anchor_state["compensation_feature_dim"]=128;
        anchor_state["compensation_feature_interpolation"]="none; one shared Anchor128 and continuous ratio gate";
        anchor_state["compensation_ratio_code_dim"]=3;
      }
      anchor_state["global_xyz"] = selector_anchor_global_xyz_;
      anchor_state["offset_reference"] = (selector_anchor_global_xyz_ || selector_explicit_anchor_blocks_) ? "current_cell_center_and_size" : "root_center_and_size";
      anchor_state["explicit_blocks"] = selector_explicit_anchor_blocks_;
      if(selector_explicit_anchor_blocks_) {
        selector_network_->anchors()->assertBlocks(gaussians_->getXYZ().size(0));
        if(selector_variable_anchor_members_) {
          anchor_state["max_gaussians_per_anchor"]=10;
          anchor_state["topology_round"]=Json::Int64(selector_network_->anchors()->topologyRound().item<int64_t>());
          if(selector_anchor_only_pruning_>=3)anchor_state["gaussians_per_anchor"]=10;
        } else anchor_state["gaussians_per_anchor"]=10;
        anchor_state["max_level"]=2;
        anchor_state["growth"]= "Full-gradient q90 first-window tau0; alpha=2; stable UID collision buckets";
        anchor_state["growth_tau0"] = gaussians_->anchorGrowthThreshold();
        anchor_state["growth_alpha"] = 2.0;
        anchor_state["anchor_only_pruning"] = selector_anchor_only_pruning_!=0;
        anchor_state["topology_mode"] = selector_anchor_only_pruning_;
        anchor_state["topology_schedule"] = selector_variable_anchor_members_ && selector_anchor_only_pruning_<3
            ? "alternating Gaussian/Anchor" : "Anchor every topology event";
        anchor_state["anchor_refine_requires_full"] = selector_anchor_refine_requires_full_;
        anchor_state["gaussian_pruned_members"] = Json::UInt64(gaussians_->gaussianPrunedMembers());
        anchor_state["anchor_pruned_members"] = Json::UInt64(gaussians_->anchorPrunedMembers());
        anchor_state["local_anchor_migrated_members"] = Json::UInt64(gaussians_->localAnchorMigratedMembers());
        anchor_state["local_anchor_dropped_full"] = Json::UInt64(gaussians_->localAnchorDroppedMembers());
        anchor_state["local_anchor_migration"] = selector_anchor_only_pruning_>=3
            ? "whole Anchor final same-level cell; stationary resident then lowest UID wins; preserve surviving members/Feature/Adam"
            : selector_variable_anchor_members_
            ? "same-level destination; capacity=10; accepted-member-weighted Feature; compact after CD" : "legacy block transport";
        anchor_state["pruning"] = "Octree-GS anchor opacity sum / Full visible visits; sufficient visits only";
        anchor_state["prune_min_opacity"] = 0.005;
        anchor_state["prune_success_threshold"] = 0.8;
        anchor_state["prune_check_interval"] = densifyInterval();
        anchor_state["slot_deactivation"] = selector_slot_deactivation_;
        if (selector_slot_deactivation_)
          anchor_state["inactive_slots"] = Json::Int64(selector_network_->anchors()->inactiveSlots().sum().item<int64_t>());
        gaussians_->saveAnchorPruningState(ply_dir / "anchor_pruning_state.pt",densifyInterval());
      }
      anchor_state["ratio_encoding"] = selector_anchor_ratio_features_ ? "per_anchor_8d_linear_raw_ratio_interpolation_endpoint_clamp" : "log_r_joint_input";
      if(selector_anchor_ratio_features_)
        for(float ratio:selector_target_ratios_)anchor_state["ratio_feature_knots"].append(ratio);
      anchor_state["gaussian_count"] = Json::Int64(selector_network_->anchors()->rows().numel());
      anchor_state["anchor_count"] = Json::Int64(selector_network_->anchors()->size());
      anchor_state["feature_dim"] = selector_anchor_ratio_features_ ? selector_anchor_ratio_shared_dim_ : selector_anchor_feature_dim_;
      anchor_state["ratio_feature_dim"] = selector_anchor_ratio_features_ ? 8 : 0;
      anchor_state["total_feature_dim_per_anchor"] = selector_anchor_ratio_features_ ? int(selector_anchor_ratio_shared_dim_+8*selector_target_ratios_.size()) : selector_anchor_feature_dim_;
      if(selector_network_->anchors()->compensationFeatures())
        anchor_state["total_feature_dim_per_anchor"]=anchor_state["total_feature_dim_per_anchor"].asInt()+
            int(selector_network_->anchors()->compensationFeatureWidth()*selector_network_->anchors()->compensationFeatureTables().size());
      if(selector_network_->anchors()->spatialFeatures())anchor_state["total_feature_dim_per_anchor"]=anchor_state["total_feature_dim_per_anchor"].asInt()+int(selector_network_->anchors()->spatialFeaturesTable().size(1));
      anchor_state["voxel_size"] = selector_network_->anchors()->voxelSize();
      anchor_state["adaptive"] = selector_anchor_adaptive_;
      anchor_state["max_level"] = (selector_anchor_adaptive_ || selector_explicit_anchor_blocks_) ? 2 : 0;
      anchor_state["leaf_capacity"] = selector_anchor_adaptive_ ? 128 : 0;
      anchor_state["selection"] = selector_selection_mode_=="flexgs" ? "flexgs_gumbel_st_quantile" : "maturity_cached_bernoulli";
      anchor_state["maturity_mode"] = protection_config_.reliability_v2 ?
          (protection_config_.rendering_contribution ? "reliability_v2_rendering_contribution" : "reliability_v2") :
          (selector_protection_enabled_ ? "legacy" : "fixed_one");
      anchor_state["contribution_weight"] = protection_config_.rendering_contribution ? protection_config_.contribution_weight : 0.f;
      anchor_state["lod_start_iteration"] = lod_start_iteration_;
      if(selector_network_->spatialCompensationEnabled())anchor_state["physical_scale_additive"]=selector_network_->spatialCompensation()->physicalScaleAdditive();
      if(selector_network_->spatialCompensationEnabled())anchor_state["softplus_scale"]=selector_network_->spatialCompensation()->softplusScale();
      if(selector_network_->spatialCompensationEnabled())
        anchor_state["geometry_head_variant"]=selector_network_->spatialCompensation()->geometryHeadVariant();
      const bool statistics=selector_network_->selectorSetEncoderSchema()==3;
      anchor_state["anchor_set_encoder"]=selector_network_->selectorSetEncoder() && !statistics;
      anchor_state["anchor_set_statistics"]=statistics;
      anchor_state["anchor_context_residual"]=selector_network_->selectorContextResidual();
      anchor_state["view_residual"]=bool(selector_network_->selectorViewResidual());
      anchor_state["view_input"]=selector_network_->selectorViewInput();
      if(selector_network_->selectorSetEncoder()) {
        anchor_state["input_dim"]=statistics?21:50;
        anchor_state["selector_parameters"]=statistics?1826:3202;
        anchor_state["selector_layout"]=statistics?"[member10,log_ratio1,Anchor mean3/covariance6/count1]->32->32->2":
            "members10->16->16; Anchor mean16/max16/count1; [member16,context33,log_ratio1]->32->32->2";
        anchor_state["selector_set_encoder_schema"]=selector_network_->selectorSetEncoderSchema();
        anchor_state["selector_context"]=statistics?
            "slot-eligible mean and population covariance of nominal-cell-relative XYZ/offsetScale; count/10; no learned encoder":
            selector_network_->selectorSetEncoderSchema()==2?
            "slot-eligible member pooling, nominal-cell-relative XYZ/offsetScale, no private features":
            "complete pre-selection member sets, centroid-relative XYZ/offsetScale, no private features";
        anchor_state["selector_execution"]=statistics?
            "Statistics21 input and MLP with optional CUDA fusion; FlexGS sampler unchanged":
            "LibTorch Set Encoder and MLP; FlexGS sampler unchanged";
      }
      if(selector_network_->selectorContextResidual()) {
        anchor_state["input_dim"]=11;
        anchor_state["selector_parameters"]=2050;
        anchor_state["selector_context_residual_schema"]=1;
        anchor_state["selector_layout"]="11->32 ReLU; first8 eligible mean/max/count17->32 residual into second32 pre-ReLU; ->2";
        anchor_state["selector_context"]="slot-eligible learned first-layer channels, original 11D path retained; zero-initialized bias-free Anchor projection";
        anchor_state["selector_execution"]="Optional CUDA pooled residual MLP; no separate member encoder; FlexGS sampler unchanged";
      }
      anchor_state["iteration"] = getIteration();
      std::ofstream anchor_output(ply_dir / "anchor_selector_state.json");
      anchor_output << anchor_state;
    }
  }
  if (transform_field_) {
    torch::serialize::OutputArchive transform_archive;
    transform_field_->save(transform_archive);
    transform_archive.save_to((result_dir / "transform_field.pt").string());
    if (transform_optimizer_)
      torch::save(*transform_optimizer_,
                  (result_dir / "transform_optimizer.pt").string());
  }
  if (transform_field_config_.enabled && selector_network_) {
    Json::Value state;
    state["initialized"] = static_cast<bool>(transform_field_);
    state["start_iter"] = transform_field_config_.start_iter;
    state["transform_step"] = Json::UInt64(normal_update_clock_.tf_updates);
    state["rejected_mapping_updates"] = Json::UInt64(normal_update_clock_.rejected_updates);
    state["mapping_iter"] = Json::UInt64(mapping_iter_);
    state["online_basic_mappings"] = Json::UInt64(0);
    state["online_enhanced_mappings"] = Json::UInt64(ultrafast_rotation_.online_ended
        ? ultrafast_rotation_.online_end_rotation_step : ultrafast_rotation_.step);
    state["post_slam_mappings"] = Json::UInt64(ultrafast_rotation_.tailSteps());
    state["force_tail_optimization_enabled"] = tail_optimization_schedule_.enabled;
    state["tail_start_iteration"] = Json::UInt64(tail_optimization_schedule_.start_iteration);
    state["tail_start_mapping"] = Json::UInt64(tail_optimization_schedule_.start_mapping);
    state["tail_target_steps"] = Json::UInt64(tail_optimization_schedule_.target_steps);
    state["tail_completed_steps"] = Json::UInt64(tail_optimization_schedule_.completed(mapping_iter_));
    state["selector_step"] = Json::UInt64(selector_step_);
    state["guard_accept"] = Json::UInt64(gradient_accept_count_);
    state["guard_shrink"] = Json::UInt64(gradient_shrink_count_);
    state["guard_fallback"] = Json::UInt64(gradient_fallback_count_);
    state["guard_cancel"] = Json::UInt64(gradient_cancel_count_);
    state["guard_noop"] = Json::UInt64(gradient_noop_count_);
    state["guard_would_cancel"] = Json::UInt64(gradient_would_cancel_count_);
    state["replay_attempts"] = Json::UInt64(replay_attempt_count_);
    state["replay_commits"] = Json::UInt64(replay_commit_count_);
    state["replay_skipped"] = Json::UInt64(replay_skip_count_);
    state["replay_enqueued"] = Json::UInt64(replay_enqueued_count_);
    state["replay_tf_updates"] = Json::UInt64(replay_tf_updates_);
    state["replay_ratio_cursor"] = Json::UInt64(replay_ratio_cursor_);
    state["replay_pending"] = Json::UInt64(conflict_replay_manager_->queueSize());
    state["replay_unfinished_cancel_window"] = Json::UInt64(conflict_replay_manager_->pendingCancellationCount());
    state["cd_selector_steps"] = Json::UInt64(cd_selector_steps_);
    state["cd_tf_steps"] = Json::UInt64(cd_tf_steps_);
    state["spatial_resolution"] =
        transform_field_config_.hexplane.spatial_resolution;
    state["ratio_resolution"] =
        transform_field_config_.hexplane.ratio_resolution;
    state["feature_dim"] = transform_field_config_.hexplane.feature_dim;
    state["hidden_dim"] = transform_field_config_.hidden_dim;
    state["lr_delay_mult"] = transform_field_config_.lr_delay_mult;
    state["lr_max_steps"] = transform_field_config_.lr_max_steps;
    state["multires"] = Json::arrayValue;
    for (const auto resolution : transform_field_config_.hexplane.multires)
      state["multires"].append(resolution);
    state["outside_identity"] = transform_field_config_.outside_identity;
    state["continuous_maturity"] = true;
    state["elastic_entry_maturity"] = transform_field_config_.elastic_entry_maturity;
    state["ready_maturity"] = transform_field_config_.ready_maturity;
    state["min_ready_gaussians"] = transform_field_config_.min_ready_gaussians;
    state["tf_initialization_iteration"] = Json::Int64(continuous_tf_statistics_.tf_initialization_iteration);
    state["first_ready_population_iteration"] = Json::Int64(continuous_tf_statistics_.first_ready_population_iteration);
    state["ready_count_when_initialized"] = Json::Int64(continuous_tf_statistics_.ready_count_when_initialized);
    state["initial_aabb_population"] = Json::Int64(continuous_tf_statistics_.initial_aabb_population);
    state["start_mode"] = transform_field_config_.start_mode;
    state["maturity_compute_epsilon"] = transform_field_config_.maturity_compute_epsilon;
    state["tf_start_mapping_iteration"] = Json::Int64(continuous_tf_statistics_.tf_start_mapping_iteration);
    state["first_valid_maturity_cycle_iteration"] = Json::Int64(gaussians_->protection_.first_valid_maturity_cycle_iteration);
    state["aabb_min"] = Json::arrayValue;
    state["aabb_max"] = Json::arrayValue;
    if (transform_field_) {
      auto aabb_min = transform_field_->aabbMin().detach().cpu();
      auto aabb_max = transform_field_->aabbMax().detach().cpu();
      for (int i = 0; i < 3; ++i) {
        state["aabb_min"].append(aabb_min.index({i}).item<float>());
        state["aabb_max"].append(aabb_max.index({i}).item<float>());
      }
    }
    std::ofstream transform_state(result_dir / "transform_state.json");
    if (!transform_state.is_open())
      throw std::runtime_error("Cannot open Transform Field state file");
    Json::StreamWriterBuilder builder;
    transform_state << Json::writeString(builder, state);
  }
  gaussians_->saveSparsePointsPly(result_dir / "input.ply");
}

void GaussianMapper::saveViewBankOrbMap(std::filesystem::path result_dir) {
  if (!pSLAM_ || !pSLAM_->getAtlas()) return;

  std::map<uint64_t, std::shared_ptr<GaussianKeyframe>> keyframes;
  for (const auto& [id, keyframe] : historical_keyframes_)
    if (keyframe) keyframes.emplace(id, keyframe);

  std::map<uint64_t, std::set<uint64_t>> orb_associations;
  std::map<uint64_t, std::array<float, 3>> orb_points;
  auto map = pSLAM_->getAtlas()->GetCurrentMap();
  if (map) {
    std::unique_lock<std::mutex> lock_map(map->mMutexMapUpdate);
    for (auto* keyframe : map->GetAllKeyFrames()) {
      if (!keyframe || keyframe->isBad()) continue;
      auto& associated = orb_associations[keyframe->mnId];
      for (auto* point : keyframe->GetMapPoints())
        if (point && !point->isBad()) associated.insert(point->mnId);
    }
    for (auto* point : map->GetAllMapPoints()) {
      if (!point || point->isBad()) continue;
      const auto pos = point->GetWorldPos();
      if (std::isfinite(pos.x()) && std::isfinite(pos.y()) && std::isfinite(pos.z()))
        orb_points.emplace(point->mnId, std::array<float, 3>{pos.x(), pos.y(), pos.z()});
    }
  }

  std::vector<std::size_t> recent_ids = recent_window_importance_
      ? recent_window_importance_->keyframeIds() : std::vector<std::size_t>{};
  std::vector<std::shared_ptr<GaussianKeyframe>> all_keyframes;
  all_keyframes.reserve(keyframes.size());
  for (const auto& [id, keyframe] : keyframes) all_keyframes.push_back(keyframe);
  const auto outlier_views = historical_outlier_selector_
      ? historical_outlier_selector_->select(all_keyframes, recent_ids)
      : std::vector<HistoricalOutlierView>{};

  std::map<std::string, std::vector<uint64_t>> bank_ids;
  bank_ids["RecentWindow"] = std::vector<uint64_t>(recent_ids.begin(), recent_ids.end());
  if (global_gi_) {
    const auto& bank = global_gi_->bank();
    bank_ids["GlobalGI anchor"] = std::vector<uint64_t>(bank.anchors().begin(), bank.anchors().end());
    bank_ids["GlobalGI representative"] = std::vector<uint64_t>(bank.representatives().begin(), bank.representatives().end());
  }
  for (const auto& view : outlier_views) bank_ids["HistoricalOutlier"].push_back(view.keyframe_id);

  Json::Value root(Json::objectValue);
  root["scene"] = "fr3_long_office_household";
  root["iteration"] = Json::UInt64(getIteration());
  root["source"] = "runtime ORB-SLAM3 MapPoint associations at final save";
  root["banks"] = Json::arrayValue;
  for (const auto& [name, ids] : bank_ids) {
    Json::Value bank(Json::objectValue);
    bank["name"] = name;
    bank["keyframes"] = Json::arrayValue;
    bank["points"] = Json::arrayValue;
    std::set<uint64_t> point_ids;
    for (const auto id : ids) {
      bank["keyframes"].append(Json::UInt64(id));
      const auto orb_it = orb_associations.find(id);
      if (orb_it != orb_associations.end())
        point_ids.insert(orb_it->second.begin(), orb_it->second.end());
    }
    for (const auto point_id : point_ids) {
      const auto point_it = orb_points.find(point_id);
      if (point_it == orb_points.end()) continue;
      Json::Value point(Json::objectValue);
      point["id"] = Json::UInt64(point_id);
      point["p"] = Json::arrayValue;
      for (const auto coordinate : point_it->second) point["p"].append(coordinate);
      bank["points"].append(point);
    }
    root["banks"].append(bank);
  }

  root["keyframes"] = Json::arrayValue;
  for (const auto& [id, keyframe] : keyframes) {
    Json::Value item(Json::objectValue);
    item["id"] = Json::UInt64(id);
    const auto pose = keyframe->getPosef().inverse().translation();
    item["p"] = Json::arrayValue;
    item["p"].append(pose.x()); item["p"].append(pose.y()); item["p"].append(pose.z());
    item["banks"] = Json::arrayValue;
    for (const auto& [name, ids] : bank_ids)
      if (std::find(ids.begin(), ids.end(), id) != ids.end()) item["banks"].append(name);
    root["keyframes"].append(item);
  }

  std::ofstream output(result_dir / "view_bank_orb.json");
  if (!output.is_open()) throw std::runtime_error("Cannot open view bank ORB export");
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  output << Json::writeString(builder, root);
}

void GaussianMapper::saveSelectorConfig(std::filesystem::path result_dir) {
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
  const auto config_path = result_dir / "selector_config.yaml";
  std::ofstream settings_file(config_path);
  if (!settings_file.is_open()) {
    throw std::runtime_error("Cannot write selector config: " +
                             config_path.string());
  }

  settings_file << std::setprecision(9) << "%YAML:1.0\n\n";
  settings_file << "Training.ultra_light_mode: " << ultra_light_mode_ << '\n';
  settings_file << "Training.real_lod: " << int(real_lod_.enabled) << '\n';
  settings_file << "RealLoD.full_loss_weight: " << real_lod_full_loss_weight_ << '\n';
  settings_file << "ViewBudget.enabled: " << int(view_budget_enabled_) << '\n';
  settings_file << "ViewBudget.view_residual: " << int(view_budget_view_residual_) << '\n';
  settings_file << "ViewBudget.conditioning: \"" << (view_budget_conditioning_==1?"mode":(view_budget_conditioning_==2?"view":
      (view_budget_conditioning_==3?"selector_view":(view_budget_conditioning_==4?"compensator_view":"off")))) << "\"\n";
  settings_file << "ViewBudget.view_residual_input: \"" << view_budget_residual_input_ << "\"\n";
  settings_file << "ViewBudget.loss_weight: " << view_budget_loss_weight_ << '\n';
  settings_file << "ViewBudget.li_weight: " << view_budget_li_weight_ << '\n';
  settings_file << "ViewBudget.global_control: " << int(view_budget_global_control_) << '\n';
  settings_file << "ViewBudget.seed: " << view_budget_seed_ << '\n';
  settings_file << "MixedRatio.enabled: " << int(mixed_ratio_config_.enabled) << '\n';
  settings_file << "ViewResidual.enabled: " << int(selector_view_residual_enabled_) << '\n';
  settings_file << "ViewInput.enabled: " << int(selector_view_input_enabled_) << '\n';
  settings_file << "ViewInput.single_view_gi_weight: " << selector_view_gi_weight_ << '\n';
  settings_file << "ViewInput.single_view_gi_lowest_two_only: " << int(selector_view_gi_lowest_two_only_) << '\n';
  settings_file << "ViewInput.visible_point_budget: " << int(selector_view_visible_budget_) << '\n';
  if(!selector_view_ratio_indices_.empty()) {
    settings_file << "ViewInput.target_ratios: [";
    for(std::size_t i=0;i<selector_view_ratio_indices_.size();++i)
      settings_file << (i?", ":"") << selector_target_ratios_[selector_view_ratio_indices_[i]];
    settings_file << "]\n";
  }
  settings_file << "MixedRatio.seed: " << mixed_ratio_config_.seed << '\n';
  settings_file << "MixedRatio.concentration_high: " << mixed_ratio_config_.concentration[0] << '\n';
  settings_file << "MixedRatio.concentration_middle: " << mixed_ratio_config_.concentration[1] << '\n';
  settings_file << "MixedRatio.concentration_low: " << mixed_ratio_config_.concentration[2] << '\n';
  settings_file << "MixedRatio.eta_min: " << mixed_ratio_config_.eta_min << '\n';
  settings_file << "MixedRatio.eta_max: " << mixed_ratio_config_.eta_max << '\n';
  settings_file << "LoDHealth.enabled: " << int(lod_health_guard_.enabled) << '\n';
  settings_file << "LoDHealth.zero_patience: " << lod_health_guard_.zero_patience << '\n';
  settings_file << "LoDHealth.saturation_patience: " << lod_health_guard_.saturation_patience << '\n';
  settings_file << "LoDHealth.probability_floor: " << lod_health_guard_.probability_floor << '\n';
  settings_file << "LoDHealth.scale_factor_limit: " << lod_health_guard_.scale_limit << '\n';
  settings_file << "LoDHealth.xyz_anchor_limit: " << lod_health_guard_.xyz_limit << '\n';
  settings_file << "LoDHealth.tail_fraction: " << lod_health_guard_.tail_fraction << '\n';
  settings_file << "FlexGSImportance.enabled: " << int(flexgs_importance_enabled_) << '\n';
  settings_file << "FlexGSImportance.refresh_interval: " << flexgs_importance_interval_ << '\n';
  settings_file << "Mapping.iteration_logging: " << int(mapping_iteration_logging_) << '\n';
  settings_file << "CUDA.fused_dense_adam: " << int(fused_dense_adam_) << '\n';
  settings_file << "CUDA.fused_compensation_decode: " << int(fused_compensation_decode_) << '\n';
  settings_file << "CUDA.fused_anchor_input_pack: " << int(fused_anchor_input_pack_) << '\n';
  settings_file << "CUDA.fused_spatial_compensation: " << int(fused_spatial_compensation_) << '\n';
  settings_file << "CUDA.fused_anchor_full_stats: " << int(fused_anchor_full_stats_) << '\n';
  settings_file << "CUDA.fused_selector_compaction: " << int(fused_selector_compaction_) << '\n';
  settings_file << "CUDA.fused_flexgs_selector: " << int(fused_flexgs_selector_) << '\n';
  settings_file << "CUDA.cached_anchor_context: " << int(cached_anchor_context_) << '\n';
  settings_file << "CUDA.reuse_renderer_workspace: " << int(reuse_renderer_workspace_) << '\n';
  settings_file << "CUDA.optimized_ssim_backward: " << int(optimized_ssim_backward_) << '\n';
  settings_file << "CUDA.cached_ssim_statistics: " << int(cached_ssim_statistics_) << '\n';
  settings_file << "CUDA.cache_ratio_code: " << int(cache_ratio_code_) << '\n';
  settings_file << "CUDA.elide_inactive_maturity_gate: " << int(elide_inactive_maturity_gate_) << '\n';
  settings_file << "CUDA.reuse_maturity_softmax: " << int(reuse_maturity_softmax_) << '\n';
  settings_file << "CUDA.fused_anchor128_input: " << int(fused_anchor128_input_) << '\n';
  settings_file << "CUDA.fused_residual48_input: " << int(fused_residual48_input_) << '\n';
  settings_file << "CUDA.fused_residual48_training_input: " << int(fused_residual48_training_input_) << '\n';
  settings_file << "CUDA.fused_residual48_heads: " << int(fused_residual48_heads_) << '\n';
  settings_file << "CUDA.parallel_residual48_heads: " << int(parallel_residual48_heads_) << '\n';
  settings_file << "CUDA.fused_maturity_loss: " << int(fused_maturity_loss_) << '\n';
  settings_file << "CUDA.fused_l1_loss: " << int(fused_l1_loss_) << '\n';
  settings_file << "CUDA.fused_empty_tf_statistics: " << int(fused_empty_tf_statistics_) << '\n';
  settings_file << "Training.post_topology_full_steps: " << ultrafast_rotation_.post_topology_full_steps << '\n';
  settings_file << "Training.serial_full_lod: " << int(ultrafast_rotation_.serial_full_lod) << '\n';
  settings_file << "Training.tail_ratio_rotation: " << int(ultrafast_rotation_.rotate_tail) << '\n';
  settings_file << "Training.serial_lod_detach_master: " << int(ultrafast_rotation_.serial_lod_detach_master) << '\n';
  settings_file << "Training.force_tail_optimization_enabled: "
                << static_cast<int>(tail_optimization_schedule_.enabled) << '\n';
  settings_file << "Training.tail_optimization_steps: "
                << tail_optimization_schedule_.steps << "\n\n";
  settings_file << "Selector.enabled: " << static_cast<int>(selector_enabled_)
                << '\n';
  settings_file << "Selector.backend: \"" << selector_backend_ << "\"\n";
  settings_file << "Selector.architecture: \"" << selector_architecture_ << "\"\n";
  settings_file << "Selector.anchor_voxel_size: " << selector_anchor_voxel_size_ << '\n';
  settings_file << "Selector.anchor_gaussian_attributes: " << static_cast<int>(selector_anchor_gaussian_attributes_) << '\n';
  settings_file << "Selector.anchor_global_xyz: " << static_cast<int>(selector_anchor_global_xyz_) << '\n';
  settings_file << "Selector.anchor_ratio_features: " << static_cast<int>(selector_anchor_ratio_features_) << '\n';
  settings_file << "Selector.anchor_ratio_shared_dim: " << selector_anchor_ratio_shared_dim_ << '\n';
  settings_file << "Selector.anchor_ratio_scalar_input: \"" << selector_anchor_ratio_scalar_input_ << "\"\n";
  settings_file << "Selector.anchor_feature_dim: " << selector_anchor_feature_dim_ << '\n';
  settings_file << "Selector.anchor_adaptive: " << static_cast<int>(selector_anchor_adaptive_) << '\n';
  settings_file << "Selector.explicit_anchor_blocks: " << static_cast<int>(selector_explicit_anchor_blocks_) << '\n';
  settings_file << "Selector.variable_anchor_members: " << static_cast<int>(selector_variable_anchor_members_) << '\n';
  settings_file << "Selector.anchor_only_pruning: " << int(selector_anchor_only_pruning_) << '\n';
  settings_file << "Selector.anchor_refine_requires_full: " << int(selector_anchor_refine_requires_full_) << '\n';
  settings_file << "Selector.compensation: " << static_cast<int>(selector_compensation_) << '\n';
  settings_file << "Selector.compensation_start_iter: " << selector_compensation_start_iter_ << '\n';
  settings_file << "Selector.physical_scale_additive: " << int(selector_physical_scale_additive_) << '\n';
  settings_file << "Selector.softplus_scale: " << int(selector_softplus_scale_) << '\n';
  settings_file << "Selector.geometry_head_variant: " << (selector_network_&&selector_network_->spatialCompensationEnabled()?
      selector_network_->spatialCompensation()->geometryHeadVariant():0) << '\n';
  settings_file << "Training.lod_start_iteration: " << lod_start_iteration_ << '\n';
  settings_file << "Selector.compensation_method: \"" << selector_compensation_method_ << "\"\n";
  settings_file << "Selector.compensation_fourier_bands: " << selector_compensation_fourier_bands_ << '\n';
  settings_file << "Selector.slot_deactivation: " << static_cast<int>(selector_slot_deactivation_) << '\n';
  settings_file << "Selector.anchor_frustum_culling: " << int(selector_anchor_frustum_culling_) << '\n';
  settings_file << "Selector.anchor_set_encoder: " << int(selector_anchor_set_encoder_) << '\n';
  settings_file << "Selector.anchor_set_statistics: " << int(selector_anchor_set_statistics_) << '\n';
  settings_file << "Selector.anchor_context_residual: " << int(selector_anchor_context_residual_) << '\n';
  settings_file << "Selector.target_ratios: [";
  for (std::size_t i = 0; i < selector_target_ratios_.size(); ++i) {
    if (i != 0) settings_file << ", ";
    settings_file << (selector_view_visible_budget_?selector_view_budget_ratio_values_[i]:selector_target_ratios_[i]);
  }
  settings_file << "]\n";
  settings_file << "Selector.protection_enabled: " << int(selector_protection_enabled_) << '\n';
  settings_file << "MaturityV2.enabled: " << protection_config_.reliability_v2 << '\n';
  settings_file << "MaturityV2.contribution_enabled: " << int(protection_config_.rendering_contribution) << '\n';
  settings_file << "MaturityV2.contribution_decay: " << protection_config_.contribution_decay << '\n';
  settings_file << "MaturityV2.tau_r: " << protection_config_.contribution_tau << '\n';
  settings_file << "MaturityV2.contribution_weight: " << protection_config_.contribution_weight << '\n';
  settings_file << "MaturityV2.contribution_every_full: " << protection_config_.contribution_every_full << '\n';
  settings_file << "MaturityV2.contribution_pixel_stride: " << protection_config_.contribution_pixel_stride << '\n';
  settings_file << "MaturityV2.tau_o: " << protection_config_.reliability_tau_o << '\n';
  settings_file << "MaturityV2.tau_v: " << protection_config_.reliability_tau_v << '\n';
  settings_file << "MaturityV2.xyz_scale_floor: " << protection_config_.reliability_xyz_floor << '\n';
  settings_file << "MaturityV2.inherit_fraction: " << protection_config_.reliability_inherit << '\n';
  settings_file << "MaturityV2.selector_loss_weight: " << protection_config_.reliability_loss_weight << '\n';
  settings_file << "MaturityV2.selector_delta: " << protection_config_.reliability_delta << '\n';
  settings_file << "MaturityV2.compensation_gate: " << protection_config_.reliability_gate << '\n';
  settings_file << "MaturityV2.gate_appearance: " << protection_config_.reliability_gate_appearance << '\n';
  settings_file << "MaturityV2.gate_low: " << protection_config_.reliability_gate_low << '\n';
  settings_file << "MaturityV2.gate_high: " << protection_config_.reliability_gate_high << '\n';
  settings_file << "Protection.drift_temperature: " << protection_config_.drift_temperature << '\n';
  settings_file << "Protection.beta_fast: " << protection_config_.beta_fast << '\n';
  settings_file << "Protection.beta_slow: " << protection_config_.beta_slow << '\n';
  settings_file << "Protection.maturity_rise_rate: " << protection_config_.maturity_rise_rate << '\n';
  settings_file << "Protection.maturity_fall_rate: " << protection_config_.maturity_fall_rate << '\n';
  settings_file << "Protection.maturity_diagnostics: " << int(protection_config_.maturity_diagnostics) << '\n';
  settings_file << "Protection.compatibility_mature_threshold: " << protection_config_.compatibility_mature_threshold << '\n';
  settings_file << "Protection.performance_logging: " << int(protection_config_.performance_logging) << '\n';
  settings_file << "Selector.temperature: " << selector_temperature_ << '\n';
  settings_file << "Selector.selection_mode: \"" << selector_selection_mode_ << "\"\n";
  settings_file << "Selector.gumbel_budget_normalize: " << int(selector_gumbel_budget_normalize_) << '\n';
  settings_file << "Selector.learning_rate: " << selector_learning_rate_ << '\n';
  settings_file << "Selector.render_loss_weight: "
                << selector_render_loss_weight_ << '\n';
  settings_file << "AnchorSI.enabled: " << int(anchor_si_config_.enabled) << "\n";
  settings_file << "AnchorSI.loss_weight: " << anchor_si_config_.weight << "\n";
  settings_file << "AnchorSI.ema_decay: " << anchor_si_config_.decay << "\n";
  settings_file << "AnchorSI.temperature: " << anchor_si_config_.temperature << "\n";
  settings_file << "AnchorSI.collect_every_full: " << anchor_si_config_.collect_every_full << "\n";
  settings_file << "AnchorSI.minimum_visits: " << anchor_si_config_.minimum_visits << "\n";
  settings_file << "AnchorSI.maximum_age_full: " << anchor_si_config_.maximum_age_full << "\n";
  settings_file << "Selector.gi_loss_weight: " << selector_gi_loss_weight_
                << '\n';
  settings_file << "Selector.ratio_loss_weight: "
                << selector_ratio_loss_weight_ << '\n';
  settings_file << "Selector.lod_render_scale: "
                << selector_lod_render_scale_ << '\n';
  settings_file << "Selector.detach_gaussian_parameters: "
                << static_cast<int>(selector_detach_gaussian_parameters_)
                << '\n';
  settings_file << "Selector.replay_enabled: "
                << static_cast<int>(selector_replay_enabled_) << '\n';
  settings_file << "Selector.replay_interval: " << selector_replay_interval_
                << '\n';
  settings_file << "Selector.replay_num_frames: "
                << selector_replay_num_frames_ << '\n';
  settings_file << "ConflictReplay.trigger_cancel_count: " << conflict_replay_config_.trigger_cancel_count << '\n';
  settings_file << "ConflictReplay.source_view_count: " << conflict_replay_config_.source_view_count << '\n';
  settings_file << "ConflictReplay.views_per_source: " << conflict_replay_config_.views_per_source << '\n';
  settings_file << "ConflictReplay.remote_candidate_count: " << conflict_replay_config_.remote_candidate_count << '\n';
  settings_file << "ConflictReplay.queue_capacity: " << conflict_replay_config_.queue_capacity << "\n\n";
  settings_file << "Selector.max_mapping_iterations: "
                << selector_max_mapping_iterations_ << "\n\n";

  settings_file << "Mapping.record_keyframe_used_times: "
                << static_cast<int>(record_keyframe_used_times_) << "\n\n";

  // Retired experiment parameters are recorded in cfg/selector/archive/retired_features.yaml.
  settings_file << "GlobalGI.mode: \"" << global_gi_config_.mode << "\"\n";
  settings_file << "GlobalGI.view_ratio: " << global_gi_config_.view_ratio << '\n';
  settings_file << "GlobalGI.coverage_anchor_fraction: " << global_gi_config_.coverage_anchor_fraction << '\n';
  settings_file << "GlobalGI.sampling_seed: " << global_gi_config_.sampling_seed << '\n';
  settings_file << "GlobalGI.refresh_interval: " << global_gi_config_.refresh_interval << '\n';
  settings_file << "GlobalGI.max_views: " << global_gi_config_.max_views << '\n';
  settings_file << "GlobalGI.performance_logging: " << static_cast<int>(global_gi_config_.performance_logging) << '\n';
  settings_file << "GlobalGI.verbose_logging: " << static_cast<int>(global_gi_config_.verbose_logging) << '\n';
  settings_file << "GlobalGI.marginal_shadow: " << static_cast<int>(global_gi_config_.marginal_shadow) << '\n';
  settings_file << "GlobalGI.marginal_diagnostics: " << static_cast<int>(global_gi_config_.marginal_diagnostics) << '\n';
  settings_file << "GlobalGI.HierarchicalRepresentative.enable: " << static_cast<int>(global_gi_config_.hierarchical_representative_enabled) << '\n';
  settings_file << "GlobalGI.HierarchicalRepresentative.diagnostics: " << static_cast<int>(global_gi_config_.hierarchical_representative_diagnostics) << '\n';
  settings_file << "Importance.ratio_aware_weight: "
                << static_cast<int>(importance_ratio_aware_weight_) << '\n';
  settings_file << "Importance.ratio_weights: [";
  for (std::size_t i = 0; i < importance_ratio_weights_.size(); ++i) {
    if (i) settings_file << ", ";
    settings_file << importance_ratio_weights_[i];
  }
  settings_file << "]\n\n";

  settings_file << "TransformField.backend: \"" << transform_field_config_.backend << "\"\n";
  settings_file << "TransformField.enable: "
                << static_cast<int>(transform_field_config_.enabled) << '\n';
  settings_file << "TransformField.start_iter: "
                << transform_field_config_.start_iter << '\n';
  settings_file << "TransformField.aabb_scale: "
                << transform_field_config_.aabb_scale << '\n';
  settings_file << "TransformField.spatial_resolution: "
                << transform_field_config_.hexplane.spatial_resolution << '\n';
  settings_file << "TransformField.ratio_resolution: "
                << transform_field_config_.hexplane.ratio_resolution << '\n';
  settings_file << "TransformField.feature_dim: "
                << transform_field_config_.hexplane.feature_dim << '\n';
  settings_file << "TransformField.multires: [";
  const auto& multires = transform_field_config_.hexplane.multires;
  for (std::size_t i = 0; i < multires.size(); ++i) {
    if (i != 0) settings_file << ", ";
    settings_file << multires[i];
  }
  settings_file << "]\n";
  settings_file << "TransformField.hidden_dim: "
                << transform_field_config_.hidden_dim << '\n';
  settings_file << "TransformField.grid_lr_init: "
                << transform_field_config_.grid_lr_init << '\n';
  settings_file << "TransformField.grid_lr_final: "
                << transform_field_config_.grid_lr_final << '\n';
  settings_file << "TransformField.mlp_lr_init: "
                << transform_field_config_.mlp_lr_init << '\n';
  settings_file << "TransformField.mlp_lr_final: "
                << transform_field_config_.mlp_lr_final << '\n';
  settings_file << "TransformField.lr_delay_mult: "
                << transform_field_config_.lr_delay_mult << '\n';
  settings_file << "TransformField.lr_max_steps: "
                << transform_field_config_.lr_max_steps << '\n';
  settings_file << "TransformField.spatial_smooth_weight: "
                << transform_field_config_.spatial_smooth_weight << '\n';
  settings_file << "TransformField.ratio_smooth_weight: "
                << transform_field_config_.ratio_smooth_weight << '\n';
  settings_file << "TransformField.ratio_l1_weight: "
                << transform_field_config_.ratio_l1_weight << '\n';
  settings_file << "TransformField.outside_identity: "
                << static_cast<int>(transform_field_config_.outside_identity)
                << '\n';
  settings_file << "Elastic.entry_maturity: " << transform_field_config_.elastic_entry_maturity << '\n';
  settings_file << "TransformField.ready_maturity: " << transform_field_config_.ready_maturity << '\n';
  settings_file << "TransformField.min_ready_gaussians: " << transform_field_config_.min_ready_gaussians << '\n';
  settings_file << "TransformField.start_mode: " << transform_field_config_.start_mode << '\n';
  settings_file << "TransformField.maturity_compute_epsilon: " << transform_field_config_.maturity_compute_epsilon << '\n';
  settings_file << "TransformField.dynamic_aabb: "
                << static_cast<int>(dynamic_aabb_config_.enabled) << '\n';
  settings_file << "TransformField.aabb_check_interval: "
                << dynamic_aabb_config_.check_interval << '\n';
  settings_file << "TransformField.outside_count_threshold: "
                << dynamic_aabb_config_.outside_count_threshold << '\n';
  settings_file << "TransformField.expand_margin_ratio: "
                << dynamic_aabb_config_.expansion_margin << '\n';
  settings_file << "TransformField.bound_quantile_low: "
                << dynamic_aabb_config_.bound_quantile_low << '\n';
  settings_file << "TransformField.bound_quantile_high: "
                << dynamic_aabb_config_.bound_quantile_high << '\n';
  settings_file << "TransformField.fixed_resolution: "
                << static_cast<int>(dynamic_aabb_config_.fixed_resolution)
                << "\n\n";

  settings_file << "ChangeDistill.enable: "
                << static_cast<int>(change_distill_config_.enabled) << '\n';
  settings_file << "ChangeDistill.distill_selector_on_geometry_change: "
                << static_cast<int>(
                       change_distill_config_.distill_selector_on_geometry_change)
                << '\n';
  settings_file << "ChangeDistill.distill_tf_on_geometry_change: "
                << static_cast<int>(
                       change_distill_config_.distill_tf_on_geometry_change)
                << '\n';
  settings_file << "ChangeDistill.distill_tf_on_aabb_change: "
                << static_cast<int>(
                       change_distill_config_.distill_tf_on_aabb_change)
                << '\n';
  settings_file << "ChangeDistill.position_weight: "
                << change_distill_config_.position_weight << '\n';
  settings_file << "ChangeDistill.scale_weight: "
                << change_distill_config_.scale_weight << '\n';
  settings_file << "ChangeDistill.rotation_weight: "
                << change_distill_config_.rotation_weight << '\n';
  settings_file << "ChangeDistill.selector_weight: "
                << change_distill_config_.selector_weight << '\n';
  settings_file << "ChangeDistill.selector_learning_rate: "
                << change_distill_config_.selector_learning_rate << '\n';
  settings_file << "ChangeDistill.tf_grid_learning_rate: "
                << change_distill_config_.tf_grid_learning_rate << '\n';
  settings_file << "ChangeDistill.tf_mlp_learning_rate: "
                << change_distill_config_.tf_mlp_learning_rate << '\n';
  settings_file << "ChangeDistill.batch_size: "
                << change_distill_config_.batch_size << '\n';
  settings_file << "ChangeDistill.post_change_replay: "
                << static_cast<int>(change_distill_config_.post_change_replay)
                << '\n';
  settings_file << "ChangeDistill.epochs: " << change_distill_config_.epochs << std::endl;
  if (change_distill_config_.selector_epochs)
    settings_file << "ChangeDistill.selector_epochs: "
                  << *change_distill_config_.selector_epochs << '\n';
  if (change_distill_config_.tf_epochs)
    settings_file << "ChangeDistill.tf_epochs: "
                  << *change_distill_config_.tf_epochs << '\n';
  settings_file << "ChangeDistill.seed: " << change_distill_config_.seed << std::endl;
  settings_file << "ChangeDistill.enable_unaffected_anchors: "
                << static_cast<int>(change_distill_config_.enable_unaffected_anchors) << std::endl;
  settings_file << "ChangeDistill.anchor_ratio_to_corrected: "
                << change_distill_config_.anchor_ratio_to_corrected << std::endl;
  settings_file << "ChangeDistill.anchor_max_count: "
                << change_distill_config_.anchor_max_count << std::endl;
  settings_file << "ChangeDistill.anchor_loss_weight: "
                << change_distill_config_.anchor_loss_weight << std::endl;
  settings_file.flush();
  if (!settings_file.good()) {
    throw std::runtime_error("Failed while writing selector config: " +
                             config_path.string());
  }
}

void GaussianMapper::keyframesToJson(std::filesystem::path result_dir) {
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

  std::filesystem::path result_path = result_dir / "cameras.json";
  std::ofstream out_stream;
  out_stream.open(result_path);
  if (!out_stream.is_open())
    throw std::runtime_error("Cannot open json file at " +
                             result_path.string());

  Json::Value json_root;
  Json::StreamWriterBuilder builder;
  const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

  int i = 0;
  for (const auto& kfit : scene_->keyframes()) {
    const auto pkf = kfit.second;
    Eigen::Matrix4f Rt;
    Rt.setZero();
    Eigen::Matrix3f R = pkf->R_quaternion_.toRotationMatrix().cast<float>();
    Rt.topLeftCorner<3, 3>() = R;
    Eigen::Vector3f t = pkf->t_.cast<float>();
    Rt.topRightCorner<3, 1>() = t;
    Rt(3, 3) = 1.0f;

    Eigen::Matrix4f Twc = Rt.inverse();
    Eigen::Vector3f pos = Twc.block<3, 1>(0, 3);
    Eigen::Matrix3f rot = Twc.block<3, 3>(0, 0);

    Json::Value json_kf;
    json_kf["id"] = static_cast<Json::Value::UInt64>(pkf->fid_);
    json_kf["img_name"] =
        pkf->img_filename_;  // (std::to_string(getIteration()) + "_" +
                             // std::to_string(pkf->fid_));
    json_kf["width"] = pkf->image_width_;
    json_kf["height"] = pkf->image_height_;

    json_kf["position"][0] = pos.x();
    json_kf["position"][1] = pos.y();
    json_kf["position"][2] = pos.z();

    json_kf["rotation"][0][0] = rot(0, 0);
    json_kf["rotation"][0][1] = rot(0, 1);
    json_kf["rotation"][0][2] = rot(0, 2);
    json_kf["rotation"][1][0] = rot(1, 0);
    json_kf["rotation"][1][1] = rot(1, 1);
    json_kf["rotation"][1][2] = rot(1, 2);
    json_kf["rotation"][2][0] = rot(2, 0);
    json_kf["rotation"][2][1] = rot(2, 1);
    json_kf["rotation"][2][2] = rot(2, 2);

    json_kf["fy"] = graphics_utils::fov2focal(pkf->FoVy_, pkf->image_height_);
    json_kf["fx"] = graphics_utils::fov2focal(pkf->FoVx_, pkf->image_width_);

    json_root[i] = Json::Value(json_kf);
    ++i;
  }

  writer->write(json_root, &out_stream);
}

void GaussianMapper::saveModelParams(std::filesystem::path result_dir) {
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
  std::filesystem::path result_path = result_dir / "cfg_args";
  std::ofstream out_stream;
  out_stream.open(result_path);
  if (!out_stream.is_open())
    throw std::runtime_error("Cannot open file at " + result_path.string());

  out_stream << "Namespace("
             << "eval=" << (model_params_.eval_ ? "True" : "False") << ", "
             << "images=" << "\'" << model_params_.images_ << "\', "
             << "model_path=" << "\'" << model_params_.model_path_.string()
             << "\', "
             << "resolution=" << model_params_.resolution_ << ", "
             << "sh_degree=" << model_params_.sh_degree_ << ", "
             << "source_path=" << "\'" << model_params_.source_path_.string()
             << "\', "
             << "white_background="
             << (model_params_.white_background_ ? "True" : "False") << ", "
             << ")";

  out_stream.close();
}

void GaussianMapper::writeKeyframeUsedTimes(std::filesystem::path result_dir,
                                            std::string name_suffix) {
  if (!record_keyframe_used_times_) return;
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
  std::filesystem::path result_path =
      result_dir / ("keyframe_used_times" + name_suffix + ".txt");
  std::ofstream out_stream;
  out_stream.open(result_path, std::ios::app);
  if (!out_stream.is_open())
    throw std::runtime_error("Cannot open json at " + result_path.string());

  out_stream << "##[Gaussian Mapper]Iteration " << getIteration()
             << " keyframe id, used times, remaining times:\n";
  for (const auto& used_times_it : kfs_used_times_)
    out_stream
        << used_times_it.first << " " << used_times_it.second << " "
        << scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
        << "\n";
  out_stream << "##=========================================" << std::endl;

  out_stream.close();
}

int GaussianMapper::getIteration() {
  std::unique_lock<std::mutex> lock(mutex_status_);
  return iteration_;
}
void GaussianMapper::increaseIteration(const int inc) {
  std::unique_lock<std::mutex> lock(mutex_status_);
  iteration_ += inc;
}

float GaussianMapper::positionLearningRateInit() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.position_lr_init_;
}
float GaussianMapper::featureLearningRate() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.feature_lr_;
}
float GaussianMapper::opacityLearningRate() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.opacity_lr_;
}
float GaussianMapper::scalingLearningRate() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.scaling_lr_;
}
float GaussianMapper::rotationLearningRate() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.rotation_lr_;
}
float GaussianMapper::percentDense() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.percent_dense_;
}
float GaussianMapper::lambdaDssim() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.lambda_dssim_;
}
int GaussianMapper::opacityResetInterval() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.opacity_reset_interval_;
}
float GaussianMapper::densifyGradThreshold() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.densify_grad_threshold_;
}
int GaussianMapper::densifyInterval() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return opt_params_.densification_interval_;
}
int GaussianMapper::newKeyframeTimesOfUse() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return new_keyframe_times_of_use_;
}
int GaussianMapper::stableNumIterExistence() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return stable_num_iter_existence_;
}
bool GaussianMapper::isKeepingTraining() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return keep_training_;
}
bool GaussianMapper::isdoingGausPyramidTraining() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return do_gaus_pyramid_training_;
}
bool GaussianMapper::isdoingInactiveGeoDensify() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  return inactive_geo_densify_;
}
bool GaussianMapper::isSelectorEnabled() const { return selector_enabled_; }

void GaussianMapper::setPositionLearningRateInit(const float lr) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.position_lr_init_ = lr;
}
void GaussianMapper::setFeatureLearningRate(const float lr) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.feature_lr_ = lr;
}
void GaussianMapper::setOpacityLearningRate(const float lr) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.opacity_lr_ = lr;
}
void GaussianMapper::setScalingLearningRate(const float lr) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.scaling_lr_ = lr;
}
void GaussianMapper::setRotationLearningRate(const float lr) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.rotation_lr_ = lr;
}
void GaussianMapper::setPercentDense(const float percent_dense) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.percent_dense_ = percent_dense;
  gaussians_->setPercentDense(percent_dense);
}
void GaussianMapper::setLambdaDssim(const float lambda_dssim) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.lambda_dssim_ = lambda_dssim;
}
void GaussianMapper::setOpacityResetInterval(const int interval) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.opacity_reset_interval_ = interval;
}
void GaussianMapper::setDensifyGradThreshold(const float th) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.densify_grad_threshold_ = th;
}
void GaussianMapper::setDensifyInterval(const int interval) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.densification_interval_ = interval;
}
void GaussianMapper::setNewKeyframeTimesOfUse(const int times) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  new_keyframe_times_of_use_ = times;
}
void GaussianMapper::setStableNumIterExistence(const int niter) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  stable_num_iter_existence_ = niter;
}
void GaussianMapper::setKeepTraining(const bool keep) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  keep_training_ = keep;
}
void GaussianMapper::setDoGausPyramidTraining(const bool gaus_pyramid) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  do_gaus_pyramid_training_ = gaus_pyramid;
}
void GaussianMapper::setDoInactiveGeoDensify(const bool inactive_geo_densify) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  inactive_geo_densify_ = inactive_geo_densify;
}

VariableParameters GaussianMapper::getVaribleParameters() {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  VariableParameters params;
  params.position_lr_init = opt_params_.position_lr_init_;
  params.feature_lr = opt_params_.feature_lr_;
  params.opacity_lr = opt_params_.opacity_lr_;
  params.scaling_lr = opt_params_.scaling_lr_;
  params.rotation_lr = opt_params_.rotation_lr_;
  params.percent_dense = opt_params_.percent_dense_;
  params.lambda_dssim = opt_params_.lambda_dssim_;
  params.opacity_reset_interval = opt_params_.opacity_reset_interval_;
  params.densify_grad_th = opt_params_.densify_grad_threshold_;
  params.densify_interval = opt_params_.densification_interval_;
  params.new_kf_times_of_use = new_keyframe_times_of_use_;
  params.stable_num_iter_existence = stable_num_iter_existence_;
  params.keep_training = keep_training_;
  params.do_gaus_pyramid_training = do_gaus_pyramid_training_;
  params.do_inactive_geo_densify = inactive_geo_densify_;
  return params;
}

void GaussianMapper::setVaribleParameters(const VariableParameters& params) {
  std::unique_lock<std::mutex> lock(mutex_settings_);
  opt_params_.position_lr_init_ = params.position_lr_init;
  opt_params_.feature_lr_ = params.feature_lr;
  opt_params_.opacity_lr_ = params.opacity_lr;
  opt_params_.scaling_lr_ = params.scaling_lr;
  opt_params_.rotation_lr_ = params.rotation_lr;
  opt_params_.percent_dense_ = params.percent_dense;
  gaussians_->setPercentDense(params.percent_dense);
  opt_params_.lambda_dssim_ = params.lambda_dssim;
  opt_params_.opacity_reset_interval_ = params.opacity_reset_interval;
  opt_params_.densify_grad_threshold_ = params.densify_grad_th;
  opt_params_.densification_interval_ = params.densify_interval;
  new_keyframe_times_of_use_ = params.new_kf_times_of_use;
  stable_num_iter_existence_ = params.stable_num_iter_existence;
  keep_training_ = params.keep_training;
  do_gaus_pyramid_training_ = params.do_gaus_pyramid_training;
  inactive_geo_densify_ = params.do_inactive_geo_densify;
}

void GaussianMapper::loadPly(std::filesystem::path ply_path,
                             std::filesystem::path camera_path) {
  this->gaussians_->loadPly(ply_path);
  const auto selector_metadata_path =
      ply_path.parent_path() / "selector_metadata.ply";
  if (std::filesystem::exists(selector_metadata_path))
    this->gaussians_->loadSelectorMetadataPly(selector_metadata_path);

  // Camera
  if (!camera_path.empty() && std::filesystem::exists(camera_path)) {
    cv::FileStorage camera_file(camera_path.string().c_str(),
                                cv::FileStorage::READ);
    if (!camera_file.isOpened())
      throw std::runtime_error(
          "[Gaussian Mapper]Failed to open settings file at: " +
          camera_path.string());

    Camera camera;
    camera.camera_id_ = 0;
    camera.width_ = camera_file["Camera.w"].operator int();
    camera.height_ = camera_file["Camera.h"].operator int();

    std::string camera_type = camera_file["Camera.type"].string();
    if (camera_type == "Pinhole") {
      camera.setModelId(Camera::CameraModelType::PINHOLE);

      float fx = camera_file["Camera.fx"].operator float();
      float fy = camera_file["Camera.fy"].operator float();
      float cx = camera_file["Camera.cx"].operator float();
      float cy = camera_file["Camera.cy"].operator float();

      float k1 = camera_file["Camera.k1"].operator float();
      float k2 = camera_file["Camera.k2"].operator float();
      float p1 = camera_file["Camera.p1"].operator float();
      float p2 = camera_file["Camera.p2"].operator float();
      float k3 = camera_file["Camera.k3"].operator float();

      cv::Mat K =
          (cv::Mat_<float>(3, 3) << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.f);

      camera.params_[0] = fx;
      camera.params_[1] = fy;
      camera.params_[2] = cx;
      camera.params_[3] = cy;

      std::vector<float> dist_coeff = {k1, k2, p1, p2, k3};
      camera.dist_coeff_ = cv::Mat(5, 1, CV_32F, dist_coeff.data());
      camera.initUndistortRectifyMapAndMask(
          K, cv::Size(camera.width_, camera.height_), K, false);

      undistort_mask_[camera.camera_id_] =
          tensor_utils::cvMat2TorchTensor_Float32(camera.undistort_mask,
                                                  device_type_);

      cv::Mat viewer_main_undistort_mask;
      int viewer_image_height_main_ =
          camera.height_ * rendered_image_viewer_scale_main_;
      int viewer_image_width_main_ =
          camera.width_ * rendered_image_viewer_scale_main_;
      cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                 cv::Size(viewer_image_width_main_, viewer_image_height_main_));
      viewer_main_undistort_mask_[camera.camera_id_] =
          tensor_utils::cvMat2TorchTensor_Float32(viewer_main_undistort_mask,
                                                  device_type_);

    } else {
      throw std::runtime_error("[Gaussian Mapper]Unsupported camera model: " +
                               camera_path.string());
    }

    if (!viewer_camera_id_set_) {
      viewer_camera_id_ = camera.camera_id_;
      viewer_camera_id_set_ = true;
    }
    this->scene_->addCamera(camera);
  }

  // Ready
  this->initial_mapped_ = true;
  increaseIteration();
}

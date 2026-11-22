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
  elastic_budget_manager_.profiler = gaussians_->protection_.profiler;
  elastic_budget_manager_.performance_logging = protection_config_.performance_logging;
  scene_ = std::make_shared<GaussianScene>(model_params_);
  recent_window_importance_ =
      std::make_unique<RecentWindowImportanceManager>(
          device_type_, recent_window_config_);
  if (global_gi_config_.streaming() && pSLAM_)
    global_gi_ = std::make_unique<GlobalGIManager>(device_type_, global_gi_config_);
  historical_outlier_selector_ =
      std::make_unique<HistoricalOutlierSelector>(historical_outlier_config_);
  conflict_replay_manager_ = std::make_unique<ConflictReplayManager>(conflict_replay_config_);
  gaussians_->setOnlineGIStateCallbacks(
      [this](std::int64_t count) {
        if (recent_window_importance_) recent_window_importance_->appendGaussians(count);
        ++topology_version_;
      },
      [this](const torch::Tensor& survivor_mask) {
        ++correspondence_version_;
        if (recent_window_importance_) recent_window_importance_->pruneGaussians(survivor_mask);
        if (global_gi_) global_gi_->pruneGaussians(survivor_mask);
        ++topology_version_;
      },
      [this](std::int64_t count) {
        if (recent_window_importance_) recent_window_importance_->resetForGaussianCount(count);
        if (global_gi_) global_gi_->resetForGaussianCount(count);
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
        global_gi_->assertAligned(gaussians_->getXYZ());
      });
  if (selector_enabled_) {
    selector_network_ = improvements::selector::GumbelNetwork();
    selector_network_->to(torch::Device(device_type_));
    torch::optim::AdamOptions selector_adam_options(selector_learning_rate_);
    selector_optimizer_.reset(new torch::optim::Adam(
        selector_network_->parameters(), selector_adam_options));
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

  int ultra_light_enabled = 0;
  readInt("Training.ultra_light_enabled", ultra_light_enabled);
  readInt("Training.ultra_light_interval", ultra_light_legacy_interval_);
  ultrafast_rotation_.enabled = ultra_light_enabled != 0;
  std::cout << "[UltrafastRotation CONFIG] ultra_light=" << ultrafast_rotation_.enabled
            << " legacy_interval_ignored=" << ultra_light_legacy_interval_ << std::endl;

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
  const cv::FileNode ratios_node = settings_file["Selector.target_ratios"];
  if (!ratios_node.empty() && ratios_node.isSeq()) {
    std::vector<float> configured_ratios;
    for (auto it = ratios_node.begin(); it != ratios_node.end(); ++it) {
      const float ratio = (*it).operator float();
      if (ratio > 0.0f && ratio <= 1.0f) configured_ratios.push_back(ratio);
    }
    if (!configured_ratios.empty()) selector_target_ratios_ = configured_ratios;
  }
  if (ultrafast_rotation_.enabled) {
    selector_target_ratios_.assign(UltrafastRotation::ratios.begin(), UltrafastRotation::ratios.end());
    std::cout << "[UltrafastRotation CONFIG] ratios=0.01,0.05,0.10,0.15,Full"
              << " stop_at_online_end=1 topology_full_only=1 tail_disabled=1\n";
  }
  int protection_enabled = selector_protection_enabled_ ? 1 : 0;
  readInt("Selector.protection_enabled", protection_enabled);
  selector_protection_enabled_ = protection_enabled != 0;
  readInt("Protection.age_ready", protection_config_.age_ready);
  readInt("Protection.optimization_ready", protection_config_.optimization_ready);
  readInt("Protection.unique_keyframe_ready", protection_config_.unique_keyframe_ready);
  readInt("Protection.stale_age", protection_config_.stale_age);
  readFloat("Protection.max_budget_fraction", protection_config_.max_budget_fraction);
  int performance_logging = protection_config_.performance_logging ? 1 : 0;
  readInt("Protection.performance_logging", performance_logging);
  protection_config_.performance_logging = performance_logging != 0;
  int weighted_bernoulli = 1, inherit_support = 1;
  readInt("Protection.weighted_bernoulli", weighted_bernoulli);
  readInt("Protection.clone_split_inherit_keyframe_support", inherit_support);
  if (!weighted_bernoulli || !inherit_support)
    throw std::invalid_argument("Evidence-aware Protection requires weighted Bernoulli and clone/split KF inheritance");
  protection_config_.validate();
  std::cout << "[Protection CONFIG] age_ready=" << protection_config_.age_ready
            << " optimization_ready=" << protection_config_.optimization_ready
            << " unique_keyframe_ready=" << protection_config_.unique_keyframe_ready
            << " stale_age=" << protection_config_.stale_age
            << " max_budget_fraction=" << protection_config_.max_budget_fraction << std::endl;
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
  int detach_gaussian_parameters =
      selector_detach_gaussian_parameters_ ? 1 : 0;
  readInt("Selector.detach_gaussian_parameters", detach_gaussian_parameters);
  selector_detach_gaussian_parameters_ = detach_gaussian_parameters != 0;
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
  int mature_only = transform_field_config_.mature_only ? 1 : 0;
  readInt("TransformField.mature_only", mature_only);
  transform_field_config_.mature_only = mature_only != 0;

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
  // Existing configs retain RecentWindow behavior. Explicit mode controls only GI,
  // while temporal membership remains available to Replay and GradientGuard.
  global_gi_config_.mode = recent_window_config_.enabled ? "recent" : "disabled";
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
            << " verbose_logging=" << global_gi_config_.verbose_logging << std::endl;
  int ratio_aware = importance_ratio_aware_weight_ ? 1 : 0;
  readInt("Importance.ratio_aware_weight", ratio_aware);
  importance_ratio_aware_weight_ = ratio_aware != 0;
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
         !(normalMappingLimit() > 0 && mapping_iter_ >= normalMappingLimit()) &&
         (!ultrafast_rotation_.enabled || opt_params_.iterations_ == -1 ||
          getIteration() + 1 < opt_params_.iterations_)) {
    if (ultrafast_rotation_.enabled && pSLAM_->isShutDown()) {
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
  if (ultrafast_rotation_.enabled) {
    if (pSLAM_->isShutDown()) markUltrafastOnlineEnd();
    reportUltrafastRotation("training_end");
    mapping_profiler_.report(mapping_iter_);
  } else if (tail_optimization_schedule_.enabled) {
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
  } else if (selector_max_mapping_iterations_ > 0) {
    std::cout << "[Gaussian Mapper]Forced mapping tail: mapping_iter="
              << mapping_iter_ << ", target="
              << selector_max_mapping_iterations_ << ", keyframes="
              << scene_->keyframes().size() << std::endl;
    while (!isStopped() && !scene_->keyframes().empty() &&
           mapping_iter_ < selector_max_mapping_iterations_) {
      trainForOneIteration();
    }
  } else {
    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (getIteration() - SLAM_stop_iter < n_delay_iters ||
           getIteration() % densify_interval < n_delay_iters ||
           isKeepingTraining()) {
      trainForOneIteration();
    }
  }

  // Save and clear
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

  signalStop();
}

void GaussianMapper::trainColmap() {
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
  // Reject new normal training before KF selection, Replay, or any backward.
  if (ultrafast_rotation_.enabled &&
      (!ultrafast_rotation_.allowsNormalStep(
          pSLAM_ && !pSLAM_->isShutDown(), mapping_iter_, normalMappingLimit()) ||
       (opt_params_.iterations_ != -1 && getIteration() + 1 >= opt_params_.iterations_))) {
    if (pSLAM_ && pSLAM_->isShutDown()) markUltrafastOnlineEnd();
    mapping_profiler_.cancel();
    global_gi_cycle_start_.reset();
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

  // Snapshot shutdown before the first post-SLAM mapping. increaseIteration()
  // above belongs to the upcoming mapping and is excluded from this snapshot.
  const bool slam_online = pSLAM_ && !pSLAM_->isShutDown();
  if (ultrafast_rotation_.enabled && !slam_online) {
    increaseIteration(-1);
    markUltrafastOnlineEnd();
    mapping_profiler_.cancel();
    global_gi_cycle_start_.reset();
    return;
  }
  if (!ultrafast_rotation_.enabled && pSLAM_ && !slam_online && tail_optimization_schedule_.enabled &&
      !tail_optimization_schedule_.active) {
    beginForcedTail(getIteration() - 1);
  }
  if (tail_optimization_schedule_.active &&
      tail_optimization_schedule_.complete(mapping_iter_)) {
    increaseIteration(-1);
    mapping_profiler_.cancel();
    global_gi_cycle_start_.reset();
    return;
  }
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
  if (shouldRunSelectorReplay()) replay_task_.pending = true;
  if (replay_task_.take(enhanced) && shouldRunSelectorReplay(true)) {
    processConflictReplay(viewpoint_cam);
  }

  if (ultrafast_rotation_.enabled && pSLAM_->isShutDown()) {
    increaseIteration(-1);
    markUltrafastOnlineEnd();
    mapping_profiler_.cancel();
    global_gi_cycle_start_.reset();
    return;
  }

  trainMappingIteration(viewpoint_cam, enhanced, false);
  ++mapping_iter_;
  if (ultrafast_rotation_.enabled && pSLAM_->isShutDown()) markUltrafastOnlineEnd();
  if (ultrafast_rotation_.enabled && mapping_iter_ % 1000 == 0)
    reportUltrafastRotation("mapping");
  if (global_gi_) {
    gi_refresh_task_.observe(mapping_iter_, global_gi_config_.refresh_interval);
    if (gi_refresh_task_.take(enhanced)) {
      auto timing = mapping_profiler_.scope(MappingProfiler::GI);
      std::unique_lock<std::mutex> lock_render(mutex_render_);
      global_gi_->refreshIfDue(mapping_iter_, [this](uint64_t id) {
        return renderGlobalGIContribution(id);
      }, true);
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
      const auto mature = selector_protection_enabled_
          ? gaussians_->getSelectorMatureMask()
          : torch::ones({gaussians_->getXYZ().size(0)}, gaussians_->getXYZ().options().dtype(torch::kBool));
      global_gi_->recordGuidance(mature, selector_enabled_ && selector_gi_loss_weight_ != 0.0f &&
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
  auto iter_start_timing = std::chrono::steady_clock::now();
  const bool rotation = ultrafast_rotation_.enabled;
  const bool full_step = !rotation || ultrafast_rotation_.full();
  const auto render_total_gaussians = gaussians_->getXYZ().size(0);
  const auto phase = ultrafast_rotation_.phase();
  if (rotation) {
    if (!ultrafast_training_timer_)
      ultrafast_training_timer_ = std::make_unique<UltrafastTrainingTimer>(device_type_ == torch::kCUDA);
    ultrafast_training_timer_->begin();
  }
  auto* rotation_timer = rotation ? ultrafast_training_timer_.get() : nullptr;
  bool densify_stats_updated = false, densify_triggered = false;
  bool selector_committed = false, tf_committed = false;
  LodGradientResult lod;

  writeKeyframeUsedTimes(result_dir_ / "used_times");

  // if (isdoingInactiveGeoDensify() &&
  // !viewpoint_cam->done_inactive_geo_densify_)
  //     increasePcdByKeyframeInactiveGeoDensify(viewpoint_cam);

  int training_level = num_gaus_pyramid_sub_levels_;
  int image_height, image_width;
  torch::Tensor gt_image, mask;
  if (isdoingGausPyramidTraining())
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
  if (rotation) gaussians_->optimizer_->zero_grad(true);
  if (!enhanced || (rotation && full_step)) {
    if (selector_optimizer_) selector_optimizer_->zero_grad(true);
    if (transform_optimizer_) transform_optimizer_->zero_grad(true);
  }

  if (protection_config_.performance_logging) gaussians_->protection_.profiler->beginIteration();
  gaussians_->protection_.advance(static_cast<int64_t>(mapping_iter_ + 1));

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
  RenderPackage render_pkg;
  torch::Tensor Ll1, full_loss;
  if (full_step) {
    {
      auto timing = mapping_profiler_.scope(MappingProfiler::FullRender);
      if (rotation_timer) rotation_timer->beginRender();
      render_pkg = GaussianRenderer::render(
          viewpoint_cam, image_height, image_width, gaussians_, pipe_params_,
          background_, override_color_, 1.0f, false, torch::Tensor(), false,
          collect_importance);
      if (rotation_timer) rotation_timer->endRender();
    }
    if (collect_importance && recent_window_importance_->enabled() &&
        render_pkg.frame_importance.defined() &&
        render_pkg.contribution_count.defined() &&
        render_pkg.frame_importance.numel() > 0) {
      auto timing = mapping_profiler_.scope(MappingProfiler::GI);
      refreshRecentWindowContribution(viewpoint_cam,
                                      render_pkg.frame_importance,
                                      render_pkg.contribution_count);
    }
    gaussians_->protection_.beginFullMapping(render_pkg.visibility_indices);
    auto l1_loss = opt_params_.smooth_l1_ ? loss_utils::smooth_l1_loss : loss_utils::l1_loss;
    Ll1 = l1_loss(render_pkg.image, gt_image, 1.0f);
    auto Lssim = loss_utils::fast_ssim(render_pkg.image, gt_image);
    const float lambda_dssim = lambdaDssim();
    full_loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - Lssim);
    if (opt_params_.opacity_reg_)
      full_loss += opt_params_.opacity_reg_ * gaussians_->getOpacityActivation().abs().mean();
    auto timing = mapping_profiler_.scope(MappingProfiler::FullBackward);
    full_loss.backward();
  }
  bool has_selector_update = false;
  if (enhanced && (!rotation || !full_step)) {
    auto timing = mapping_profiler_.scope(MappingProfiler::LOD);
    lod = accumulateLodGradients(
        viewpoint_cam, image_height, image_width, gt_image,
        rotation ? phase : (selector_target_ratios_.empty() ? 0 : selector_step_ % selector_target_ratios_.size()),
        rotation ? false : selector_detach_gaussian_parameters_, true, rotation_timer);
    has_selector_update = lod.trained;
  }

  {
    torch::NoGradGuard no_grad;
    bool gaussian_committed = false;
    auto commit_optimizer = [&] {
      if (!hasNormalOptimizationBudget()) return;
      auto timing = mapping_profiler_.scope(MappingProfiler::Optimizer);
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
      if (enhanced) gaussian_committed = stepWithHistoricalGuard(viewpoint_cam);
      else {
        gaussians_->optimizer_->step();
        normal_update_clock_.record(true, false);
        gaussian_committed = true;
      }
      gaussian_committed &= has_gaussian_gradient;
      selector_committed = selector_had_gradient &&
          normal_update_clock_.rejected_updates == rejected_before;
      tf_committed = normal_update_clock_.tf_updates > tf_before;
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
      densify_stats_updated = true;

      if (rotation ? ultrafast_densify_task_.take(true) :
          ((getIteration() > opt_params_.densify_from_iter_) &&
           (getIteration() % densifyInterval() == 0))) {
        int size_threshold = (getIteration() < prune_big_point_after_iter_ ||
                              prune_big_point_after_iter_ == -1)
                                 ? 0
                                 : 20;
        gaussians_->densifyAndPrune(densifyGradThreshold(),
                                    densify_min_opacity_,  // 0.005,//
                                    scene_->cameras_extent_, size_threshold);
        densify_triggered = true;
        if (recent_window_importance_ && recent_window_importance_->enabled())
          recent_window_importance_->assertAligned(gaussians_->getXYZ());
        if (global_gi_) global_gi_->assertAligned(gaussians_->getXYZ());
      }

      if (rotation ? ultrafast_opacity_task_.take(true) : (opacityResetInterval() &&
          (getIteration() % opacityResetInterval() == 0 ||
           (model_params_.white_background_ &&
            getIteration() == opt_params_.densify_from_iter_))))
        gaussians_->resetOpacity();
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

    // Commit evidence after topology remaps the pending Full visibility indices.
    if (!rotation) commit_optimizer();
    if (full_step) gaussians_->protection_.finishFullMapping(
        gaussian_committed, static_cast<int64_t>(viewpoint_cam->fid_));
    if (protection_config_.performance_logging) gaussians_->protection_.profiler->endIteration();
    if (rotation) {
      rotation_timer->end();
      std::ostringstream line;
      line << std::fixed << std::setprecision(4) << "[UltrafastRotation] normal_training_step="
           << ultrafast_rotation_.step << " mapping_iteration=" << mapping_iter_ + 1
           << " phase=" << UltrafastRotation::names[phase]
           << " ratio=" << ultrafast_rotation_.ratio() << " keyframe_id=" << viewpoint_cam->fid_
           << " is_lod_step=" << !full_step << " is_full_step=" << full_step
           << " gaussian_optimizer_step=" << gaussian_committed
           << " selector_optimizer_step=" << selector_committed << " tf_optimizer_step=" << tf_committed
           << " kfs_loss_updated=" << full_step << " densify_stats_updated=" << densify_stats_updated
           << " densify_triggered=" << densify_triggered << " prune_triggered=" << densify_triggered
           << " active_gaussians=" << (full_step ? render_total_gaussians : lod.active_gaussians)
           << " tf_transformed_gaussians=" << lod.transformed_gaussians
           << " total_gaussians=" << render_total_gaussians
           << " post_topology_gaussians=" << gaussians_->getXYZ().size(0)
           << " loss=" << (full_step ? full_loss.item<double>() : lod.loss)
           << " training_ms=" << rotation_timer->wall_ms << " render_ms=" << rotation_timer->render_ms
           << " cuda_elapsed_ms=" << rotation_timer->cuda_elapsed_ms;
      std::cout << line.str() << '\n';
      ultrafast_rotation_.complete(rotation_timer->wall_ms, rotation_timer->render_ms,
                                  full_step || lod.rendered, selector_committed, tf_committed);
      if (all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0) {
        renderAndRecordAllKeyframes();
        savePly(result_dir_ / std::to_string(getIteration()) / "ply");
      }
    }
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
  TORCH_CHECK(r.balanced() && r.tailSteps() == 0 &&
              r.full_render_count + r.lod_render_count == r.step,
              "Ultrafast rotation invariant failed: balance/tail/render count");
  const char* keys[] = {"lod_001", "lod_005", "lod_010", "lod_015", "full"};
  std::ostringstream line;
  line << std::fixed << std::setprecision(4) << "[UltrafastRotation SUMMARY] event=" << event;
  for (std::size_t i = 0; i < r.counts.size(); ++i)
    line << " ultrafast_" << keys[i] << "_steps=" << r.counts[i]
         << ' ' << keys[i] << "_mean_ms=" << (r.counts[i] ? r.elapsed_sum[i] / r.counts[i] : 0.0)
         << ' ' << keys[i] << "_render_mean_ms=" << (r.counts[i] ? r.render_sum[i] / r.counts[i] : 0.0);
  line << " ultrafast_total_normal_steps=" << r.step << " balanced=" << r.balanced()
       << " mean_mapping_training_ms=" << r.mean() << " p95_mapping_training_ms=" << r.p95()
       << " full_render_count=" << r.full_render_count << " lod_render_count=" << r.lod_render_count
       << " online_ended=" << r.online_ended
       << " online_end_mapping_iteration=" << r.online_end_mapping_iteration
       << " online_end_rotation_step=" << r.online_end_rotation_step
       << " tail_training_steps=" << r.tailSteps()
       << " selector_online_lod_update_count=" << r.selector_updates
       << " tf_online_lod_update_count=" << r.tf_updates
       << " tf_first_initialized_mapping_iteration=" << ultrafast_tf_first_initialized_
       << " tf_first_applied_lod_iteration=" << ultrafast_tf_first_applied_
       << " timing=" << (device_type_ == torch::kCUDA ? "synchronized_wall_and_cuda_event_render" : "cpu_wall");
  std::cout << line.str() << '\n';
}

void GaussianMapper::beginForcedTail(std::uint64_t iteration) {
  if (!tail_optimization_schedule_.begin(iteration, mapping_iter_)) return;
  std::cout << "[TailOptimization] event=begin slam_stop_iter=" << iteration
            << " start_mapping=" << mapping_iter_
            << " target=" << tail_optimization_schedule_.target_steps
            << " ignored_max_mapping_iterations=" << selector_max_mapping_iterations_ << '\n';
}

std::uint64_t GaussianMapper::normalMappingLimit() const {
  if (ultrafast_rotation_.enabled) return selector_max_mapping_iterations_;
  return tail_optimization_schedule_.mappingLimit(selector_max_mapping_iterations_, pSLAM_ != nullptr);
}

bool GaussianMapper::hasNormalOptimizationBudget() {
  if (ultrafast_rotation_.enabled)
    return opt_params_.iterations_ == -1 || getIteration() < opt_params_.iterations_;
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

void GaussianMapper::initializeTransformFieldIfNeeded() {
  if (transform_field_ || !transform_field_config_.enabled ||
      !selector_network_ ||
      mapping_iter_ < static_cast<std::uint64_t>(
                           std::max(0, transform_field_config_.start_iter)))
    return;
  const auto xyz = gaussians_->getXYZ().detach();
  if (xyz.numel() == 0) return;

  torch::NoGradGuard no_grad;
  const auto mature_mask = selector_protection_enabled_
                               ? gaussians_->getSelectorMatureMask()
                               : torch::ones(
                                      {xyz.size(0)},
                                      torch::TensorOptions()
                                          .dtype(torch::kBool)
                                          .device(xyz.device()));
  const auto mature_xyz = xyz.index({mature_mask});
  if (mature_xyz.numel() == 0) return;
  auto aabb_min = std::get<0>(mature_xyz.min(/*dim=*/0));
  auto aabb_max = std::get<0>(mature_xyz.max(/*dim=*/0));
  auto center = (aabb_min + aabb_max) * 0.5;
  auto half_extent = (aabb_max - aabb_min) *
                     (0.5 * std::max(transform_field_config_.aabb_scale,
                                     1.0f));
  half_extent = torch::clamp_min(half_extent, 1e-3);
  aabb_min = (center - half_extent).detach();
  aabb_max = (center + half_extent).detach();

  transform_field_ = TransformField(aabb_min, aabb_max,
                                    transform_field_config_);
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

bool GaussianMapper::stepWithHistoricalGuard(const std::shared_ptr<GaussianKeyframe>& current) {
  const auto cuda_tf_groups = transform_field_ && transform_field_->config().backend == "cuda" &&
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
  if (!gradient_constraint_config_.enabled) {
    tf_cuda::tfCudaAdamMixedStep(*gaussians_->optimizer_, cuda_tf_groups);
    normal_update_clock_.record(true, tf_had_gradient); return true;
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
    normal_update_clock_.record(true, tf_had_gradient);
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
  normal_update_clock_.record(report.committed, tf_updated);
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
    torch::Tensor mature, mature_indices, uniform;
    std::map<float, ElasticBudgetPlan> budgets;
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
  state->mature = selector_protection_enabled_
      ? gaussians_->getSelectorMatureMask()
      : torch::ones({gaussians_->xyz_.size(0)}, gaussians_->xyz_.options().dtype(torch::kBool));
  state->mature_indices = torch::nonzero(state->mature).reshape({-1});
  elastic_budget_manager_.useUniform(gaussians_->protection_.cached_uniform);
  const auto population = elastic_budget_manager_.population(state->mature,
      gaussians_->protection_.classify().need, state->mature_indices.numel());
  state->uniform = torch::rand({gaussians_->xyz_.size(0)}, gaussians_->xyz_.options().requires_grad(false));
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
    if (!objective.full && !state->budgets.count(objective.ratio))
      state->budgets.emplace(objective.ratio,
          elastic_budget_manager_.plan(population, objective.ratio, protection_config_));
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
        input = historicalLoDInput(full, state->mature, state->mature_indices, state->budgets.at(view.ratio),
            state->uniform, selector, field, view.ratio, selector_temperature_);
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
  }
  if (!mature_mask.any().item<bool>()) return capture;
  capture = change_coordinator_->captureBeforeGeometryChange(
      flag, mature_mask, {xyz, gaussians_->scaling_.detach(),
                         gaussians_->getRotationActivation().detach()},
      old_aabb, topology_version_, correspondence_version_, mapping_iter_);
  if (change_distill_config_.enabled && !field_only && selector_network_ &&
      change_distill_config_.distill_selector_on_geometry_change)
    capture.teacher_selector = improvements::selector::GumbelNetwork(
        selector_network_->cloneFrozen());
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
  // KF pose changes also invalidate GI, even in an empty CD event.
  if (geometry_operation) {
    if (recent_window_importance_)
      recent_window_importance_->invalidateContributions();
  }
  if (capture.snapshot_count == 0) {
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
  context.selector = selector_network_;
  context.tf = transform_field_;
  context.normal_selector_optimizer = selector_optimizer_.get();
  context.normal_tf_optimizer = transform_optimizer_.get();
  context.shared_mapping_optimizer = gaussians_->optimizer_.get();
  context.ratios = selector_target_ratios_;
  context.temperature = selector_temperature_;
  auto report = change_coordinator_->finalizeGeometryChange(
      std::move(capture), context, correspondence_version_, prepared_plan);
  cd_selector_steps_ += report.selector_steps;
  cd_tf_steps_ += report.tf_steps;
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
    const auto mature_mask = selector_protection_enabled_
                                 ? gaussians_->getSelectorMatureMask()
                                 : torch::ones(
                                       {xyz.size(0)},
                                       torch::TensorOptions()
                                           .dtype(torch::kBool)
                                           .device(xyz.device()));
    mature_xyz = xyz.index({mature_mask});
    current_aabb.min_xyz = transform_field_->aabbMin().detach();
    current_aabb.max_xyz = transform_field_->aabbMax().detach();
    pressure = dynamic_aabb_manager_->evaluate(mature_xyz, current_aabb);
  }
  std::ostringstream check_log;
  check_log << std::fixed << std::setprecision(4)
            << "[Gaussian Mapper]Dynamic AABB check: mapping_iter="
            << mapping_iter_ << ", mature=" << pressure.mature_count
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
  elastic_budget_manager_.useUniform(gaussians_->protection_.cached_uniform);
  const auto budget_population = elastic_budget_manager_.population(
      mature_mask, gaussians_->protection_.classify().need, mature_indices.numel());
  std::map<float, ElasticBudgetPlan> quality_budgets;
  for (const float ratio : selector_target_ratios_)
    quality_budgets.emplace(ratio, elastic_budget_manager_.plan(
        budget_population, ratio, protection_config_));
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
  {
    ScopedTorchRNG rng(xyz.device(), drift_stability_config_.seed ^ mapping_iter_);
    quality_uniform = torch::rand({mature_indices.numel()}, xyz.options());
  }
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
      const auto& budget = quality_budgets.at(ratio);
      const RenderKey key{selector_id, field_id, ratio};
      auto input = render_inputs.find(key);
      if (input == render_inputs.end()) {
        auto selected = selector_inputs.find({selector_id, ratio});
        if (selected == selector_inputs.end()) {
          auto probability = probe_cache.probability(selector, ratio, mature_indices);
          // The distribution of Gumbel hard argmax does not depend on tau.
          // Function drift still uses its configured deterministic temperature.
          if (selector && selector_temperature_ != 1.0) {
            const auto attrs = current.select(mature_indices);
            probability = selector.ptr()->forwardProbability(
                attrs.xyz, attrs.rotation, attrs.log_scaling.exp(), ratio, 1.0);
          }
          const auto keep = quality_uniform < probability;
          const auto mask = budget.admitted.to(torch::kFloat32)
                                .index_copy(0, mature_indices, keep.to(torch::kFloat32));
          const auto active = keep.sum();
          selected = selector_inputs.emplace(ModuleKey{selector_id, ratio},
              std::make_pair(mask.unsqueeze(1) * probe_opacity,
                  torch::stack({active.to(torch::kDouble), probability.sum().to(torch::kDouble)}))).first;
        }
        auto elastic = field_inputs.find({field_id, ratio});
        if (elastic == field_inputs.end()) {
          ElasticGaussianAttributes attrs{xyz, probe_scaling, probe_rotation};
          if (field) {
            auto transform_mask = field.ptr()->config().mature_only ? mature_mask : torch::ones_like(mature_mask);
            if (field.ptr()->config().outside_identity) transform_mask &= field.ptr()->insideMask(xyz);
            const auto indices = torch::nonzero(transform_mask).reshape({-1});
            if (indices.numel() > 0) {
              const auto state = probe_cache.transformed(field, ratio, indices);
              attrs.xyz = xyz.index_copy(0, indices, state.xyz);
              attrs.scaling = probe_scaling.index_copy(0, indices, state.log_scaling.exp());
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
          metrics[0].item<double>(), budget.protected_count,
          static_cast<int64_t>(metrics[1].item<double>()),
          budget.target.total,
          metrics[2].item<double>() + budget.protected_count,
          budget.target.mature, metrics[2].item<double>()};
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
  AdamStepPreview preview(optimizer, transform_field_ && transform_field_->config().backend == "cuda" &&
      shared_transform_optimizer_configured_
      ? std::vector<std::size_t>{shared_transform_mlp_group_index_, shared_transform_grid_group_index_}
      : std::vector<std::size_t>{});
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
    UltrafastTrainingTimer* rotation_timer) {
  LodGradientResult result;
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

  torch::Tensor mature_mask, mature_indices;
  {
    auto timing = mapping_profiler_.scope(MappingProfiler::MaturePreparation);
    mature_mask = selector_protection_enabled_
        ? gaussians_->getSelectorMatureMask()
        : torch::ones({num_gaussians}, gaussians_->getXYZ().options().dtype(torch::kBool));
    mature_indices = torch::nonzero(mature_mask).reshape({-1});
  }
  const int64_t mature_count = mature_indices.numel();
  const bool log_budget = record_supervision && (getIteration() % 1000 == 0 ||
      (training_report_interval_ > 0 && getIteration() % training_report_interval_ == 0));
  elastic_budget_manager_.useUniform(gaussians_->protection_.cached_uniform);
  const auto budget_population = elastic_budget_manager_.population(
      mature_mask, gaussians_->protection_.classify().need, mature_count);
  ratio_index %= selector_target_ratios_.size();
  const float target_ratio = selector_target_ratios_[ratio_index];
  const auto budget = elastic_budget_manager_.plan(
      budget_population, target_ratio, protection_config_);

  initializeTransformFieldIfNeeded();
  ensureSharedMappingOptimizer();
  selector_optimizer_->zero_grad();
  if (transform_optimizer_) {
    updateTransformLearningRate();
    transform_optimizer_->zero_grad();
  }

  auto xyz = detach_gaussian_parameters ? gaussians_->getXYZ().detach()
                                        : gaussians_->getXYZ();
  auto scaling = detach_gaussian_parameters
                     ? gaussians_->getScalingActivation().detach()
                     : gaussians_->getScalingActivation();
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
  const bool selector_all = mature_count == num_gaussians;
  torch::Tensor keep_probability;
  auto hard_selected = mature_count > 0 ? selector_network_->forwardHard(
      selector_all ? xyz : xyz.index_select(0, mature_indices),
      selector_all ? rotation : rotation.index_select(0, mature_indices),
      selector_all ? scaling : scaling.index_select(0, mature_indices),
      target_ratio, selector_temperature_, &keep_probability)
      : torch::empty({0}, xyz.options());
  // Scatter preserves the mature straight-through graph. In mature-only mode
  // Immature rows have no Selector supervision; admission is handled separately.
  auto selector_hard_mask = selector_all ? hard_selected
      : torch::zeros({num_gaussians}, xyz.options()).index_copy(0, mature_indices, hard_selected);
  const auto expected_mature = mature_count == 0 ? torch::zeros({}, xyz.options())
      : (selector_all ? keep_probability.index_select(0, mature_indices) : keep_probability).sum();

  const auto protected_mask = ~mature_mask;
  GITeacher gi_teacher;
  {
    auto timing = mapping_profiler_.scope(MappingProfiler::GITeacher);
  if (global_gi_) {
    gi_teacher = gi_teacher_builder_.build(global_gi_->importance(), mature_mask,
        protected_mask, budget.mature_ratio, global_gi_->version(),
        global_gi_->topologyVersion(), mature_count, global_gi_->validMask());
  } else if (recent_window_importance_ && recent_window_config_.collect_importance &&
             recent_window_importance_->enabled()) {
    auto combined_gi = recent_window_importance_->getImportance();
    if (recent_window_importance_->hasValidContribution() &&
        combined_gi.defined() && combined_gi.size(0) == num_gaussians)
      gi_teacher = gi_teacher_builder_.build(
          combined_gi, mature_mask, protected_mask, budget.mature_ratio,
          recent_window_importance_->version(),
          recent_window_importance_->topologyVersion(), mature_count,
          recent_window_importance_->observedMask());
  }
  }

  const auto protected_f = budget.admitted.to(torch::kFloat32);
  auto selector_mask =
      protected_f + (1.0f - protected_f) * selector_hard_mask;
  auto ratio_loss = mature_count == 0 ? torch::zeros({}, xyz.options())
      : torch::abs(expected_mature - budget.target.mature) / std::max(1.0, budget.target.total);

  auto l1_loss =
      opt_params_.smooth_l1_ ? loss_utils::smooth_l1_loss : loss_utils::l1_loss;
  torch::Tensor gi_loss;
  if (gi_teacher.valid() && gi_teacher.supervised_k > 0 &&
      selector_gi_loss_weight_ != 0.0f) {
    auto timing = mapping_profiler_.scope(MappingProfiler::GILoss);
    auto selector_supervised =
        selector_hard_mask.index({gi_teacher.supervised_mask});
    auto teacher_supervised =
        gi_teacher.label.index({gi_teacher.supervised_mask});
    gi_loss = l1_loss(selector_supervised, teacher_supervised, 1.0f);
  }

  // This map belongs only to this LOD forward/backward. Master rows and all
  // global Selector/GI/ratio losses keep their original population.
  const auto compact_to_global =
      torch::nonzero(selector_mask.detach() > 0.0f).reshape({-1});
  const int64_t compact_count = compact_to_global.numel();
  result.active_gaussians = compact_count;
  GaussianRenderInput full_input{
      xyz, opacity, scaling, rotation, features_dc, features_rest,
      gaussians_->active_sh_degree_, gaussians_->max_sh_degree_};
  auto selected_input = full_input.indexSelect(compact_to_global);
  const auto compact_mask = selector_mask.index_select(0, compact_to_global);
  // Do not replace selected masks with ones: this is the Selector's
  // straight-through photometric gradient, even though forward values are 1.
  selected_input.opacity = compact_mask.unsqueeze(1) * selected_input.opacity;

  int64_t transform_count = 0;
  if (transform_field_ && compact_count > 0) {
    auto transform_mask = selector_mask.detach()
                              .index_select(0, compact_to_global) > 0.0f;
    if (transform_field_config_.mature_only)
      transform_mask = torch::logical_and(
          transform_mask, mature_mask.index_select(0, compact_to_global));
    if (transform_field_config_.outside_identity)
      transform_mask = torch::logical_and(
          transform_mask, transform_field_->insideMask(selected_input.xyz));
    const auto tf_local_indices = torch::nonzero(transform_mask).reshape({-1});
    transform_count = tf_local_indices.numel();
    if (transform_count > 0) {
      if (rotation_timer && ultrafast_tf_first_applied_ < 0) {
        ultrafast_tf_first_applied_ = mapping_iter_ + 1;
        std::cout << "[UltrafastRotation TF] event=first_applied_lod mapping_iteration="
                  << mapping_iter_ + 1 << " normal_training_step=" << ultrafast_rotation_.step << '\n';
      }
      auto transformed = transform_field_->applyResidual(
          selected_input.xyz.index_select(0, tf_local_indices),
          selected_input.scaling.index_select(0, tf_local_indices),
          selected_input.rotation.index_select(0, tf_local_indices),
          target_ratio);
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
  if (compact_count > 0 || rotation_timer) {
    const bool slam_online = pSLAM_ && !pSLAM_->isShutDown();
    const bool post_slam = pSLAM_ && !slam_online;
    const float lod_render_scale = slam_online ? selector_lod_render_scale_ : 1.0f;
    // Tail optimization uses native images even if a KF still has pyramid uses.
    const int source_height = post_slam ? viewpoint_cam->image_height_ : image_height;
    const int source_width = post_slam ? viewpoint_cam->image_width_ : image_width;
    const int lod_height = std::max(
        1, static_cast<int>(source_height * lod_render_scale));
    const int lod_width = std::max(
        1, static_cast<int>(source_width * lod_render_scale));
    auto lod_gt_image = post_slam
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
  }
  if (transform_field_ && mature_count > 0)
    selector_loss += transform_field_->regularizationLoss();

  if (!record_supervision || rotation_timer) {
    result.loss = selector_loss.detach().item<double>();
    result.finite_loss = std::isfinite(result.loss);
    if (!result.finite_loss) return result;
  }
  if (selector_loss.requires_grad()) {
    auto timing = mapping_profiler_.scope(MappingProfiler::LodBackward);
    selector_loss.backward();
  }
  if (log_budget) {
    std::cout << "[LOD Compact] mapping_iter=" << gaussians_->protection_.mapping_iteration
              << " ratio=" << target_ratio << " active_gaussians=" << compact_count
              << " protection_demand=" << budget.target.protection_demand
              << " protection_budget_target=" << budget.target.protection
              << " protection_actual=" << budget.protected_count
              << " mature_budget_target=" << budget.target.mature
              << " mature_actual=" << compact_count - budget.protected_count
              << " target_total_gaussians=" << budget.target.total
              << " actual_vs_target=" << compact_count / std::max(1.0, budget.target.total)
              << " expected_mature=" << expected_mature.detach().item<float>()
              << " tf=" << transform_count << " visible=" << visible_count << '\n';
    gaussians_->protection_.report(std::cout, selector_protection_enabled_);
  }
  result.trained = mature_count > 0;
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
        if (initial_mapped_ && points.size() >= 30) {
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
        if (initial_mapped_ && points.size() >= 30) {
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
  reportUltrafastRotation("save");
  gaussians_->protection_.report(std::cout, selector_protection_enabled_);
  if (global_gi_) {
    const auto mature = selector_protection_enabled_
        ? gaussians_->getSelectorMatureMask()
        : torch::ones({gaussians_->getXYZ().size(0)}, gaussians_->getXYZ().options().dtype(torch::kBool));
    global_gi_->recordGuidance(mature, selector_enabled_ && selector_gi_loss_weight_ != 0.0f &&
        hasNormalOptimizationBudget());
    global_gi_->report();
  }
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
  saveViewBankOrbMap(result_dir);
  saveSelectorConfig(result_dir);
  keyframesToJson(result_dir);
  saveModelParams(result_dir);
  if (ultrafast_rotation_.enabled) {
    const auto& r = ultrafast_rotation_;
    Json::Value state;
    const char* keys[] = {"lod_001", "lod_005", "lod_010", "lod_015", "full"};
    for (std::size_t i = 0; i < r.counts.size(); ++i) {
      state[std::string("ultrafast_") + keys[i] + "_steps"] = Json::UInt64(r.counts[i]);
      if (r.counts[i]) {
        state[std::string(keys[i]) + "_mean_ms"] = r.elapsed_sum[i] / r.counts[i];
        state[std::string(keys[i]) + "_render_mean_ms"] = r.render_sum[i] / r.counts[i];
      }
    }
    state["ultrafast_total_normal_steps"] = Json::UInt64(r.step);
    state["balanced"] = r.balanced();
    state["mean_mapping_training_ms"] = r.mean();
    state["p95_mapping_training_ms"] = r.p95();
    state["full_render_count"] = Json::UInt64(r.full_render_count);
    state["lod_render_count"] = Json::UInt64(r.lod_render_count);
    state["online_ended"] = r.online_ended;
    state["online_end_mapping_iteration"] = Json::UInt64(r.online_end_mapping_iteration);
    state["online_end_rotation_step"] = Json::UInt64(r.online_end_rotation_step);
    state["tail_training_steps"] = Json::UInt64(r.tailSteps());
    state["selector_online_lod_update_count"] = Json::UInt64(r.selector_updates);
    state["tf_online_lod_update_count"] = Json::UInt64(r.tf_updates);
    state["tf_first_initialized_mapping_iteration"] = Json::Int64(ultrafast_tf_first_initialized_);
    state["tf_first_applied_lod_iteration"] = Json::Int64(ultrafast_tf_first_applied_);
    state["timing"] = device_type_ == torch::kCUDA ? "synchronized_wall_and_cuda_event_render" : "cpu_wall";
    std::ofstream output(result_dir / "ultrafast_rotation_state.json");
    output << state;
  }

  std::filesystem::path ply_dir = result_dir / "point_cloud";
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

  ply_dir = ply_dir / ("iteration_" + std::to_string(getIteration()));
  CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

  gaussians_->savePly(ply_dir / "point_cloud.ply");
  gaussians_->saveSelectorMetadataPly(ply_dir / "selector_metadata.ply");
  Json::Value protection_state;
  protection_state["schema"] = "evidence_aware";
  protection_state["protection_enabled"] = selector_protection_enabled_;
  protection_state["selector_enabled"] = selector_enabled_;
  protection_state["mapping_iteration"] = Json::Int64(gaussians_->protection_.mapping_iteration);
  protection_state["metadata_memory_mb"] = gaussians_->protection_.metadataMB();
  for (const auto& entry : gaussians_->protection_.statistics(selector_protection_enabled_)) protection_state[entry.first] = entry.second;
  std::ofstream protection_output(ply_dir / "protection_state.json");
  protection_output << protection_state;

  if (selector_network_) {
    // Avoid torch::save's generic operator<< path; LibTorch serializes the
    // registered module parameters through Module::save(OutputArchive&).
    torch::serialize::OutputArchive selector_archive;
    selector_network_->save(selector_archive);
    selector_archive.save_to((result_dir / "selector.pt").string());
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
    state["online_enhanced_mappings"] = Json::UInt64(ultrafast_rotation_.step);
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
    state["mature_only"] = transform_field_config_.mature_only;
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
  settings_file << "Training.ultra_light_enabled: "
                << static_cast<int>(ultrafast_rotation_.enabled) << '\n';
  settings_file << "Training.ultra_light_interval: "
                << ultra_light_legacy_interval_ << "\n\n";
  settings_file << "Training.force_tail_optimization_enabled: "
                << static_cast<int>(tail_optimization_schedule_.enabled) << '\n';
  settings_file << "Training.tail_optimization_steps: "
                << tail_optimization_schedule_.steps << "\n\n";
  settings_file << "Selector.enabled: " << static_cast<int>(selector_enabled_)
                << '\n';
  settings_file << "Selector.target_ratios: [";
  for (std::size_t i = 0; i < selector_target_ratios_.size(); ++i) {
    if (i != 0) settings_file << ", ";
    settings_file << selector_target_ratios_[i];
  }
  settings_file << "]\n";
  settings_file << "Selector.protection_enabled: " << int(selector_protection_enabled_) << '\n';
  settings_file << "Protection.age_ready: " << protection_config_.age_ready << '\n';
  settings_file << "Protection.optimization_ready: " << protection_config_.optimization_ready << '\n';
  settings_file << "Protection.unique_keyframe_ready: " << protection_config_.unique_keyframe_ready << '\n';
  settings_file << "Protection.stale_age: " << protection_config_.stale_age << '\n';
  settings_file << "Protection.max_budget_fraction: " << protection_config_.max_budget_fraction << '\n';
  settings_file << "Protection.weighted_bernoulli: 1\n";
  settings_file << "Protection.clone_split_inherit_keyframe_support: 1\n";
  settings_file << "Protection.performance_logging: " << int(protection_config_.performance_logging) << '\n';
  settings_file << "Selector.temperature: " << selector_temperature_ << '\n';
  settings_file << "Selector.learning_rate: " << selector_learning_rate_ << '\n';
  settings_file << "Selector.render_loss_weight: "
                << selector_render_loss_weight_ << '\n';
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

  settings_file << "RecentWindow.enable: "
                << static_cast<int>(recent_window_config_.enabled) << '\n';
  settings_file << "RecentWindow.size: " << recent_window_config_.size << '\n';
  settings_file << "RecentWindow.ema_decay: "
                << recent_window_config_.ema_decay << '\n';
  settings_file << "GlobalGI.mode: \"" << global_gi_config_.mode << "\"\n";
  settings_file << "GlobalGI.view_ratio: " << global_gi_config_.view_ratio << '\n';
  settings_file << "GlobalGI.coverage_anchor_fraction: " << global_gi_config_.coverage_anchor_fraction << '\n';
  settings_file << "GlobalGI.sampling_seed: " << global_gi_config_.sampling_seed << '\n';
  settings_file << "GlobalGI.refresh_interval: " << global_gi_config_.refresh_interval << '\n';
  settings_file << "GlobalGI.max_views: " << global_gi_config_.max_views << '\n';
  settings_file << "GlobalGI.performance_logging: " << static_cast<int>(global_gi_config_.performance_logging) << '\n';
  settings_file << "GlobalGI.verbose_logging: " << static_cast<int>(global_gi_config_.verbose_logging) << '\n';
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
  settings_file << "TransformField.mature_only: "
                << static_cast<int>(transform_field_config_.mature_only) << '\n';
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
  settings_file << "ChangeDistill.seed: " << change_distill_config_.seed << std::endl;
  settings_file << "ChangeDistill.enable_unaffected_anchors: "
                << static_cast<int>(change_distill_config_.enable_unaffected_anchors) << std::endl;
  settings_file << "ChangeDistill.anchor_ratio_to_corrected: "
                << change_distill_config_.anchor_ratio_to_corrected << std::endl;
  settings_file << "ChangeDistill.anchor_max_count: "
                << change_distill_config_.anchor_max_count << std::endl;
  settings_file << "ChangeDistill.anchor_loss_weight: "
                << change_distill_config_.anchor_loss_weight << std::endl;
  settings_file << "GradientConstraint.enable: " << static_cast<int>(gradient_constraint_config_.enabled) << '\n';
  settings_file << "GradientConstraint.check_interval: " << gradient_constraint_config_.check_interval << '\n';
  settings_file << "GradientConstraint.memory_views_per_check: " << gradient_constraint_config_.memory_views_per_check << '\n';
  settings_file << "GradientConstraint.memory_ratios_per_check: " << gradient_constraint_config_.memory_ratios_per_check << '\n';
  settings_file << "GradientConstraint.max_candidate_checks: " << gradient_constraint_config_.max_candidate_checks << '\n';
  settings_file << "GradientConstraint.tf_warmup_steps: " << gradient_constraint_config_.tf_warmup_steps << '\n';
  settings_file << "GradientConstraint.log_interval: " << gradient_constraint_config_.log_interval << '\n';
  settings_file << "GradientConstraint.cancel_fallback_enable: " << static_cast<int>(gradient_constraint_config_.cancel_fallback_enable) << '\n';
  settings_file << "GradientConstraint.cancel_fallback_scale: " << gradient_constraint_config_.cancel_fallback_scale << '\n';
  settings_file << "DriftStability.enable: "
                << static_cast<int>(drift_stability_config_.enabled) << std::endl;
  settings_file << "DriftStability.probe_interval: "
                << drift_stability_config_.probe_interval << std::endl;
  settings_file << "DriftStability.min_bad_views: "
                << drift_stability_config_.min_bad_views << std::endl;
  settings_file << "DriftStability.max_gaussians: "
                << drift_stability_config_.max_gaussians << std::endl;
  settings_file << "DriftStability.batch_size: "
                << drift_stability_config_.batch_size << std::endl;
  settings_file << "DriftStability.consecutive_trigger_count: "
                << drift_stability_config_.consecutive_trigger_count << std::endl;
  settings_file << "DriftStability.reference_horizon: "
                << drift_stability_config_.reference_horizon << std::endl;
  settings_file << "DriftStability.reference_min_age: "
                << drift_stability_config_.reference_min_age << std::endl;
  settings_file << "DriftStability.reference_max_age: "
                << drift_stability_config_.reference_max_age << std::endl;
  settings_file << "DriftStability.reference_ema_decay: "
                << drift_stability_config_.reference_ema_decay << std::endl;
  settings_file << "DriftStability.non_regression_refresh_count: "
                << drift_stability_config_.non_regression_refresh_count << std::endl;
  settings_file << "DriftStability.function_threshold: "
                << drift_stability_config_.function_threshold << std::endl;
  settings_file << "DriftStability.psnr_regression_threshold: "
                << drift_stability_config_.psnr_regression_threshold << std::endl;
  settings_file << "DriftStability.budget_relative_tolerance: "
                << drift_stability_config_.budget_relative_tolerance << std::endl;
  settings_file << "DriftStability.position_weight: "
                << drift_stability_config_.position_weight << std::endl;
  settings_file << "DriftStability.scale_weight: "
                << drift_stability_config_.scale_weight << std::endl;
  settings_file << "DriftStability.rotation_weight: "
                << drift_stability_config_.rotation_weight << std::endl;
  settings_file << "DriftStability.selector_learning_rate: "
                << drift_stability_config_.selector_learning_rate << std::endl;
  settings_file << "DriftStability.tf_grid_learning_rate: "
                << drift_stability_config_.tf_grid_learning_rate << std::endl;
  settings_file << "DriftStability.tf_mlp_learning_rate: "
                << drift_stability_config_.tf_mlp_learning_rate << std::endl;
  settings_file << "DriftStability.epochs: "
                << drift_stability_config_.epochs << std::endl;
  settings_file << "DriftStability.seed: "
                << drift_stability_config_.seed << std::endl;
  settings_file << "HistoricalOutlier.diversity_enabled: "
                << static_cast<int>(historical_outlier_config_.diversity_enabled) << std::endl;
  settings_file << "HistoricalOutlier.count: "
                << historical_outlier_config_.count << std::endl;
  settings_file << "HistoricalOutlier.translation_weight: "
                << historical_outlier_config_.translation_weight << std::endl;
  settings_file << "HistoricalOutlier.rotation_weight: "
                << historical_outlier_config_.rotation_weight << std::endl;
  settings_file << "HistoricalOutlier.diversity_recent_weight: "
                << historical_outlier_config_.diversity_recent_weight << std::endl;
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

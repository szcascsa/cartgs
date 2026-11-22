#include "include/gradient_constraint.h"
#include "include/historical_gradient_render.h"
#include "include/graphics_utils.h"
#include "include/loss_utils.h"
#include <json/json.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>

// Read-only checkpoint benchmark: shared Gaussian/S/TF photometric updates on
// a recent view, with/without historical guards, from identical weights/noise.
// Adam starts fresh in BOTH cases; GI/topology and long-run SLAM are not simulated.
namespace {
namespace fs = std::filesystem;
void check(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
Json::Value readJson(const fs::path& path) {
  std::ifstream input(path); check(input.good(), "Cannot read " + path.string());
  Json::Value result; input >> result; return result;
}
float number(const cv::FileStorage& config, const char* key, float fallback) {
  return config[key].empty() ? fallback : static_cast<float>(config[key]);
}
std::shared_ptr<GaussianKeyframe> view(const Json::Value& saved, const fs::path& dataset) {
  auto camera = std::make_shared<GaussianKeyframe>(saved["id"].asUInt64());
  camera->image_width_ = saved["width"].asInt(); camera->image_height_ = saved["height"].asInt();
  camera->FoVx_ = graphics_utils::focal2fov(saved["fx"].asFloat(), camera->image_width_);
  camera->FoVy_ = graphics_utils::focal2fov(saved["fy"].asFloat(), camera->image_height_);
  auto twc = torch::eye(4);
  for (int r = 0; r < 3; ++r) {
    twc.index_put_({r, 3}, saved["position"][r].asFloat());
    for (int c = 0; c < 3; ++c) twc.index_put_({r, c}, saved["rotation"][r][c].asFloat());
  }
  twc = twc.to(torch::kCUDA);
  camera->world_view_transform_ = twc.inverse().transpose(0, 1).contiguous();
  camera->camera_center_ = twc.slice(0, 0, 3).select(1, 3).contiguous();
  camera->full_proj_transform_ = camera->world_view_transform_.matmul(camera->getProjectionMatrix(
      0.01f, 100.0f, camera->FoVx_, camera->FoVy_, torch::kCUDA).transpose(0, 1)).contiguous();
  auto image = cv::imread((dataset / saved["img_name"].asString()).string(), cv::IMREAD_COLOR);
  check(!image.empty(), "Missing checkpoint RGB image");
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB); image.convertTo(image, CV_32FC3, 1.0 / 255.0);
  camera->original_image_ = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kFloat32)
      .clone().permute({2, 0, 1}).contiguous().to(torch::kCUDA);
  return camera;
}
}
int main(int argc, char** argv) {
  try {
    check(argc >= 5 && argc <= 6, "Usage: gradient_guard_checkpoint_test CHECKPOINT DATASET GAUSSIAN_CONFIG SELECTOR_CONFIG [STEPS]");
    if (!torch::cuda::is_available()) return 77;
    torch::manual_seed(2026); std::cout << std::fixed << std::setprecision(4);
    const fs::path checkpoint(argv[1]), dataset(argv[2]);
    const int steps = argc == 6 ? std::stoi(argv[5]) : 4; check(steps > 0, "STEPS must be positive");
    cv::FileStorage gaussian_config(argv[3], cv::FileStorage::READ), config(argv[4], cv::FileStorage::READ);
    check(gaussian_config.isOpened() && config.isOpened(), "Cannot read experiment configurations");
    const auto saved = readJson(checkpoint / "transform_state.json"), cameras = readJson(checkpoint / "cameras.json");
    check(cameras.size() >= 4, "Need four saved cameras");
    const auto iteration = saved["mapping_iter"].asInt64();
    GaussianModel model(3);
    const auto population_path = checkpoint / "point_cloud" / ("iteration_" + std::to_string(iteration));
    model.loadPly(population_path / "point_cloud.ply"); model.loadSelectorMetadataPly(population_path / "selector_metadata.ply");
    GuardGaussianParameters raw{model.xyz_, model.features_dc_, model.features_rest_, model.opacity_,
        model.scaling_, model.rotation_, model.active_sh_degree_, model.max_sh_degree_};
    auto selector = std::make_shared<improvements::selector::GumbelNetworkImpl>(); selector->to(torch::kCUDA);
    TransformFieldConfig tf;
    tf.hexplane.feature_dim = saved["feature_dim"].asInt(); tf.hexplane.spatial_resolution = saved["spatial_resolution"].asInt();
    tf.hexplane.ratio_resolution = saved["ratio_resolution"].asInt(); tf.hidden_dim = saved["hidden_dim"].asInt();
    tf.hexplane.multires.clear(); for (const auto& level : saved["multires"]) tf.hexplane.multires.push_back(level.asInt());
    tf.outside_identity = saved["outside_identity"].asBool();
    tf.maturity_compute_epsilon = saved.get("maturity_compute_epsilon", 0).asFloat();
    std::vector<float> low, high;
    for (const auto& x : saved["aabb_min"]) low.push_back(x.asFloat());
    for (const auto& x : saved["aabb_max"]) high.push_back(x.asFloat());
    auto field = std::make_shared<TransformFieldImpl>(torch::tensor(low, raw.xyz.options()), torch::tensor(high, raw.xyz.options()), tf);
    field->to(torch::kCUDA);
    torch::serialize::InputArchive sa, fa;
    sa.load_from((checkpoint / "selector.pt").string(), torch::kCUDA); selector->load(sa);
    fa.load_from((checkpoint / "transform_field.pt").string(), torch::kCUDA); field->load(fa);
    const std::vector<torch::Tensor> gaussian_params{raw.xyz, raw.features_dc, raw.features_rest, raw.opacity, raw.log_scaling, raw.rotation};
    auto all = gaussian_params;
    for (const auto& p : selector->parameters()) all.push_back(p);
    for (const auto& p : field->parameters()) all.push_back(p);
    std::vector<torch::Tensor> initial;
    { torch::NoGradGuard no_grad; for (auto& p : all) { p.requires_grad_(true); initial.push_back(p.detach().clone()); } }
    const auto mature = model.getSelectorMatureMask();
    const auto mi = torch::nonzero(mature).reshape({-1}); check(mi.numel() > 0, "Checkpoint has no mature population");
    const double temperature = number(config, "Selector.temperature", 1.0f);
    const float scale = number(config, "Selector.lod_render_scale", 0.5f);
    const auto maturity = model.protection_.maturity;
    std::vector<float> ratios; config["Selector.target_ratios"] >> ratios; check(!ratios.empty(), "No ratios");
    const auto uniform = model.protection_.cached_uniform;
    auto history = view(cameras[static_cast<Json::ArrayIndex>(cameras.size() / 4)], dataset);
    auto current = view(cameras[static_cast<Json::ArrayIndex>(cameras.size() * 7 / 8)], dataset);
    GaussianPipelineParams pipe; auto bg = torch::zeros({3}, raw.xyz.options()); torch::Tensor color;
    auto render = [&](const std::shared_ptr<GaussianKeyframe>& camera, const GaussianRenderInput& input, bool gradients) {
      return GaussianRenderer::render(camera, std::max(1, int(camera->image_height_ * scale)),
          std::max(1, int(camera->image_width_ * scale)), input, pipe, bg, color, 1.0f, false, false,
          gradients ? RasterizeMode::Training : RasterizeMode::Inference).image;
    };
    auto gt = [&](const std::shared_ptr<GaussianKeyframe>& camera) {
      return camera->resized_image_cache_.get(camera->original_image_,
          std::max(1, int(camera->image_height_ * scale)), std::max(1, int(camera->image_width_ * scale)));
    };
    auto history_gt = gt(history), current_gt = gt(current);
    const float dssim = number(gaussian_config, "Optimization.lambda_dssim", 0.2f);
    auto photo = [&](torch::Tensor image) {
      return (1.0f - dssim) * (image - current_gt).abs().mean() + dssim * (1.0f - loss_utils::ssim(image, current_gt));
    };
    const char* keys[] = {"Optimization.position_lr_init", "Optimization.feature_lr", "Optimization.feature_lr",
        "Optimization.opacity_lr", "Optimization.scaling_lr", "Optimization.rotation_lr"};
    const float defaults[] = {0.00016f, 0.0025f, 0.000125f, 0.05f, 0.005f, 0.001f};
    for (bool constrained : {false, true}) {
      { torch::NoGradGuard no_grad; for (std::size_t i = 0; i < all.size(); ++i) all[i].copy_(initial[i]); }
      torch::optim::Adam optimizer(std::vector<torch::Tensor>{gaussian_params[0]},
          torch::optim::AdamOptions(number(gaussian_config, keys[0], defaults[0])).eps(1e-15));
      for (int g = 1; g < 6; ++g) {
        optimizer.add_param_group(std::vector<torch::Tensor>{gaussian_params[g]});
        optimizer.param_groups()[g].options().set_lr(number(gaussian_config, keys[g], defaults[g]) / (g == 2 ? 20.0f : 1.0f));
      }
      optimizer.add_param_group(selector->parameters()); optimizer.param_groups()[6].options().set_lr(number(config, "Selector.learning_rate", 0.001f));
      optimizer.add_param_group(field->mlpParameters()); optimizer.param_groups()[7].options().set_lr(number(config, "TransformField.mlp_lr_init", 0.00016f));
      optimizer.add_param_group(field->gridParameters()); optimizer.param_groups()[8].options().set_lr(number(config, "TransformField.grid_lr_init", 0.0016f));
      GuardModuleCandidates scratch;
      for (int step = 0; step < steps; ++step) {
        const float ratio = ratios[step % ratios.size()];
        optimizer.zero_grad(true);
        AdamStepPreview empty(optimizer);
        const auto full_image = render(current, raw.input(empty), true);
        photo(full_image).backward();
        const auto lod_input = historicalLoDInput(raw.input(empty), maturity, uniform, selector, field, ratio, temperature);
        (photo(render(current, lod_input, true)) + field->regularizationLoss()).backward();
        GuardEvaluation e; e.objectives = {{"full"}, {"lod"}};
        GaussianRenderInput candidate_input;
        e.prepare_candidate = [&](const AdamStepPreview& p, const std::vector<torch::Tensor>& values) {
          scratch.prepare(p, values, selector, field); candidate_input = raw.input(p, &values);
        };
        e.evaluate = [&](std::size_t objective, const AdamStepPreview& p, const std::vector<torch::Tensor>* values) {
          const auto input = values ? candidate_input : raw.input(p);
          const auto s = values ? scratch.selector() : selector; const auto f = values ? scratch.field() : field;
          const auto selected = objective == 0 ? input : historicalLoDInput(input, maturity, uniform, s, f, ratio, temperature);
          return (render(history, selected, values == nullptr && torch::GradMode::is_enabled()) - history_gt).square().mean();
        };
        GradientConstraintReport report;
        std::vector<double> budget_before;
        {
          torch::NoGradGuard no_grad;
          const auto input = raw.input(empty);
          for (float r : ratios) budget_before.push_back(mixSelectorProbability(selector->forwardProbability(
              input.xyz, input.rotation, input.scaling, r, temperature), maturity, r).sum().item<double>());
        }
        const auto start = std::chrono::steady_clock::now();
        if (constrained) report = GradientConstraintManager().step(optimizer, e);
        else {
          torch::NoGradGuard no_grad;
          for (std::size_t j = 0; j < e.objectives.size(); ++j) report.before.push_back(e.evaluate(j, empty, nullptr).item<double>());
          optimizer.step(); report.status = "UNCONSTRAINED"; report.reason = "baseline";
        }
        {
          torch::NoGradGuard no_grad;
          report.after.clear();
          for (std::size_t j = 0; j < e.objectives.size(); ++j) report.after.push_back(e.evaluate(j, empty, nullptr).item<double>());
        }
        if (constrained) {
          for (int j = 0; j < 2; ++j) check(report.after[j] <= report.before[j] + 1e-7 + std::abs(report.before[j]) * 1e-5,
              "Committed checkpoint guard increased actual historical MSE");
        }
        std::cout << "[GradientCheckpoint] mode=" << (constrained ? "GUARDED" : "BASELINE") << " step=" << step
                  << " ratio=" << ratio << " status=" << report.status << " reason=" << report.reason << " alpha=" << report.alpha
                  << " full_psnr=" << -10.0 * std::log10(report.before[0]) << '/' << -10.0 * std::log10(report.after[0])
                  << " lod_psnr=" << -10.0 * std::log10(report.before[1]) << '/' << -10.0 * std::log10(report.after[1])
                  << " ms=" << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() << '\n';
        {
          torch::NoGradGuard no_grad;
          const auto input = raw.input(empty);
          for (std::size_t r = 0; r < ratios.size(); ++r) {
            const double count = mixSelectorProbability(selector->forwardProbability(input.xyz,
                input.rotation, input.scaling, ratios[r], temperature), maturity, ratios[r]).sum().item<double>();
            std::cout << "[GradientCheckpoint Budget] ratio=" << ratios[r]
                      << " expected=" << budget_before[r] << '/' << count
                      << " target=" << double(ratios[r]) * input.xyz.size(0) << " gate=OFF\n";
          }
        }
      }
    }
    std::cout << "[GradientCheckpoint] PASS population=" << raw.xyz.size(0) << " mature=" << mi.numel() << '\n';
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

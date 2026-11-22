#include "include/gaussian_model.h"
#include "improvements/selector/gumbel_network.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
  TORCH_CHECK(argc == 5 || argc == 6, "Usage: checkpoint_probe selector.pt point_cloud.ply selector_metadata.ply output.pt [attributes]");
  torch::NoGradGuard no_grad;
  GaussianModel model(3);
  model.loadPly(argv[2]); model.loadSelectorMetadataPly(argv[3]);
  const bool global_xyz = argc == 6 && std::string(argv[5]) == "global_xyz";
  const bool blocks = argc == 6 && std::string(argv[5]) == "explicit_blocks";
  auto selector = improvements::selector::GumbelNetwork(.1, argc == 6 && (std::string(argv[5]) == "attributes" || global_xyz || blocks), global_xyz,blocks);
  selector->to(model.getXYZ().device());
  torch::serialize::InputArchive archive;
  archive.load_from(argv[1], model.getXYZ().device()); selector->load(archive);
  TORCH_CHECK(selector->anchors()->rows().numel() == model.getXYZ().size(0), "Checkpoint source-row mismatch");
  TORCH_CHECK((model.protection_.maturity == 1).all().item<bool>(), "Anchor checkpoint maturity must be one");
  torch::serialize::OutputArchive result;
  const std::vector<float> ratios=selector->privateCompensation()?std::vector<float>{.01f,.025f,.05f,.075f,.1f,.125f,.15f,1.f}:
      std::vector<float>{.01f,.05f,.1f,.15f};
  result.write("ratios",torch::tensor(ratios));
  for (size_t r=0;r<ratios.size();++r) {
    auto p = selector->forwardProbability(model.getXYZ(), {}, model.getScalingActivation(), ratios[r], 1, {},
        model.features_dc_, model.getOpacityActivation());
    p = selector->maskInactiveSlots(p,model.getOpacityActivation());
    result.write("probability_"+std::to_string(r), p);
    result.write("mask_"+std::to_string(r), model.protection_.cached_uniform < p);
    if(selector->compensationEnabled()) {
      auto encoded=selector->forwardSelectionFeatures(model.getXYZ(),ratios[r],1,{},model.getScalingActivation(),
          model.features_dc_,model.getOpacityActivation());
      auto ids=torch::nonzero(model.protection_.cached_uniform<p).reshape({-1});
      torch::Tensor residual;
      if(selector->spatialCompensationEnabled()) {
        auto& spatial=selector->spatialCompensation();auto owner=selector->anchors()->rows().index_select(0,ids);
        TORCH_CHECK(spatial->ready(),"Spatial checkpoint has no initialized normalization");
        residual=spatial->forward(model.getXYZ().index_select(0,ids),model.getScalingActivation().index_select(0,ids),
            model.getRotationActivation().index_select(0,ids),model.features_dc_.index_select(0,ids),model.opacity_.index_select(0,ids),
            selector->anchors()->centers().index_select(0,owner),selector->anchors()->offsetScales().index_select(0,owner),ratios[r]);
      } else residual=selector->compensationResidual(encoded.shared.index_select(0,ids),ids,ratios[r]);
      auto attributes=improvements::selector::GumbelNetworkImpl::applyCompensationResidual(
          {model.getXYZ().index_select(0,ids),model.scaling_.index_select(0,ids),model.getRotationActivation().index_select(0,ids)},residual);
      result.write("compensated_xyz_"+std::to_string(r),attributes.xyz);
      result.write("compensated_scale_"+std::to_string(r),model.getScalingActivation().index_select(0,ids)*residual.slice(1,3,6).exp());
      result.write("compensated_rotation_"+std::to_string(r),attributes.rotation);
      if(residual.size(1)==14) {
        auto appearance=improvements::selector::GumbelNetworkImpl::applyAppearanceResidual(
            model.features_dc_.index_select(0,ids),model.opacity_.index_select(0,ids),residual);
        result.write("compensated_dc_"+std::to_string(r),appearance.first);
        result.write("compensated_opacity_logits_"+std::to_string(r),appearance.second);
      }
    }
  }
  // Both fixed/adaptive checkpoints use the identical native timing protocol.
  const auto scale = model.getScalingActivation(), opacity = model.getOpacityActivation();
  std::vector<double> forward_ms;
  for (int repeat=0;repeat<5;++repeat) {
    for (int warm=0;warm<20;++warm)
      selector->forwardAnchorProbability(model.getXYZ(),.1f,1,{},scale,model.features_dc_,opacity);
    torch::cuda::synchronize();
    const auto start=std::chrono::steady_clock::now();
    for (int i=0;i<200;++i)
      selector->forwardAnchorProbability(model.getXYZ(),.1f,1,{},scale,model.features_dc_,opacity);
    torch::cuda::synchronize();
    forward_ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/200);
  }
  result.write("native_forward_ms",torch::tensor(forward_ms,torch::kFloat64));
  if (!selector->anchors()->adaptive() && !blocks) {
    auto input_before=selector->anchors()->inputs(model.getXYZ());
    std::vector<torch::Tensor> before;
    for(float ratio:ratios)before.push_back(selector->forwardAnchorProbability(model.getXYZ(),ratio,1,{},scale,model.features_dc_,opacity));
    selector->anchors()->enableAdaptive();
    const auto splits=selector->anchors()->refine(model.getXYZ());
    auto input_after=selector->anchors()->inputs(model.getXYZ());
    TORCH_CHECK(torch::equal(input_before.first,input_after.first),"Real-map refinement changed inherited Feature");
    if (!global_xyz) TORCH_CHECK(torch::equal(input_before.second,input_after.second),"Real-map refinement changed old offset");
    double error=0;
    std::vector<double> changed_fractions,mean_errors;
    for(size_t i=0;i<ratios.size();++i) {
      auto after=selector->forwardAnchorProbability(model.getXYZ(),ratios[i],1,{},scale,model.features_dc_,opacity);
      error=std::max(error,(after-before[i]).abs().max().item<double>());
      changed_fractions.push_back(((model.protection_.cached_uniform<after)!=(model.protection_.cached_uniform<before[i])).to(torch::kFloat32).mean().item<double>());
      mean_errors.push_back((after-before[i]).abs().mean().item<double>());
      if (!global_xyz) {
        TORCH_CHECK(torch::allclose(after,before[i],1e-6,1e-7),"Real-map refinement discontinuity");
        TORCH_CHECK(changed_fractions.back()==0,"Real-map split changed Cached Bernoulli set");
      }
    }
    result.write(global_xyz?"refinement_jump_error":"refinement_continuity_error",torch::tensor(error,torch::kFloat64));
    result.write("refinement_jump_changed_mask_fraction",torch::tensor(changed_fractions,torch::kFloat64));
    result.write("refinement_jump_mean_probability_error",torch::tensor(mean_errors,torch::kFloat64));
    result.write("refinement_continuity_splits",torch::tensor(splits,torch::kLong));
  }
  result.save_to(argv[4]);
  std::cout << "PASS: native real-checkpoint inference, population=" << model.getXYZ().size(0)
            << " anchors=" << selector->anchors()->size() << '\n';
}

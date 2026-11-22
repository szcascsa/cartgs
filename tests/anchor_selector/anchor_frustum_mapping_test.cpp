#include "include/gaussian_mapper.h"
#include "include/anchor_frustum.h"
#include <iostream>

class FrustumMapper : public GaussianMapper {
 public:
  FrustumMapper(const std::filesystem::path& output,const std::filesystem::path& config)
      : GaussianMapper(nullptr,std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/
          "cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml",output,0,torch::kCUDA,true,config) {}
  void verify(const std::filesystem::path& output) {
    TORCH_CHECK(selector_anchor_frustum_culling_,"Production default must enable Anchor frustum culling");
    auto o=torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto& map=*gaussians_;
    std::map<point3D_id_t,Point3D> observed;
    for(int i=0;i<8;++i) {
      Point3D p;p.xyz_=Eigen::Vector3d(i<2?((selector_anchor_set_encoder_ || selector_anchor_set_statistics_ || selector_anchor_context_residual_)?.01+i*.004:i*.2):20+i*.2,.02,2.02);
      p.color_=Eigen::Vector3f(.4,.4,.4);observed[i]=p;
    }
    map.createFromPcd(observed,1);map.trainingSetup(opt_params_);
    map.active_sh_degree_=1;default_sh_=1;
    map.protection_.maturity.fill_(1);map.protection_.mapping_iteration=6000;
    iteration_=mapping_iter_=6000;opt_params_.iterations_=10000;
    scene_->cameras_extent_=1;selector_compensation_start_iter_=0;
    selector_render_loss_weight_=0;selector_ratio_loss_weight_=1;
    selector_replay_enabled_=false;gradient_constraint_config_.enabled=false;
    auto camera=std::make_shared<GaussianKeyframe>(0);
    camera->setPose(Eigen::Quaterniond::Identity(),Eigen::Vector3d::Zero());
    camera->image_height_=32;camera->image_width_=40;
    camera->FoVx_=camera->FoVy_=1;
    camera->world_view_transform_=torch::eye(4,o);
    camera->full_proj_transform_=camera->getProjectionMatrix(.01f,100.f,1.f,1.f,torch::kCUDA).transpose(0,1).contiguous();
    camera->camera_center_=torch::zeros({3},o);
    camera->original_image_=torch::full({3,32,40},.1,o);
    auto& bank=selector_network_->anchors();
    auto candidates=anchorNominalFrustumMask(bank->centers(),bank->offsetScales(),camera->full_proj_transform_).index_select(0,bank->rows());
    const auto C=candidates.sum().item<int64_t>(),N=map.getXYZ().size(0);
    TORCH_CHECK(C>0 && C<.4*N,"Fixture must include visible/offscreen Anchors and a budget shortage");
    if(selector_network_->selectorSetEncoder() || selector_network_->selectorContextResidual())
      TORCH_CHECK(bank->memberCounts().max().item<int64_t>()>1,"Mapping fixture must exercise multi-member Anchor context");
    auto rows=torch::nonzero(candidates).flatten();
    if(!selector_network_->selectorSetEncoder() && !selector_network_->selectorContextResidual()) {
    auto full_cuda=selector_network_->forwardFlexgsCUDA(map.getXYZ(),.1f,map.getScalingActivation(),map.features_dc_,map.getOpacityActivation());
    auto compact_cuda=selector_network_->forwardFlexgsCUDA(map.getXYZ().index_select(0,rows),.1f,
        map.getScalingActivation().index_select(0,rows),map.features_dc_.index_select(0,rows),
        map.getOpacityActivation().index_select(0,rows),rows);
    TORCH_CHECK(compact_cuda.logits.equal(full_cuda.logits.index_select(0,rows)) &&
        compact_cuda.shared.equal(full_cuda.shared.index_select(0,rows)),"CUDA Selector must gather global Anchor owners");
    } else {
      auto full=selector_network_->forwardSelectionFeatures(map.getXYZ(),.1f,1.,{},map.getScalingActivation(),map.features_dc_,map.getOpacityActivation());
      auto compact=selector_network_->forwardSelectionFeatures(map.getXYZ().index_select(0,rows),.1f,1.,rows,
          map.getScalingActivation().index_select(0,rows),map.features_dc_.index_select(0,rows),map.getOpacityActivation().index_select(0,rows));
      TORCH_CHECK(torch::allclose(compact.logits,full.logits.index_select(0,rows),1e-5,1e-6),"Complete candidate Anchor sets differ from full context");
    }
    {torch::NoGradGuard guard;selector_network_->named_parameters()["soft_net.bias"].copy_(torch::tensor({0.f,80.f},o));}
    initializeTransformFieldIfNeeded();ensureSharedMappingOptimizer();
    for(size_t ratio=0;ratio<selector_target_ratios_.size();++ratio) {
      map.optimizer_->zero_grad(true);
      auto result=accumulateLodGradients(camera,32,40,camera->original_image_,ratio,false,false);
      TORCH_CHECK(result.finite_loss && result.active_gaussians==C && result.compensated_gaussians==C,"Candidate selection/compensation count");
      const double target=double(selector_target_ratios_[ratio])*N;
      TORCH_CHECK(std::abs(result.loss-std::abs(C-target)/std::max(1.,target))<1e-5,"Budget loss must use global N");
      if(result.supervised_indices.defined())
        TORCH_CHECK(candidates.index_select(0,result.supervised_indices).all().item<bool>(),"Renderer IDs escaped candidates");
      for(const auto& p:selector_network_->named_parameters())if(p.value().grad().defined())
        TORCH_CHECK(!p.value().grad().is_view(),"Gradient view cannot survive to LibTorch zero_grad");
      for(auto p:{map.xyz_,map.scaling_,map.rotation_,map.opacity_,map.features_dc_,map.features_rest_})
        if(p.grad().defined())TORCH_CHECK(p.grad().index({~candidates}).abs().sum().item<float>()==0,"Noncandidate Master gradient");
    }
    // Maturity's private VJP may update Selector only, despite using the same
    // candidate logits as the photo/budget graph and the shared Anchor table.
    {torch::NoGradGuard guard;
      selector_network_->named_parameters()["soft_net.bias"].copy_(torch::tensor({0.f,2.f},o));
      if(selector_network_->selectorSetEncoder() || selector_network_->selectorContextResidual())
        selector_network_->named_parameters()["soft_net.weight"].normal_(0,.02);
      map.protection_.maturity.copy_(torch::linspace(.1f,.9f,N,o));
      map.protection_.contribution_maturity.copy_(map.protection_.maturity*.6);}
    selector_ratio_loss_weight_=0;protection_config_.reliability_delta=1;
    map.optimizer_->zero_grad(true);
    auto private_result=accumulateLodGradients(camera,32,40,camera->original_image_,0,false,false);
    TORCH_CHECK(private_result.finite_loss,"Private maturity backward");
    double selector_norm=0,encoder_norm=0;
    for(const auto& p:selector_network_->named_parameters())if(p.value().grad().defined()) {
      const auto norm=p.value().grad().abs().sum().item<double>();
      if(p.key().rfind("selector_set_encoder.",0)==0 || p.key().rfind("selector_context_residual.",0)==0)encoder_norm+=norm;
      if(p.key().rfind("local_emd.",0)==0 || p.key().rfind("soft_net.",0)==0 ||
          p.key().rfind("selector_set_encoder.",0)==0 || p.key().rfind("selector_context_residual.",0)==0)selector_norm+=norm;
      else TORCH_CHECK(norm==0,"Maturity VJP leaked into Compensation/Anchor Feature");
    }
    TORCH_CHECK(selector_norm>0,"Maturity VJP must reach Selector parameters");
    if(selector_network_->selectorSetEncoder() && selector_network_->selectorSetEncoderSchema()!=3)
      TORCH_CHECK(encoder_norm>0,"Maturity VJP omitted Set Encoder parameters");
    if(selector_network_->selectorContextResidual())TORCH_CHECK(encoder_norm>0,"Maturity VJP omitted context residual projection");
    for(auto p:{map.xyz_,map.scaling_,map.rotation_,map.opacity_,map.features_dc_,map.features_rest_})
      if(p.grad().defined())TORCH_CHECK(p.grad().abs().sum().item<float>()==0,"Maturity VJP leaked into Master");
    // A real photo/budget backward and joint Adam step must remain executable.
    selector_render_loss_weight_=1;selector_ratio_loss_weight_=1;
    const std::string tested_weight=selector_network_->selectorContextResidual()?"selector_context_residual.projection.weight":
        (selector_network_->selectorSetEncoderSchema()==3?"local_emd.0.weight":"selector_set_encoder.network.0.weight");
    auto encoder_before=(selector_network_->selectorSetEncoder() || selector_network_->selectorContextResidual())?
        selector_network_->named_parameters()[tested_weight].detach().clone():torch::Tensor{};
    map.optimizer_->zero_grad(true);
    auto joint=accumulateLodGradients(camera,32,40,camera->original_image_,0,false,false);
    TORCH_CHECK(joint.finite_loss && map.xyz_.grad().defined() && torch::isfinite(map.xyz_.grad()).all().item<bool>(),"Joint Master backward");
    TORCH_CHECK(stepWithHistoricalGuard(camera,false),"Joint Adam commit");
    if(encoder_before.defined())TORCH_CHECK(!encoder_before.equal(
        selector_network_->named_parameters()[tested_weight]),"Joint Adam did not update shared Selector representation");
    if(selector_network_->selectorSetEncoder() || selector_network_->selectorContextResidual()) {
      torch::serialize::OutputArchive model;selector_network_->save(model);model.save_to((output/"selector.pt").string());
      auto expected=selector_network_->forwardSelectionFeatures(map.getXYZ(),.1f,1.,{},map.getScalingActivation(),map.features_dc_,map.getOpacityActivation());
      torch::serialize::OutputArchive fixture;
      fixture.write("xyz",map.getXYZ().detach(),true);fixture.write("scale",map.getScalingActivation().detach(),true);
      fixture.write("dc",map.features_dc_.detach(),true);fixture.write("opacity",map.getOpacityActivation().detach(),true);
      fixture.write("logits",expected.logits.detach(),true);fixture.save_to((output/"selector_fixture.pt").string());
      auto frozen=selector_network_->cloneFrozen();
      auto teacher=frozen->forwardSelectionFeatures(map.getXYZ(),.1f,1.,{},map.getScalingActivation(),map.features_dc_,map.getOpacityActivation());
      TORCH_CHECK(torch::allclose(teacher.logits,expected.logits,1e-5,1e-6),"Frozen Set Encoder/Compensation clone identity");
      auto restored=improvements::selector::GumbelNetwork(.1,true,false,true,std::vector<double>{},0);restored->to(o.device());
      torch::serialize::InputArchive archive;archive.load_from((output/"selector.pt").string(),o.device());restored->load(archive);
      auto loaded=restored->forwardSelectionFeatures(map.getXYZ(),.1f,1.,{},map.getScalingActivation(),map.features_dc_,map.getOpacityActivation());
      TORCH_CHECK((restored->selectorSetEncoder() || restored->selectorContextResidual()) &&
          torch::allclose(loaded.logits,expected.logits,1e-5,1e-6),"Native populated context/Compensation checkpoint identity");
    }
    selector_render_loss_weight_=0;
    {torch::NoGradGuard guard;selector_network_->named_parameters()["soft_net.bias"].copy_(torch::tensor({0.f,80.f},o));}
    auto before=map.getXYZ().detach().clone();
    selector_anchor_frustum_culling_=false;
    auto all=accumulateLodGradients(camera,32,40,camera->original_image_,0,false,false);
    TORCH_CHECK(all.finite_loss && all.active_gaussians==N,"Disabled switch must recover full population");
    selector_anchor_frustum_culling_=true;
    auto projection=camera->full_proj_transform_.clone();
    {torch::NoGradGuard guard;camera->full_proj_transform_.select(0,3).select(0,0).fill_(10000);}
    auto empty=accumulateLodGradients(camera,32,40,camera->original_image_,0,false,false);
    TORCH_CHECK(empty.finite_loss && empty.active_gaussians==0 && empty.compensated_gaussians==0,"Empty frustum handling");
    TORCH_CHECK(map.getXYZ().equal(before),"Candidate screening must not modify Master");
    camera->full_proj_transform_=projection;
    saveSelectorConfig(output/"roundtrip");
    cv::FileStorage saved((output/"roundtrip/selector_config.yaml").string(),cv::FileStorage::READ);
    TORCH_CHECK(int(saved["Selector.anchor_frustum_culling"])==1,"Checkpoint config roundtrip");
    TORCH_CHECK(int(saved["Selector.anchor_set_encoder"])==int(selector_network_->selectorSetEncoder() &&
        selector_network_->selectorSetEncoderSchema()!=3),"Set Encoder config roundtrip");
    TORCH_CHECK(int(saved["Selector.anchor_set_statistics"])==int(selector_network_->selectorSetEncoderSchema()==3),"Statistics config roundtrip");
    TORCH_CHECK(int(saved["Selector.anchor_context_residual"])==int(selector_network_->selectorContextResidual()),"Context residual config roundtrip");
    // Exercise the fallback MLP with the same noncontiguous global owners.
    fused_flexgs_selector_=false;
    auto fallback=accumulateLodGradients(camera,32,40,camera->original_image_,0,false,false);
    TORCH_CHECK(fallback.finite_loss && fallback.active_gaussians==C,"Fallback MLP candidate path");
    {torch::NoGradGuard guard;
      map.xyz_.index_put_({rows[0],0},1000.f);
      // Isolate nominal-Anchor culling from representation-dependent logits.
      selector_network_->named_parameters()["soft_net.weight"].zero_();
      selector_network_->named_parameters()["soft_net.bias"].copy_(torch::tensor({0.f,1000.f},o));}
    auto displaced=accumulateLodGradients(camera,32,40,camera->original_image_,0,false,false);
    TORCH_CHECK(displaced.finite_loss && displaced.active_gaussians==C,"Members must follow nominal Anchor, not live Gaussian XYZ");
    {torch::NoGradGuard guard;
      map.opacity_.index_fill_(0,rows,-12.f);
      bank->inactiveSlots().reshape({-1}).index_fill_(0,rows,true);}
    auto inactive=accumulateLodGradients(camera,32,40,camera->original_image_,0,false,false);
    TORCH_CHECK(inactive.finite_loss && inactive.active_gaussians==0,"Inactive slots escaped candidate eligibility");
    std::cout<<"PASS actual LoD Mapping: candidates="<<C<<" global_N="<<N
        <<" six ratios, global budget, private maturity VJP, joint Adam, off/empty/fallback, displaced/inactive members, config roundtrip\n";
  }
};

int main(int argc,char** argv) {
  if(!torch::cuda::is_available())return 77;
  const auto output=std::filesystem::path(argc>1?argv[1]:"/tmp/cartgs_anchor_frustum_test");
  std::filesystem::create_directories(output);
  FrustumMapper mapper(output,argc>2?std::filesystem::path(argv[2]):
      std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/"cfg/selector/selector_config.yaml");
  mapper.verify(output);
}

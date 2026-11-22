#include "include/gaussian_mapper.h"
#include "include/gaussian_renderer.h"
#include "include/flexgs_importance.h"
#include "include/mapping_cuda_optimizer.h"
#include "cuda_rasterizer/render_workspace.h"
#include "cuda_rasterizer/ssim_backend.h"
#include <iostream>
#include <limits>

class RealLodMapperTest : public GaussianMapper {
 public:
  RealLodMapperTest(const std::filesystem::path& out):GaussianMapper(nullptr,
      std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/"cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml",
      out,7,torch::kCUDA,true,std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/"cfg/selector/spatial_offset3_residual48_global_only.yaml") {}
  void verify(bool freeze_master,bool fused_adam=true,bool full_supervision=true,bool reject_nonfinite=false) {
    TORCH_CHECK(real_lod_.enabled&&flexgs_importance_enabled_&&!ultrafast_rotation_.enabled,
                "Full stages must not use the legacy ratio cursor");
    TORCH_CHECK(selector_anchor_only_pruning_==1&&!protection_config_.reliability_v2&&
        !protection_config_.reliability_gate&&protection_config_.fixed_maturity_one,"RealLoD defaults");
    TORCH_CHECK(selector_target_ratios_.size()==4&&selector_network_->anchors()->compensationFeatureTables().size()==4,
                "Four training ratios must allocate Base48 plus three residuals");
    TORCH_CHECK(!selector_detach_gaussian_parameters_&&real_lod_full_loss_weight_==1.f,
                "RealLoD defaults must use joint Full+LoD supervision");
    selector_detach_gaussian_parameters_=freeze_master;
    real_lod_full_loss_weight_=full_supervision&&!freeze_master?1.f:0.f;
    fused_dense_adam_=fused_adam;
    auto options=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto scores=torch::tensor({0.f,0.f,1.f,2.f,2.f,3.f},options).requires_grad_(true);
    auto target=flexgsImportanceTarget(scores,.5);
    TORCH_CHECK(target.equal(torch::tensor({0.f,0.f,0.f,1.f,1.f,1.f},options))&&!target.requires_grad(),
                "Official strict percentile/zero/tie semantics");
    TORCH_CHECK(flexgsImportanceTarget(torch::zeros({4},options),.8).sum().item<float>()==0,
                "Zero ties are not an exact top-K target");
    std::map<point3D_id_t,Point3D> points;
    for(int i=0;i<80;++i){Point3D p;
      p.xyz_=Eigen::Vector3d((i%10-5)*.025+.013,((i/10)%8-4)*.025+.017,2.02+(i%3)*.005);
      p.color_=Eigen::Vector3f(.4,.3,.2);points[i]=p;}
    gaussians_->createFromPcd(points,1);gaussians_->trainingSetup(opt_params_);
    gaussians_->active_sh_degree_=default_sh_=1;scene_->cameras_extent_=1;
    opt_params_.densify_from_iter_=1000000;opt_params_.opacity_reset_interval_=0;
    opt_params_.iterations_=1; // New protocol must not turn later updates into no-ops.
    do_gaus_pyramid_training_=false;
    keyframe_record_interval_=all_keyframes_record_interval_=training_report_interval_=0;
    undistort_mask_[0]=torch::ones({32,40},options);
    std::vector<std::shared_ptr<GaussianKeyframe>> cameras;
    for(int j=0;j<2;++j){auto camera=std::make_shared<GaussianKeyframe>(j);
      camera->camera_id_=0;camera->setPose(Eigen::Quaterniond::Identity(),Eigen::Vector3d::Zero());
      camera->image_height_=32;camera->image_width_=40;camera->FoVx_=camera->FoVy_=1;
      camera->world_view_transform_=torch::eye(4,options);
      camera->full_proj_transform_=camera->getProjectionMatrix(.01f,100.f,1.f,1.f,torch::kCUDA).transpose(0,1).contiguous();
      camera->camera_center_=torch::zeros({3},options);camera->original_image_=torch::full({3,32,40},.1,options);
      scene_->addKeyframe(camera,&kfid_shuffled_);cameras.push_back(camera);}
    {
      torch::NoGradGuard guard;
      GaussianRenderInput one{torch::tensor({{0.f,0.f,2.f}},options),torch::full({1,1},.5,options),
          torch::full({1,3},.1,options),torch::tensor({{1.f,0.f,0.f,0.f}},options),
          torch::zeros({1,1,3},options),torch::zeros({1,15,3},options),0,3};
      auto raw=GaussianRenderer::render(cameras[0],32,40,one,pipe_params_,background_,override_color_,
                                       1.f,false,true,RasterizeMode::Inference);
      TORCH_CHECK(raw.contribution_count.sum().item<int>()>0&&
          raw.frame_importance.equal(raw.contribution_count.to(torch::kFloat32)*.5f),
          "FlexGS statistic must sum opacity*T at valid pixels, not alpha*T or a pixel mean");
    }
    auto bank=selector_network_->anchors();
    std::vector<torch::Tensor> untouched;for(auto p:selector_network_->parameters())untouched.push_back(p.detach().clone());
    for(int step=0;step<8;++step){
      ++iteration_;trainMappingIteration(cameras[step%2],true,false);++mapping_iter_;real_lod_.completeFull();
    }
    real_lod_.endOnline();
    for(int step=0;step<3;++step){
      ++iteration_;trainMappingIteration(cameras[step%2],true,true);++mapping_iter_;real_lod_.completeFull();
    }
    size_t index=0;for(auto p:selector_network_->parameters()){
      TORCH_CHECK(p.equal(untouched[index++]),"Full stages changed Selector/Compensation/Feature parameters");
      TORCH_CHECK(gaussians_->optimizer_->state().find(p.unsafeGetTensorImpl())==gaussians_->optimizer_->state().end(),
                  "Full stages must not advance network Adam states");}
    TORCH_CHECK(mode2_online_.lod_renders==0&&mode2_tail_.lod_renders==0&&selector_step_==0,
                "Full phases executed a LoD training branch");
    auto before=gaussians_->features_dc_.detach().clone();
    auto key=gaussians_->features_dc_.unsafeGetTensorImpl();
    auto adam_step=[&](){return static_cast<const torch::optim::AdamParamState&>(*gaussians_->optimizer_->state().at(key)).step();};
    auto full_clock=adam_step();auto owner=bank->rows().clone();auto counts=bank->memberCounts().clone();
    const auto topology=topology_version_;const auto N=gaussians_->getXYZ().size(0);
    struct MasterSnapshot {
      torch::Tensor parameter,value,mean,variance;
      int64_t step=0;
      bool has_state=false;
      double learning_rate=0;
      size_t group=0;
    };
    std::vector<MasterSnapshot> master;
    for(size_t g=0;g<6;++g) {
      auto& group=gaussians_->optimizer_->param_groups()[g];
      for(const auto& p:group.params()) {
        MasterSnapshot saved;saved.parameter=p;saved.value=p.detach().clone();saved.group=g;
        saved.learning_rate=static_cast<const torch::optim::AdamOptions&>(group.options()).lr();
        auto it=gaussians_->optimizer_->state().find(p.unsafeGetTensorImpl());
        if(it!=gaussians_->optimizer_->state().end()) {
          const auto& state=static_cast<const torch::optim::AdamParamState&>(*it->second);
          saved.has_state=true;saved.step=state.step();
          saved.mean=state.exp_avg().clone();saved.variance=state.exp_avg_sq().clone();
        }
        master.push_back(saved);
      }
    }
    real_lod_.beginLod();flexgs_importance_interval_=4;
    if(reject_nonfinite) {
      cameras[0]->original_image_.fill_(std::numeric_limits<float>::quiet_NaN());
      bool rejected=false;
      try { trainRealLodIteration(cameras[0]); } catch(const c10::Error&) { rejected=true; }
      TORCH_CHECK(rejected&&real_lod_.lod_steps==0&&adam_step()==full_clock&&
          flexgs_lod_nonfinite_==1&&real_lod_full_backwards_==0,
          "Nonfinite Full loss advanced Adam or counted a successful Fine step");
      for(const auto& saved:master)
        TORCH_CHECK(saved.parameter.equal(saved.value),"Rejected Full loss changed Master");
      index=0;
      for(const auto& p:selector_network_->parameters())
        TORCH_CHECK(p.equal(untouched[index++]),"Rejected Full loss changed a network parameter");
      std::cout<<"PASS nonfinite Full loss rejected before backward/Adam\n";
      return;
    }
    {torch::NoGradGuard guard;refreshFlexgsImportance();
      auto manual=torch::zeros_like(flexgs_importance_);
      for(auto it=cameras.rbegin();it!=cameras.rend();++it)
        manual.add_(GaussianRenderer::render(*it,32,40,gaussians_,pipe_params_,background_,override_color_,
                    1.f,false,{},true,true,RasterizeMode::Inference).frame_importance);
      TORCH_CHECK(torch::allclose(manual,flexgs_importance_,1e-5,1e-5),"Full-view raw importance sum");
      TORCH_CHECK(!flexgs_importance_.requires_grad()&&flexgs_importance_.sum().item<float>()>0,"Detached importance");}
    if(real_lod_full_loss_weight_>0.f)verifyFullGradientRouting(cameras[0]);
    while(real_lod_.optimizingLod())trainRealLodIteration(cameras[real_lod_.lod_steps%2]);
    TORCH_CHECK(real_lod_.lod_steps==8&&real_lod_.target_steps==8&&real_lod_.tail_steps==3,
                "LoD budget includes tail or drops online updates");
    TORCH_CHECK(real_lod_ratio_counts_==std::vector<std::uint64_t>({2,2,2,2}),"Ratio-only cursor");
    TORCH_CHECK(real_lod_full_renders_==(real_lod_full_loss_weight_>0.f?8:0)&&
        real_lod_full_backwards_==real_lod_full_renders_,"Fine Full exposure mismatch");
    if(freeze_master) {
      for(const auto& saved:master) {
        TORCH_CHECK(saved.parameter.view(torch::kInt32).equal(saved.value.view(torch::kInt32)),
                    "Frozen LoD changed a Master attribute");
        TORCH_CHECK(!saved.parameter.grad().defined(),"Frozen Master gradient must stay undefined");
        TORCH_CHECK(static_cast<const torch::optim::AdamOptions&>(gaussians_->optimizer_->param_groups()[saved.group].options()).lr()==saved.learning_rate,
                    "Frozen LoD changed a Master learning rate");
        auto it=gaussians_->optimizer_->state().find(saved.parameter.unsafeGetTensorImpl());
        TORCH_CHECK(saved.has_state==(it!=gaussians_->optimizer_->state().end()),"Frozen LoD created Master Adam state");
        if(saved.has_state) {
          const auto& state=static_cast<const torch::optim::AdamParamState&>(*it->second);
          TORCH_CHECK(state.step()==saved.step&&state.exp_avg().view(torch::kInt32).equal(saved.mean.view(torch::kInt32))&&
              state.exp_avg_sq().view(torch::kInt32).equal(saved.variance.view(torch::kInt32)),"Frozen LoD changed Master Adam moments/clock");
        }
      }
    } else TORCH_CHECK(adam_step()==full_clock+8&&!before.equal(gaussians_->features_dc_),"Joint Master Adam did not continue");
    TORCH_CHECK(flexgs_lod_master_updates_==(freeze_master?0:8)&&flexgs_lod_selector_updates_==8&&flexgs_lod_nonfinite_==0,
                "LoD steps missing or nonfinite");
    TORCH_CHECK(topology==topology_version_&&N==gaussians_->getXYZ().size(0)&&owner.equal(bank->rows())&&counts.equal(bank->memberCounts()),
                "LoD changed Anchor/Gaussian topology");
    auto tables=bank->compensationFeatureTables();const std::vector<int> expected{8,6,4,2};
    for(int j=0;j<4;++j){auto it=gaussians_->optimizer_->state().find(tables[j].unsafeGetTensorImpl());
      TORCH_CHECK(it!=gaussians_->optimizer_->state().end(),"Feature missing Adam state");
      TORCH_CHECK(static_cast<const torch::optim::AdamParamState&>(*it->second).step()==expected[j],"Inactive residual Adam clock");}
    bool selector_changed=false,compensation_changed=false,feature_changed=false;index=0;
    for(const auto& p:selector_network_->named_parameters()){
      const bool changed=!p.value().equal(untouched[index++]);
      if(p.key().rfind("local_emd.",0)==0||p.key().rfind("soft_net.",0)==0)selector_changed|=changed;
      if(p.key().rfind("spatial_compensation.",0)==0)compensation_changed|=changed;
      if(p.key().find("compensation_features_")!=std::string::npos)feature_changed|=changed;
    }
    TORCH_CHECK(selector_changed&&compensation_changed&&feature_changed,"Missing trainable branch updates");
    savePly(result_dir_/"verified/ply");
    auto loaded=improvements::selector::GumbelNetwork(.1,true,false,true,std::vector<double>{},0);
    loaded->to(torch::kCUDA);torch::serialize::InputArchive archive;
    archive.load_from((result_dir_/"verified/ply/selector.pt").string(),torch::Device(torch::kCUDA));loaded->load(archive);
    TORCH_CHECK(loaded->anchors()->compensationFeatureTables().size()==4,"Four-node checkpoint restore");
    auto legacy=improvements::selector::GumbelNetwork(.1,true,false,true,std::vector<double>{},0);
    legacy->to(torch::kCUDA);legacy->anchors()->enableVariableMembers();
    legacy->enableSpatialCompensation(false,false,0,{.01,.05,.1,.2,.3,.4},true,false,true,false,48,1);
    torch::serialize::OutputArchive old;legacy->save(old);old.save_to((result_dir_/"legacy_six.pt").string());
    auto restored=improvements::selector::GumbelNetwork(.1,true,false,true,std::vector<double>{},0);
    restored->to(torch::kCUDA);torch::serialize::InputArchive old_in;
    old_in.load_from((result_dir_/"legacy_six.pt").string(),torch::Device(torch::kCUDA));restored->load(old_in);
    TORCH_CHECK(restored->anchors()->compensationFeatureTables().size()==6,"Legacy six-node checkpoint restore");
    std::cout<<"PASS RealLoD Full isolation, raw all-view GI, four-node checkpoint, "
             <<(freeze_master?"bitwise frozen Master and Adam":"joint Master Adam")
             <<", Fine Full renders="<<real_lod_full_renders_<<", network/Feature training, frozen topology\n";
  }
 private:
  void verifyFullGradientRouting(const std::shared_ptr<GaussianKeyframe>& camera) {
    renderer_workspace::Scope workspace(reuse_renderer_workspace_);
    ssim_backend::Scope ssim(optimized_ssim_backward_);
    ssim_backend::CacheScope ssim_cache(cached_ssim_statistics_);
    mapping_cuda::L1FusionScope l1(fused_l1_loss_);
    auto clear=[&] {gaussians_->optimizer_->zero_grad(true);selector_optimizer_->zero_grad(true);};
    std::vector<torch::Tensor> parameters;
    for(size_t group=0;group<6;++group)
      for(auto p:gaussians_->optimizer_->param_groups()[group].params())parameters.push_back(p);
    auto gradients=[&] {
      std::vector<torch::Tensor> result;
      for(auto p:parameters)result.push_back(p.grad().defined()?p.grad().detach().clone():torch::zeros_like(p));
      return result;
    };
    auto truth=camera->original_image_.to(device_type_);
    clear();
    const auto before=static_cast<const torch::optim::AdamParamState&>(
        *gaussians_->optimizer_->state().at(gaussians_->features_dc_.unsafeGetTensorImpl())).step();
    accumulateRealLodFullGradients(camera,truth);
    const auto full=gradients();
    double full_norm=0;
    for(const auto& g:full)full_norm+=g.abs().sum().item<double>();
    TORCH_CHECK(full_norm>0,"Full branch produced no Master gradient");
    for(auto p:selector_network_->parameters())
      TORCH_CHECK(!p.grad().defined(),"Full branch leaked into Selector/Compensation/Feature");
    clear();torch::manual_seed(12345);
    const auto lod=accumulateLodGradients(camera,32,40,truth,0,false,false,nullptr,false,true);
    TORCH_CHECK(lod.rendered&&lod.trained&&lod.finite_loss,"LoD gradient probe failed");
    const auto lod_gradient=gradients();
    clear();
    accumulateRealLodFullGradients(camera,truth);
    torch::manual_seed(12345);
    const auto joint=accumulateLodGradients(camera,32,40,truth,0,false,false,nullptr,false,true);
    TORCH_CHECK(joint.rendered&&joint.trained&&joint.finite_loss,"Joint gradient probe failed");
    const auto combined=gradients();
    double max_error=0;
    for(size_t i=0;i<parameters.size();++i) {
      auto expected=full[i]+lod_gradient[i];
      max_error=std::max(max_error,(combined[i]-expected).abs().max().item<double>());
      // Repeated rasterizer VJPs use atomics; this verifies accumulation, not bitwise parity.
      TORCH_CHECK(torch::allclose(combined[i],expected,1e-4,1e-5),
                  "LoD backward cleared or corrupted the accumulated Full Master gradient");
    }
    TORCH_CHECK(static_cast<const torch::optim::AdamParamState&>(
        *gaussians_->optimizer_->state().at(gaussians_->features_dc_.unsafeGetTensorImpl())).step()==before,
        "A branch submitted Adam before the joint step");
    clear();
    real_lod_full_renders_=real_lod_full_backwards_=0; // Probes are outside the eight scheduled Fine updates.
    std::cout<<"PASS Full-only gradient isolation and Full+LoD VJP accumulation; max_abs_error="<<max_error<<'\n';
  }
};

int main(int argc,char** argv) {
  if(!torch::cuda::is_available())return 77;
  torch::set_num_threads(2);at::globalContext().setAllowTF32CuBLAS(false);
  const auto out=argc>1?std::filesystem::path(argv[1]):std::filesystem::temp_directory_path()/"cartgs_real_lod_test";
  const std::string mode=argc>2?argv[2]:"joint";
  RealLodMapperTest mapper(out);mapper.verify(mode=="frozen",
      !(argc>3&&std::string(argv[3])=="legacy"),mode!="joint_lod_only",mode=="nonfinite");
}

#include "include/gaussian_mapper.h"
#include "include/anchor_frustum.h"
#include <iostream>

class ViewInputMapperTest:public GaussianMapper {
 public:
  ViewInputMapperTest(const std::filesystem::path& output,const std::string& variant=""):GaussianMapper(nullptr,
      std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/"cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml",output,7,torch::kCUDA,true,
      std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/(variant=="visible_gi"||variant=="visible_empty"?
          "cfg/selector/spatial_offset3_residual48_view_input_visible_gi.yaml":variant=="visible_photo"?
          "cfg/selector/spatial_offset3_residual48_view_input_visible_no_view_gi.yaml":variant=="mid_gi"?
          "cfg/selector/spatial_offset3_residual48_view_input_mid3_gi.yaml":variant=="low_gi"?
          "cfg/selector/spatial_offset3_residual48_view_input_low_gi.yaml":
          "cfg/selector/spatial_offset3_residual48_view_input.yaml")) {
    TORCH_CHECK((variant=="mid_gi")==!selector_view_ratio_indices_.empty(),"View ratio config was not loaded");
    if(variant=="mid_gi")TORCH_CHECK(selector_view_ratio_indices_==std::vector<std::size_t>({1,2,3}),"Wrong view subset");
    TORCH_CHECK((variant.rfind("visible",0)==0)==selector_view_visible_budget_,"Visible budget config was not loaded");
    empty_view_=variant=="visible_empty";
  }
  void verify() {
    TORCH_CHECK(selector_view_input_enabled_&&!selector_view_residual_enabled_&&!mixed_ratio_config_.enabled&&!selector_anchor_frustum_culling_,"View mode configuration");
    auto o=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    std::map<point3D_id_t,Point3D> points;
    for(int i=0;i<80;++i) {
      Point3D p;p.xyz_=Eigen::Vector3d((i%10-5)*.025+.013,((i/10)%8-4)*.025+.017,2.02+(i%3)*.005);
      p.color_=Eigen::Vector3f(.4,.3,.2);points[i]=p;
      if(selector_view_visible_budget_&&(i>=40||empty_view_))points[i].xyz_.x()+=5.;
    }
    gaussians_->createFromPcd(points,1);gaussians_->trainingSetup(opt_params_);
    gaussians_->active_sh_degree_=default_sh_=1;scene_->cameras_extent_=1;
    opt_params_.densify_from_iter_=1000000;opt_params_.opacity_reset_interval_=0;
    do_gaus_pyramid_training_=false;initial_mapped_=true;SLAM_ended_=true;
    auto camera=std::make_shared<GaussianKeyframe>(0);
    camera->camera_id_=0;camera->setPose(Eigen::Quaterniond::Identity(),Eigen::Vector3d::Zero());
    camera->image_height_=32;camera->image_width_=40;camera->FoVx_=camera->FoVy_=1;
    camera->world_view_transform_=torch::eye(4,o);
    camera->full_proj_transform_=camera->getProjectionMatrix(.01f,100.f,1.f,1.f,torch::kCUDA).transpose(0,1).contiguous();
    camera->camera_center_=torch::zeros({3},o);camera->original_image_=torch::full({3,32,40},.1,o);
    scene_->addKeyframe(camera,&kfid_shuffled_);
    real_lod_.online_steps=24;real_lod_.phase=RealLodSchedule::Phase::TailFull;real_lod_.beginLod();
    const auto owners=selector_network_->anchors()->rows().clone();
    const auto visible_n=anchorNominalFrustumMask(selector_network_->anchors()->centers(),
        selector_network_->anchors()->offsetScales(),camera->full_proj_transform_).index_select(0,owners).sum().item<int64_t>();
    if(selector_view_visible_budget_)TORCH_CHECK(visible_n<owners.numel()&&
        (empty_view_?visible_n==0:visible_n>0),"Fixture must exercise a reduced or empty visible population");
    std::uint64_t global_count=0,view_count=0;
    for(int step=0;step<24;++step) {
      const bool view_round=real_lod_.viewResidualRound(selector_target_ratios_.size());
      std::map<std::string,torch::Tensor> before;
      for(auto p:selector_network_->named_parameters())before[p.key()]=p.value().detach().clone();
      trainRealLodIteration(camera);
      bool changed=false;
      for(auto p:selector_network_->named_parameters()) {
        const bool shared=p.key().rfind("local_emd.",0)==0||p.key().rfind("soft_net.",0)==0;
        if(shared)changed|=!p.value().equal(before[p.key()]);
      }
      TORCH_CHECK(changed,"Shared Selector did not update in one mode");
      if(view_round) {++view_count;TORCH_CHECK(view_importance_.defined()&&!view_importance_.requires_grad(),"Missing current Full GI");}
      else {++global_count;TORCH_CHECK(!view_importance_.defined(),"Global round reused view GI");}
    }
    TORCH_CHECK(global_count==12&&view_count==12&&real_lod_full_renders_==24&&real_lod_full_backwards_==24&&
        flexgs_lod_master_updates_==24&&flexgs_lod_selector_updates_==24&&flexgs_lod_nonfinite_==0,"Exposure/commit mismatch");
    for(auto p:selector_network_->named_parameters()) {
      if(p.key().rfind("local_emd.",0)!=0&&p.key().rfind("soft_net.",0)!=0)continue;
      auto found=gaussians_->optimizer_->state().find(p.value().unsafeGetTensorImpl());
      TORCH_CHECK(found!=gaussians_->optimizer_->state().end(),"Missing branch Adam state");
      TORCH_CHECK(static_cast<const torch::optim::AdamParamState&>(*found->second).step()==24,"Shared Adam must update on both modes");
    }
    TORCH_CHECK(owners.equal(selector_network_->anchors()->rows()),"View training changed topology");
    if(!selector_view_ratio_indices_.empty()) {
      TORCH_CHECK(global_view_ratio_counts_==std::vector<std::uint64_t>({2,2,2,2,2,2})&&
          residual_view_ratio_counts_==std::vector<std::uint64_t>({0,4,4,4,0,0}),"View subset exposure mismatch");
    }
    for(const auto& file:{"view_input_steps.jsonl"}) {
      std::ifstream input(result_dir_/file);std::string line;int count=0;
      while(std::getline(input,line)) {
        Json::Value value;Json::Reader reader;TORCH_CHECK(reader.parse(line,value),"Invalid GI trace");
        TORCH_CHECK(value["label_K"].asInt64()==std::min(value["requested_label_K"].asInt64(),value["eligible_N"].asInt64()),"GI label budget mismatch");
        TORCH_CHECK(value["gi_supervision"].asString()=="deterministic_logit_BCE_exact_K","Wrong GI loss");
        const bool view=value["round"].asString()=="view_input";
        const bool visible_budget=selector_view_visible_budget_&&view;
        TORCH_CHECK(value["visible_point_budget"].asBool()==visible_budget&&
            value["candidate_N"].asInt64()==owners.numel()&&!value["anchor_frustum_culling"].asBool(),
            "Visible budget must not cull training candidates");
        TORCH_CHECK(value["budget_N"].asInt64()==(visible_budget?visible_n:owners.numel()),"Wrong budget population");
        if(visible_budget) {
          const auto requested=static_cast<int64_t>(std::floor(value["budget_ratio"].asDouble()*visible_n));
          TORCH_CHECK(value["requested_label_K"].asInt64()==requested&&value["budget_target_K"].asInt64()==requested&&
              value["groups"][0]["target_K"].asInt64()==requested,"GI and Budget Loss target K differ");
        }
        if(!selector_view_ratio_indices_.empty()) {
          auto slot=(value["step"].asUInt64()-1)%6;
          auto expected=value["round"].asString()=="view_input"?std::array<std::size_t,3>{1,2,3}[slot%3]:slot;
          TORCH_CHECK(std::abs(value["target_ratio"].asDouble()-selector_target_ratios_[expected])<1e-6&&
              std::abs(value["groups"][0]["ratio"].asDouble()-selector_target_ratios_[expected])<1e-6,
              "Actual Selector/Compensator ratio differs from the requested schedule");
        }
        const bool high_view=selector_view_gi_lowest_two_only_&&value["round"].asString()=="view_input"&&
            value["target_ratio"].asDouble()>.05+1e-6;
        const double expected_gi=view?(high_view?0.:selector_view_gi_weight_):selector_gi_loss_weight_;
        TORCH_CHECK(value["gi_weight"].asDouble()==expected_gi,"Wrong per-ratio view GI weight");
        if(expected_gi==0)TORCH_CHECK(value["gi_loss"].asDouble()==0,"GI-off view must use only Photo/Budget");
        ++count;
      }
      TORCH_CHECK(count==24,"Missing GI trace steps");
    }
    savePly(result_dir_/"checkpoint");saveRealLodState(result_dir_);
    std::cout<<"PASS ViewInput Mapping: 24 Full+LoD steps/24 Master commits, 12 global+12 view, shared24 Adam, current Full GI, full-map candidates, topology, checkpoint\n";
  }
 private:
  bool empty_view_=false;
};
int main(int argc,char** argv) {
  if(!torch::cuda::is_available())return 77;
  torch::set_num_threads(2);at::globalContext().setAllowTF32CuBLAS(false);
  ViewInputMapperTest mapper(argc>1?argv[1]:"/tmp/cartgs_view_input_mapping",argc>2?argv[2]:"");mapper.verify();
}

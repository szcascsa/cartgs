#include "include/gaussian_mapper.h"
#include <iostream>

class ViewMapperTest:public GaussianMapper {
 public:
  ViewMapperTest(const std::filesystem::path& output):GaussianMapper(nullptr,
      std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/"cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml",output,7,torch::kCUDA,true,
      std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/"cfg/selector/spatial_offset3_residual48_view_residual.yaml") {}
  void verify() {
    TORCH_CHECK(selector_view_residual_enabled_&&!mixed_ratio_config_.enabled&&!selector_anchor_frustum_culling_,"View mode configuration");
    auto o=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    std::map<point3D_id_t,Point3D> points;
    for(int i=0;i<80;++i) {
      Point3D p;p.xyz_=Eigen::Vector3d((i%10-5)*.025+.013,((i/10)%8-4)*.025+.017,2.02+(i%3)*.005);
      p.color_=Eigen::Vector3f(.4,.3,.2);points[i]=p;
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
    std::uint64_t global_count=0,view_count=0;
    for(int step=0;step<24;++step) {
      const bool view_round=real_lod_.viewResidualRound(selector_target_ratios_.size());
      std::map<std::string,torch::Tensor> before;
      for(auto p:selector_network_->named_parameters())before[p.key()]=p.value().detach().clone();
      trainRealLodIteration(camera);
      bool branch_changed=false;
      for(auto p:selector_network_->named_parameters()) {
        const bool branch=p.key().rfind("selector_view_residual.",0)==0;
        const bool global=p.key().rfind("local_emd.",0)==0||p.key().rfind("soft_net.",0)==0;
        if(view_round&&global)TORCH_CHECK(p.value().equal(before[p.key()]),"View round changed global Selector/Adam momentum");
        if(!view_round&&branch)TORCH_CHECK(p.value().equal(before[p.key()]),"Global round changed residual/Adam momentum");
        if(branch)branch_changed|=!p.value().equal(before[p.key()]);
      }
      if(view_round) {
        ++view_count;TORCH_CHECK(branch_changed&&view_importance_.defined()&&!view_importance_.requires_grad(),"View round lacks updates/current Full GI");
      } else {++global_count;TORCH_CHECK(!view_importance_.defined(),"Global round reused stale single-view GI");}
    }
    TORCH_CHECK(global_count==12&&view_count==12&&real_lod_full_renders_==24&&real_lod_full_backwards_==24&&
        flexgs_lod_master_updates_==24&&flexgs_lod_selector_updates_==24&&flexgs_lod_nonfinite_==0,"Exposure/commit mismatch");
    for(auto p:selector_network_->named_parameters()) {
      if(p.key().rfind("selector_view_residual.",0)!=0&&p.key().rfind("local_emd.",0)!=0&&p.key().rfind("soft_net.",0)!=0)continue;
      auto found=gaussians_->optimizer_->state().find(p.value().unsafeGetTensorImpl());
      TORCH_CHECK(found!=gaussians_->optimizer_->state().end(),"Missing branch Adam state");
      TORCH_CHECK(static_cast<const torch::optim::AdamParamState&>(*found->second).step()==12,"Branch Adam advanced outside its round");
    }
    TORCH_CHECK(owners.equal(selector_network_->anchors()->rows()),"View training changed topology");
    savePly(result_dir_/"checkpoint");saveRealLodState(result_dir_);
    std::cout<<"PASS ViewResidual Mapping: 24 Full+LoD steps/24 Master commits, 12 global+12 view, branch Adam isolation, current Full GI, full-map candidates, topology, checkpoint\n";
  }
};
int main(int argc,char** argv) {
  if(!torch::cuda::is_available())return 77;
  torch::set_num_threads(2);at::globalContext().setAllowTF32CuBLAS(false);
  ViewMapperTest mapper(argc>1?argv[1]:"/tmp/cartgs_view_mapping");mapper.verify();
}

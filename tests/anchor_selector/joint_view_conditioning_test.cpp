#include "include/gaussian_mapper.h"
#include "include/view_budget_training.h"
#include "include/local_importance.h"
#include <iostream>

using namespace improvements::selector;

void unit(torch::Device device,const std::filesystem::path& out,int64_t mode=2) {
  auto o=torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto net=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);net->to(device);
  net->anchors()->enableVariableMembers();
  net->enableSpatialCompensation(false,false,0,{.01,.05,.1,.3,.5,.7},true,false,true,false,48,1);
  auto spatial=net->spatialCompensation();spatial->enablePhysicalScaleAdditive();
  std::map<std::string,torch::Tensor> before;int64_t original=0;
  for(auto p:net->named_parameters()){before[p.key()]=p.value().detach().clone();original+=p.value().numel();}
  auto rng=at::globalContext().defaultGenerator(torch::Device(torch::kCPU)).get_state().clone();
  net->enableJointViewConditioning(mode);
  TORCH_CHECK(rng.equal(at::globalContext().defaultGenerator(torch::Device(torch::kCPU)).get_state()),"Conditioning consumed initialization RNG");
  int64_t count=0;auto parameters=net->named_parameters();
  for(auto p:parameters) {
    count+=p.value().numel();
    if(before.count(p.key()))TORCH_CHECK((p.key()=="local_emd.0.weight"?p.value().slice(1,0,11):p.value()).equal(before[p.key()]),"Old weight changed");
  }
  TORCH_CHECK(count-original==768&&spatial->viewInput(),"Unexpected network capacity");
  auto select=[&](const torch::Tensor& packed) {
    auto h=torch::relu(torch::nn::functional::linear(packed,parameters["local_emd.0.weight"],parameters["local_emd.0.bias"]));
    h=torch::relu(torch::nn::functional::linear(h,parameters["local_emd.2.weight"],parameters["local_emd.2.bias"]));
    return torch::nn::functional::linear(h,parameters["soft_net.weight"],parameters["soft_net.bias"]);
  };
  auto x=torch::randn({17,3},o).requires_grad_(true),center=torch::zeros({17,3},o),reference=torch::ones({17,1},o);
  auto scale=torch::zeros({17,3},o),rotation=torch::zeros({17,4},o),dc=torch::zeros({17,1,3},o),opacity=torch::zeros({17,1},o);
  rotation.select(1,0).fill_(1);auto feature=torch::randn({17,48},o).requires_grad_(true);
  spatial->initialize(x,scale,rotation,dc,opacity,center,reference,true);net->initializeSelectorViewInput(x);
  {torch::NoGradGuard guard;for(const auto& name:{"xyz","scale","rotation","other"})
    spatial->heads->at<torch::nn::SequentialImpl>(name).at<torch::nn::LinearImpl>(2).weight.fill_(.02);}
  auto forward=[&](double ratio,const torch::Tensor& condition) {
    return spatial->forward(x,scale,rotation,dc,opacity,center,reference,ratio,true,feature,{},condition);
  };
  auto code=torch::randn({17,7},o).requires_grad_(true),condition=net->packCompensationViewInput(code);
  auto selector_code=net->packSelectorViewInput(torch::zeros({17,11},o),code).slice(1,11,18);
  TORCH_CHECK(selector_code.equal(mode==4?torch::zeros_like(code):code.detach()),"Selector cross condition misrouted");
  TORCH_CHECK(condition.slice(1,0,7).equal(mode==3?torch::zeros_like(code):code.detach()),"Compensation cross condition misrouted");
  for(double r:{.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7})
    TORCH_CHECK(forward(r,{}).equal(forward(r,condition)),"Zero view projection changed Compensation");
  auto input=torch::randn({17,11},o).requires_grad_(true);
  auto initial_global=select(net->packSelectorViewInput(input)),initial_local=select(net->packSelectorViewInput(input,code));
  TORCH_CHECK(initial_global.equal(initial_local),"Zero view columns changed Selector modes");
  {torch::NoGradGuard guard;spatial->view_projection->weight.normal_(0,.03);
    parameters["local_emd.0.weight"].slice(1,11,19).normal_(0,.03);
    parameters["soft_net.weight"].normal_(0,.1);}
  auto global=forward(.05,{}),local=forward(.05,condition);
  TORCH_CHECK(!global.equal(local),"Compensation ignores view condition");
  net->zero_grad(true);global.square().mean().backward();
  TORCH_CHECK(spatial->view_projection->weight.grad().defined()&&spatial->view_projection->weight.grad().abs().sum().item<float>()==0,
      "Global zero condition must preserve a zero gradient and Adam momentum updates");
  net->zero_grad(true);x.mutable_grad()=torch::Tensor{};feature.mutable_grad()=torch::Tensor{};
  (forward(.05,condition).square().mean()+select(net->packSelectorViewInput(input,code)).square().mean()).backward();
  TORCH_CHECK(!code.grad().defined()&&x.grad().defined()&&feature.grad().defined()&&input.grad().defined(),"View detach or original gradient paths broken");
  TORCH_CHECK(spatial->view_projection->weight.grad().abs().sum().item<double>()>0&&
      parameters["local_emd.0.weight"].grad().slice(1,11,19).abs().sum().item<double>()>0,"New conditions cannot train");
  if(mode==3)TORCH_CHECK(spatial->view_projection->weight.grad().slice(1,0,7).count_nonzero().item<int64_t>()==0,
      "Mode-only Compensation received a camera gradient");
  if(mode==4)TORCH_CHECK(parameters["local_emd.0.weight"].grad().slice(1,11,18).count_nonzero().item<int64_t>()==0,
      "Mode-only Selector received a camera gradient");
  for(int n:{0,1,31,2048}) {
    auto packed=net->packSelectorViewInput(torch::randn({n,11},o),torch::randn({n,7},o));
    TORCH_CHECK(select(packed).sizes()==torch::IntArrayRef({n,2}),"Dynamic/empty Selector shape");
    if(device.is_cuda()) {
      auto fast=mapping_cuda::flexgsMLP(packed,parameters["local_emd.0.weight"],parameters["local_emd.0.bias"],
          parameters["local_emd.2.weight"],parameters["local_emd.2.bias"],parameters["soft_net.weight"],parameters["soft_net.bias"]);
      TORCH_CHECK(torch::allclose(fast[0],select(packed),3e-5,3e-6),"Fused 19D mismatch");
    }
  }
  std::filesystem::create_directories(out);torch::save(net,(out/"selector.pt").string());
  auto restored=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);restored->to(device);
  torch::serialize::InputArchive archive;archive.load_from((out/"selector.pt").string(),device);restored->load(archive);
  TORCH_CHECK(restored->jointViewConditioning()==mode&&restored->spatialCompensation()->viewInput(),"Conditioning schema restore");
  auto frozen=restored->cloneFrozen();
  for(auto p:net->named_parameters())TORCH_CHECK(restored->named_parameters()[p.key()].equal(p.value())&&
      frozen->named_parameters()[p.key()].equal(p.value())&&!frozen->named_parameters()[p.key()].requires_grad(),"Reload/frozen weight mismatch");
  bool rejected=false;try{restored->enableJointViewConditioning(1);}catch(const c10::Error&){rejected=true;}
  TORCH_CHECK(rejected,"Schema mismatch silently accepted");
  torch::serialize::OutputArchive fixture;
  for(auto item:std::vector<std::pair<std::string,torch::Tensor>>{{"xyz",x},{"scale",scale},{"rotation",rotation},{"dc",dc},{"opacity",opacity},
      {"center",center},{"reference",reference},{"feature",feature},{"condition",condition},{"view_code",code},{"input11",input},
      {"comp_global",forward(.05,{})},{"comp_local",forward(.05,condition)},{"selector_global",select(net->packSelectorViewInput(input))},
      {"selector_local",select(net->packSelectorViewInput(input,code))}})fixture.write(item.first,item.second.detach(),true);
  fixture.save_to((out/"fixture.pt").string());
  std::cout<<"PASS joint conditioning "<<device<<" mode="<<mode<<": 768 parameters, RNG, initial identity, view detach, original VJP, dynamic K, ratio interpolation, checkpoint/frozen\n";
}

class JointViewMapperTest:public GaussianMapper {
 public:
  JointViewMapperTest(const std::filesystem::path& out,const std::string& mode):GaussianMapper(nullptr,
      std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/"cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml",out,7,torch::kCUDA,true,
      std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/("cfg/selector/spatial_offset3_residual48_joint_"+mode+".yaml")){}
  void verify(bool empty,bool isolation=false) {
    auto o=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    TORCH_CHECK(view_budget_conditioning_>0&&!selector_view_input_enabled_&&!view_budget_view_residual_&&
        (view_budget_li_weight_==0||view_budget_li_weight_==.1f),"Wrong joint controls");
    std::map<point3D_id_t,Point3D> points;
    for(int i=0;i<80;++i) {Point3D p;p.xyz_=Eigen::Vector3d((i%10-5)*.025+.013+(empty||i>=40?5.:0.),((i/10)%8-4)*.025+.017,2.02+(i%3)*.005);
      p.color_=Eigen::Vector3f(.4,.3,.2);points[i]=p;}
    gaussians_->createFromPcd(points,1);gaussians_->trainingSetup(opt_params_);gaussians_->active_sh_degree_=default_sh_=1;scene_->cameras_extent_=1;
    opt_params_.densify_from_iter_=1000000;opt_params_.opacity_reset_interval_=0;do_gaus_pyramid_training_=false;initial_mapped_=true;SLAM_ended_=true;
    keyframe_record_interval_=all_keyframes_record_interval_=training_report_interval_=0;
    auto camera=std::make_shared<GaussianKeyframe>(0);camera->camera_id_=0;camera->setPose(Eigen::Quaterniond::Identity(),Eigen::Vector3d::Zero());
    camera->image_height_=32;camera->image_width_=40;camera->FoVx_=camera->FoVy_=1;camera->znear_=.01f;camera->world_view_transform_=torch::eye(4,o);
    camera->full_proj_transform_=camera->getProjectionMatrix(.01f,100.f,1.f,1.f,torch::kCUDA).transpose(0,1).contiguous();
    camera->camera_center_=torch::zeros({3},o);camera->original_image_=torch::full({3,32,40},.1,o);undistort_mask_[0]=torch::ones({32,40},o);scene_->addKeyframe(camera,&kfid_shuffled_);
    real_lod_.online_steps=12;real_lod_.phase=RealLodSchedule::Phase::TailFull;real_lod_.beginLod();
    if(isolation) {
      TORCH_CHECK(view_budget_li_weight_==.1f&&!empty,"LI isolation requires nonempty LI configuration");
      selector_network_->initializeSelectorViewInput(gaussians_->getXYZ());
      auto& anchors=selector_network_->anchors();
      auto code=anchorViewCode(anchors->centers(),anchors->offsetScales(),camera->camera_center_,camera->world_view_transform_,
          camera->full_proj_transform_,32,40,.01f,selector_network_->selectorViewSceneScale()).index_select(0,anchors->rows());
      {torch::NoGradGuard guard;selector_network_->named_parameters()["soft_net.weight"].normal_(0,.1);}
      auto input=selector_network_->forwardFlexgsCUDA(gaussians_->getXYZ(),.1f,gaussians_->getScalingActivation(),
          gaussians_->features_dc_,gaussians_->getOpacityActivation(),{},{},code);
      auto eligible=torch::ones({gaussians_->xyz_.size(0)},o.dtype(torch::kBool));
      auto sampled=viewBudgetSelection(input.logits,eligible,torch::ones_like(input.logits),1.);
      auto labels=torch::zeros_like(sampled.mask);labels.slice(0,0,12).fill_(1);
      auto loss=.1*localImportanceLoss(sampled.mask,labels);
      std::vector<torch::Tensor> mlp;
      for(const auto& p:selector_network_->named_parameters())
        if(p.key().rfind("local_emd.",0)==0||p.key().rfind("soft_net.",0)==0)mlp.push_back(p.value());
      TORCH_CHECK(mlp.size()==6,"Wrong conditional Selector gradient scope");
      auto expected=torch::autograd::grad({loss},mlp,{},true);
      auto all=selector_network_->parameters();
      for(auto p:{gaussians_->xyz_,gaussians_->scaling_,gaussians_->rotation_,gaussians_->opacity_,gaussians_->features_dc_,gaussians_->features_rest_})all.push_back(p);
      {torch::NoGradGuard guard;for(auto p:all)p.mutable_grad()=torch::full_like(p,.125);}
      accumulateLocalImportanceGradients(loss,mlp);
      for(auto p:all) {
        size_t i=0;while(i<mlp.size()&&p.unsafeGetTensorImpl()!=mlp[i].unsafeGetTensorImpl())++i;
        auto target=i<mlp.size()?expected[i]+.125:torch::full_like(p,.125);
        TORCH_CHECK(p.grad().contiguous().reshape({-1}).view(torch::kUInt8).equal(target.contiguous().reshape({-1}).view(torch::kUInt8)),
            "LI changed a non-Selector gradient or failed additive accumulation");
      }
      TORCH_CHECK(!code.grad().defined(),"LI view code must be detached");
      std::cout<<"PASS conditional LI VJP mode="<<view_budget_conditioning_
          <<": six Selector gradients added, all Master/Compensation/Feature gradients bitwise intact\n";return;
    }
    auto owners=selector_network_->anchors()->rows().clone();auto topology=topology_version_;
    auto projection=selector_network_->spatialCompensation()->view_projection->weight;
    while(real_lod_.optimizingLod())trainRealLodIteration(camera);
    TORCH_CHECK(real_lod_full_renders_==12&&view_budget_renders_==(empty?0:12)&&flexgs_lod_master_updates_==12&&flexgs_lod_selector_updates_==12&&
        topology==topology_version_&&owners.equal(selector_network_->anchors()->rows())&&
        view_budget_li_backwards_==((!empty&&view_budget_li_weight_>0)?12:0),"Exposure/topology/LI changed");
    TORCH_CHECK(view_budget_li_full_renders_==(view_budget_li_weight_>0?12:0),"LI collection must reuse existing Full renders");
    auto it=gaussians_->optimizer_->state().find(projection.unsafeGetTensorImpl());
    TORCH_CHECK(it!=gaussians_->optimizer_->state().end()&&
        static_cast<const torch::optim::AdamParamState&>(*it->second).step()==12,"Conditional Adam clock mismatch");
    if(!empty)TORCH_CHECK(projection.abs().sum().item<float>()>0,"Compensation condition did not learn");
    if(view_budget_conditioning_==1||view_budget_conditioning_==3)
      TORCH_CHECK(projection.slice(1,0,7).abs().sum().item<float>()==0,"Mode-only Compensation learned view columns");
    auto selector_columns=selector_network_->named_parameters()["local_emd.0.weight"].slice(1,11,18);
    if(view_budget_conditioning_==1||view_budget_conditioning_==4)
      TORCH_CHECK(selector_columns.count_nonzero().item<int64_t>()==0,"Mode-only Selector learned view columns");
    if(!empty&&(view_budget_conditioning_==2||view_budget_conditioning_==3))
      TORCH_CHECK(selector_columns.count_nonzero().item<int64_t>()>0,"Real-view Selector did not learn");
    if(!empty&&(view_budget_conditioning_==2||view_budget_conditioning_==4))
      TORCH_CHECK(projection.slice(1,0,7).count_nonzero().item<int64_t>()>0,"Real-view Compensation did not learn");
    savePly(result_dir_/"checkpoint");saveRealLodState(result_dir_);
    std::cout<<"PASS joint Mapping mode="<<view_budget_conditioning_<<" empty="<<empty<<" LI="<<view_budget_li_weight_
        <<": three branches, one Adam, unchanged topology, correct LI routing, learned condition\n";
  }
};

int main(int argc,char**argv) {
  torch::set_num_threads(2);torch::manual_seed(7);at::globalContext().setAllowTF32CuBLAS(false);
  std::string mode=argc>1?argv[1]:"unit";std::filesystem::path out=argc>2?argv[2]:"/tmp/cartgs_joint_view";
  if(mode=="unit") {unit(torch::kCPU,out/"cpu");if(torch::cuda::is_available())unit(torch::kCUDA,out/"cuda");return 0;}
  if(mode=="cross_unit") {for(int64_t schema:{3,4}) {
    unit(torch::kCPU,out/std::to_string(schema)/"cpu",schema);
    if(torch::cuda::is_available())unit(torch::kCUDA,out/std::to_string(schema)/"cuda",schema);
  }return 0;}
  if(!torch::cuda::is_available())return 77;
  const bool isolation=mode.size()>10&&mode.substr(mode.size()-10)=="_isolation";
  if(isolation)mode=mode.substr(0,mode.size()-10);
  const bool empty=mode.size()>6&&mode.substr(mode.size()-6)=="_empty";
  JointViewMapperTest mapper(out,empty?mode.substr(0,mode.size()-6):mode);mapper.verify(empty,isolation);
}

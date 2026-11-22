#include "include/gaussian_mapper.h"
#include "ORB-SLAM3/include/Settings.h"
#include <json/json.h>
#include <cstdlib>

namespace fs=std::filesystem;
Json::Value readJson(const fs::path& path) {
  std::ifstream in(path);TORCH_CHECK(in.good(),"Missing JSON: ",path.string());
  Json::Value value;in>>value;return value;
}

// Repack only the untrained ratio-feature bank for six knots; every shared
// network tensor and Anchor membership is read from the selected Full export.
void convertArchive(torch::nn::Module& target,torch::serialize::InputArchive& source,
                    torch::serialize::OutputArchive& out,const std::string& path="") {
  for(auto& item:target.named_parameters(false)) {
    torch::Tensor value;
    if(path=="anchors." && item.key().find("compensation_features_")==0) {
      source.read("compensation_features_0",value);
      if(item.key()!="compensation_features_0")value=torch::zeros_like(value);
    } else {
      source.read(item.key(),value);
      if(path=="local_emd.0."&&item.key()=="weight"&&value.size(1)==11&&item.value().size(1)==19) {
        auto expanded=torch::zeros_like(item.value().detach().cpu());expanded.slice(1,0,11).copy_(value);value=expanded;
      }
    }
    out.write(item.key(),value,false);
  }
  for(auto& item:target.named_buffers(false)) {
    torch::Tensor value;
    if((path=="anchors."&&item.key()=="compensation_ratio_knots") ||
       (path=="spatial_compensation."&&item.key()=="ratio_encoding_bounds"))value=item.value().detach().cpu();
    else if(path=="spatial_compensation."&&item.key()=="view_input_schema"&&
            !source.try_read(item.key(),value,true))value=item.value().detach().cpu();
    else if(path=="spatial_compensation."&&item.key()=="geometry_head_schema"&&
            !source.try_read(item.key(),value,true))value=item.value().detach().cpu();
    else if(path.empty()&&(item.key()=="selector_view_residual_schema"||item.key()=="selector_view_input_schema"||
            item.key()=="selector_view_scene_scale"||item.key()=="selector_view_initialized"||item.key()=="joint_view_conditioning_schema")&&
            !source.try_read(item.key(),value,true))value=item.value().detach().cpu();
    else source.read(item.key(),value,true);
    out.write(item.key(),value,true);
  }
  for(auto& child:target.named_children()) {
    torch::serialize::InputArchive sub;
    if(((path.empty()&&child.key()=="selector_view_residual")||
        (path=="spatial_compensation."&&child.key()=="view_projection"))&&!source.try_read(child.key(),sub)) {
      torch::serialize::OutputArchive initial;child.value()->save(initial);out.write(child.key(),initial);
      continue; // New zero-initialized input/branch; preserve the old checkpoint.
    }
    source.read(child.key(),sub);
    torch::serialize::OutputArchive converted;
    convertArchive(*child.value(),sub,converted,path+child.key()+".");out.write(child.key(),converted);
  }
}

class StandaloneLod:public GaussianMapper {
 public:
  StandaloneLod(const fs::path& mapping,const fs::path& selector,const fs::path& output)
    :GaussianMapper(nullptr,mapping,output,0,torch::kCUDA,true,selector) {}

  void initialize(const Json::Value& manifest) {
    TORCH_CHECK(real_lod_.enabled&&!selector_detach_gaussian_parameters_&&real_lod_full_loss_weight_>0,
                "Standalone protocol requires existing Full+LoD joint Fine");
    const fs::path ply=manifest["full_ply"].asString();
    const auto full_state=readJson(ply/"real_lod_state.json");
    TORCH_CHECK(full_state["lod_steps"].asUInt64()==0,"Only untouched Full exports may initialize this stage");
    const auto original=readJson(manifest["anchor_state"].asString());
    for(const auto& item:original["compensation_feature_state"])
      TORCH_CHECK(item["adam_step"].asUInt64()==0,"Cannot remap trained residual knots");
    torch::serialize::InputArchive input;input.load_from((ply/"selector.pt").string(),torch::kCPU);
    torch::serialize::OutputArchive converted;
    convertArchive(*selector_network_,input,converted);
    converted.save_to((result_dir_/"converted_selector_init.pt").string());
    torch::serialize::InputArchive checkpoint;
    checkpoint.load_from((result_dir_/"converted_selector_init.pt").string(),torch::kCUDA);
    selector_network_->load(checkpoint);
    selector_network_->to(torch::kCUDA);
    selector_optimizer_=std::make_unique<torch::optim::Adam>(selector_network_->parameters(),
        torch::optim::AdamOptions(selector_learning_rate_));
    gaussians_->enableExplicitAnchors(selector_network_->anchors(),selector_optimizer_.get(),result_dir_/"anchor_pruning");
    loadPly(manifest["point_cloud"].asString());
    auto bank=selector_network_->anchors();bank->assertBlocks(gaussians_->getXYZ().size(0));
    TORCH_CHECK(bank->size()==original["anchor_count"].asInt64()&&
        gaussians_->getXYZ().size(0)==original["gaussian_count"].asInt64(),"Full map membership changed");
    // loadPly restores Master/metadata, not the per-member pruning window.
    // Preserve the saved window so graph-free export sees aligned slot tensors.
    torch::serialize::InputArchive pruning;
    pruning.load_from((fs::path(manifest["point_cloud"].asString()).parent_path()/"anchor_pruning_state.pt").string(),torch::kCUDA);
    for(auto entry:std::vector<std::pair<std::string,torch::Tensor>>{
        {"opacity_accum",gaussians_->anchorOpacityAccum()},
        {"anchor_visits",gaussians_->anchorVisits()},
        {"slot_opacity_accum",gaussians_->anchorSlotOpacityAccum()},
        {"slot_below_002_count",gaussians_->anchorSlotBelowPrune()},
        {"slot_below_cutoff_count",gaussians_->anchorSlotBelowCutoff()}}) {
      torch::Tensor value;pruning.read(entry.first,value,true);entry.second.set_data(value);
    }
    TORCH_CHECK(gaussians_->anchorSlotOpacityAccum().numel()==gaussians_->getXYZ().size(0),"Unaligned saved slot history");
    for(auto tensor:{gaussians_->xyz_,gaussians_->features_dc_,gaussians_->features_rest_,
                     gaussians_->opacity_,gaussians_->scaling_,gaussians_->rotation_})tensor.requires_grad_(true);
    gaussians_->Tensor_vec_xyz_={gaussians_->xyz_};gaussians_->Tensor_vec_feature_dc_={gaussians_->features_dc_};
    gaussians_->Tensor_vec_feature_rest_={gaussians_->features_rest_};gaussians_->Tensor_vec_opacity_={gaussians_->opacity_};
    gaussians_->Tensor_vec_scaling_={gaussians_->scaling_};gaussians_->Tensor_vec_rotation_={gaussians_->rotation_};
    gaussians_->sparse_points_xyz_=torch::empty({0,3},gaussians_->xyz_.options().requires_grad(false));
    gaussians_->sparse_points_color_=torch::empty_like(gaussians_->sparse_points_xyz_);
    loadViews(manifest);
    scene_->cameras_extent_=std::get<1>(scene_->getNerfppNorm());
    TORCH_CHECK(scene_->cameras_extent_>0,"Invalid camera spatial extent");
    gaussians_->spatial_lr_scale_=scene_->cameras_extent_;
    gaussians_->trainingSetup(opt_params_);
    real_lod_.online_steps=full_state["online_full_steps"].asUInt64();
    real_lod_.tail_steps=full_state["tail_full_steps"].asUInt64();
    real_lod_.phase=RealLodSchedule::Phase::TailFull;
    iteration_=manifest["initial_iteration"].asInt();mapping_iter_=iteration_;
    initial_mapped_=true;SLAM_ended_=true;
    real_lod_ratio_counts_.assign(selector_target_ratios_.size(),0);
    ensureSharedMappingOptimizer();
    gaussians_->savePly(result_dir_/"step0_master.ply");
    torch::save(selector_network_,(result_dir_/"step0_selector.pt").string());
    Json::Value report;report["gaussians"]=Json::Int64(gaussians_->xyz_.size(0));
    report["anchors"]=Json::Int64(bank->size());report["training_views"]=Json::UInt64(scene_->keyframes().size());
    report["lod_target_steps"]=Json::UInt64(real_lod_.online_steps);
    report["optimizer_reset"]=true;report["slam_resume"]=false;report["spatial_lr_scale"]=gaussians_->spatial_lr_scale_;
    report["spatial_lr_scale_source"]="radius of saved final training keyframe centers";
    report["view_reconstruction"]="original image loader/float conversion and ORB Settings stereo rectification";
    report["ratio_feature_conversion"]="untrained original Base copied bitwise; five zero residual tables; new ratio encoding bounds";
    for(auto r:selector_target_ratios_)report["ratios"].append(r);
    std::ofstream out(result_dir_/"initialization.json");out<<report;
    std::cout<<"[StandaloneLoD READY] N="<<gaussians_->xyz_.size(0)<<" A="<<bank->size()
             <<" views="<<scene_->keyframes().size()<<" fresh_adam=1 target="<<real_lod_.online_steps<<std::endl;
  }

  void execute(std::uint64_t smoke_steps) {
    const auto count=gaussians_->xyz_.size(0);
    const auto owners=selector_network_->anchors()->rows().clone();
    auto initial_master=gaussians_->xyz_.detach().clone();
    real_lod_.beginLod();
    if(smoke_steps)real_lod_.target_steps=smoke_steps;
    // Paired runs begin from the same camera and global-sampler RNG sequence.
    torch::manual_seed(0);std::srand(0);
    std::ofstream keyframes(result_dir_/"training_keyframes.csv");keyframes<<"step,keyframe_id\n";
    while(real_lod_.optimizingLod()) {
      auto view=useOneRandomKeyframe();TORCH_CHECK(view,"No training keyframe");
      keyframes<<real_lod_.lod_steps+1<<','<<view->fid_<<'\n';
      trainRealLodIteration(view);
      if(real_lod_.lod_steps%1000==0)saveRealLodState(result_dir_);
    }
    TORCH_CHECK(count==gaussians_->xyz_.size(0)&&torch::equal(owners,selector_network_->anchors()->rows()),"Fine mutated topology");
    TORCH_CHECK(real_lod_.lod_steps==real_lod_.target_steps&&flexgs_lod_nonfinite_==0,"Incomplete/nonfinite Fine");
    TORCH_CHECK(real_lod_full_renders_==real_lod_.lod_steps&&real_lod_full_backwards_==real_lod_.lod_steps,
                "Missing Full supervision");
    TORCH_CHECK(flexgs_lod_master_updates_==real_lod_.lod_steps&&flexgs_lod_selector_updates_==real_lod_.lod_steps,
                "Missing Master/Selector Adam commits");
    const auto final=result_dir_/(std::to_string(getIteration())+"_shutdown");
    fs::create_directories(final);
    // This harness saves checkpoints and numeric traces only; no render images.
    savePly(final/"ply");saveRealLodState(final/"ply");
    torch::save(*gaussians_->optimizer_,(final/"shared_optimizer.pt").string());
    Json::Value report;report["passed"]=true;report["steps"]=Json::UInt64(real_lod_.lod_steps);
    report["gaussians"]=Json::Int64(count);report["topology_unchanged"]=true;
    report["master_xyz_changed"]=!torch::equal(initial_master,gaussians_->xyz_.detach());
    report["full_renders"]=Json::UInt64(real_lod_full_renders_);report["master_updates"]=Json::UInt64(flexgs_lod_master_updates_);
    report["selector_updates"]=Json::UInt64(flexgs_lod_selector_updates_);
    report["view_renders"]=Json::UInt64(view_budget_renders_);
    report["view_backwards"]=Json::UInt64(view_budget_backwards_);
    std::ofstream out(result_dir_/"training_verified.json");out<<report;
  }

 private:
  void loadViews(const Json::Value& manifest) {
    const auto mode=manifest["mode"].asString(),dataset=manifest["dataset"].asString();
    const int sensor=mode=="mono"?ORB_SLAM3::System::MONOCULAR:
        (mode=="stereo"?ORB_SLAM3::System::STEREO:ORB_SLAM3::System::RGBD);
    ORB_SLAM3::Settings settings(manifest["camera_config"].asString(),sensor);
    const auto camera_model=settings.camera1();
    Camera camera;camera.camera_id_=0;camera.setModelId(Camera::CameraModelType::PINHOLE);
    camera.width_=settings.newImSize().width;camera.height_=settings.newImSize().height;
    camera.num_gaus_pyramid_sub_levels_=0;
    for(int k=0;k<4;++k)camera.params_[k]=camera_model->getParameter(k);
    scene_->addCamera(camera);
    const auto views=readJson(fs::path(manifest["full_ply"].asString())/"cameras.json");
    TORCH_CHECK(!views.empty(),"Empty saved training views");
    for(const auto& item:views) {
      TORCH_CHECK(item["width"].asInt()==int(camera.width_)&&item["height"].asInt()==int(camera.height_),"Camera dimensions mismatch");
      TORCH_CHECK(std::abs(item["fx"].asFloat()-camera.params_[0])<1e-3&&
                  std::abs(item["fy"].asFloat()-camera.params_[1])<1e-3,"Saved focal calibration mismatch");
      const auto name=item["img_name"].asString();
      fs::path image_path(name);
      if(image_path.is_relative()) {
        TORCH_CHECK(manifest.isMember("dataset_root"),"Relative training image requires dataset_root: ",name);
        image_path=fs::path(manifest["dataset_root"].asString())/image_path;
      }
      cv::Mat image=cv::imread(image_path.string(),cv::IMREAD_UNCHANGED);
      TORCH_CHECK(!image.empty(),"Missing training image ",name);
      if(dataset=="Replica"||dataset=="TUM_RGBD")cv::cvtColor(image,image,cv::COLOR_BGR2RGB);
      if(mode=="stereo"&&settings.needToRectify())cv::remap(image,image,settings.M1l(),settings.M2l(),cv::INTER_LINEAR);
      else if(image.cols!=int(camera.width_)||image.rows!=int(camera.height_))
        cv::resize(image,image,settings.newImSize());
      if(image.channels()==1)cv::cvtColor(image,image,cv::COLOR_GRAY2RGB);
      TORCH_CHECK(image.channels()==3,"Unsupported training image channels");
      double scale=image.depth()==CV_8U?1./255.:(image.depth()==CV_16U?1./65535.:1.);
      image.convertTo(image,CV_32FC3,scale);
      auto keyframe=std::make_shared<GaussianKeyframe>(item["id"].asUInt64(),0);
      keyframe->setCameraParams(camera);keyframe->img_filename_=name;keyframe->znear_=z_near_;keyframe->zfar_=z_far_;
      Eigen::Matrix3f rotation;Eigen::Vector3f position;
      for(int a=0;a<3;++a){position(a)=item["position"][a].asFloat();for(int b=0;b<3;++b)rotation(a,b)=item["rotation"][a][b].asFloat();}
      Eigen::Matrix3f w2c=rotation.transpose();
      keyframe->setPose(Eigen::Quaterniond(w2c.cast<double>()),(-w2c*position).cast<double>());
      keyframe->original_image_=tensor_utils::cvMat2TorchTensor_Float32(image,torch::kCPU);
      keyframe->computeTransformTensors();scene_->addKeyframe(keyframe,&kfid_shuffled_);
    }
  }
};

int main(int argc,char** argv) {
  TORCH_CHECK(argc>=2,"Usage: lod_from_full manifest.json [prepare|smoke_steps]");
  unsetenv("CARTGS_BATCH_FULL_TAIL_ONLY");torch::set_num_threads(2);
  const auto manifest=readJson(argv[1]);const fs::path output=manifest["output"].asString();fs::create_directories(output);
  StandaloneLod mapper(manifest["mapping_config"].asString(),manifest["selector_config"].asString(),output);
  mapper.initialize(manifest);
  if(argc>2&&std::string(argv[2])=="prepare")return 0;
  mapper.execute(argc>2?std::stoull(argv[2]):0);
}

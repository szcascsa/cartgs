#include "include/gaussian_mapper.h"
#include <cuda_runtime_api.h>
#include <nvToolsExt.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

class ContextReplay : public GaussianMapper {
 public:
  ContextReplay(const std::filesystem::path& out,const std::filesystem::path& config)
      :GaussianMapper(nullptr,std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/"cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml",
          out,0,torch::kCUDA,true,config) {}
  void run(const std::filesystem::path& out) {
    auto o=torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto& map=*gaussians_;std::map<point3D_id_t,Point3D> points;
    for(int i=0;i<2400;++i){Point3D p;
      p.xyz_=Eigen::Vector3d(i<1800?(i%30-15)*.05+.013:20+(i%30)*.1,((i/30)%30-15)*.05+.021,2.02+(i/900)*.025);
      p.color_=Eigen::Vector3f(.2+(i%7)*.05,.4,.3);points[i]=p;}
    map.createFromPcd(points,1);map.trainingSetup(opt_params_);map.active_sh_degree_=1;default_sh_=1;
    map.protection_.mapping_iteration=6000;iteration_=mapping_iter_=6000;opt_params_.iterations_=10000;
    map.protection_.maturity.copy_(torch::linspace(.1,.95,map.getXYZ().size(0),o));
    map.protection_.contribution_maturity.copy_(map.protection_.maturity*.6);
    scene_->cameras_extent_=1;selector_compensation_start_iter_=0;selector_replay_enabled_=false;gradient_constraint_config_.enabled=false;
    auto camera=std::make_shared<GaussianKeyframe>(0);
    camera->setPose(Eigen::Quaterniond::Identity(),Eigen::Vector3d::Zero());
    camera->image_height_=480;camera->image_width_=640;camera->FoVx_=camera->FoVy_=1;
    camera->world_view_transform_=torch::eye(4,o);
    camera->full_proj_transform_=camera->getProjectionMatrix(.01,100.,1.,1.,torch::kCUDA).transpose(0,1).contiguous();
    camera->camera_center_=torch::zeros({3},o);camera->original_image_=torch::full({3,480,640},.1,o);
    {torch::NoGradGuard guard;selector_network_->named_parameters()["soft_net.weight"].normal_(0,.02);
      selector_network_->named_parameters()["soft_net.bias"].copy_(torch::tensor({0.f,-1.f},o));
      selector_network_->named_parameters()["selector_context_residual.projection.weight"].normal_(0,.03);}
    initializeTransformFieldIfNeeded();ensureSharedMappingOptimizer();
    auto step=[&](bool fast,int r,int seed){
      cached_anchor_context_=fast;selector_network_->setCachedAnchorContext(fast);
      map.optimizer_->zero_grad(true);torch::manual_seed(seed);
      return accumulateLodGradients(camera,480,640,camera->original_image_,r,false,false);};
    auto bits=[](torch::Tensor a,torch::Tensor b){return a.contiguous().view(torch::kInt32).equal(b.contiguous().view(torch::kInt32));};
    selector_render_loss_weight_=0;selector_ratio_loss_weight_=1;
    for(int r=0;r<6;++r){
      auto original=step(false,r,800+r);std::vector<torch::Tensor> gradients;
      for(const auto& p:selector_network_->named_parameters())gradients.push_back(p.value().grad().defined()?p.value().grad().clone():torch::Tensor{});
      auto xgrad=map.xyz_.grad().defined()?map.xyz_.grad().clone():torch::Tensor{};
      auto fast=step(true,r,800+r);TORCH_CHECK(original.loss==fast.loss && original.active_gaussians==fast.active_gaussians,"Mapping loss/K");
      if(original.supervised_indices.defined())TORCH_CHECK(original.supervised_indices.equal(fast.supervised_indices),"Mapping supervised IDs");
      int j=0;for(const auto& p:selector_network_->named_parameters()){
        TORCH_CHECK(gradients[j].defined()==p.value().grad().defined(),"Mapping gradient presence");
        if(gradients[j].defined())TORCH_CHECK(bits(gradients[j],p.value().grad()),"Mapping bitwise gradient ",p.key());++j;}
      if(xgrad.defined())TORCH_CHECK(bits(xgrad,map.xyz_.grad()),"Mapping Master input gradient");
    }
    std::cout<<"PASS real accumulateLodGradients replay: frozen synthetic map, six ratios, same RNG/loss/K/IDs/dual VJP; Master and Selector gradients bitwise (photo weight zero in parity gate)\n";
    selector_render_loss_weight_=1;
    std::ofstream csv(out/"mapping_times.csv");csv<<"ratio,round,cached,N,K,gpu_ms,wall_ms\n"<<std::setprecision(9);
    for(int r:{0,1,2,5}) {
      for(int j=0;j<4;++j)step(j%2,r,901+j);
      for(int round=0;round<16;++round)for(int index=0;index<2;++index){
        const bool fast=(round%2)?index==0:index==1;
        cudaEvent_t start,end;cudaEventCreate(&start);cudaEventCreate(&end);cudaDeviceSynchronize();
        const auto before=std::chrono::steady_clock::now();cudaEventRecord(start);
        nvtxRangePushA(fast?"MappingReplay.cached":"MappingReplay.original");
        auto result=step(fast,r,1001+round);nvtxRangePop();cudaEventRecord(end);cudaEventSynchronize(end);
        const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
        float gpu;cudaEventElapsedTime(&gpu,start,end);cudaEventDestroy(start);cudaEventDestroy(end);
        TORCH_CHECK(result.finite_loss,"Mapping replay finite loss");
        csv<<selector_target_ratios_[r]<<','<<round<<','<<fast<<','<<map.getXYZ().size(0)<<','<<result.active_gaussians<<','<<gpu<<','<<wall<<'\n';
      }
    }
    std::cout<<"PASS matched 640x480 LoD Mapping replay timings (includes actual renderer/backward and maturity private VJP; no Full/Tracking/Adam/topology)\n";
  }
};
int main(int argc,char**argv) {
  if(!torch::cuda::is_available())return 77;
  at::globalContext().setAllowTF32CuBLAS(false);torch::set_num_threads(2);torch::manual_seed(0);
  TORCH_CHECK(argc==3,"output and config required");std::filesystem::create_directories(argv[1]);
  ContextReplay mapper(argv[1],argv[2]);mapper.run(argv[1]);
}

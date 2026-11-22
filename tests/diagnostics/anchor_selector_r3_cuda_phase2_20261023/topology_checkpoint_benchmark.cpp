// 在真实checkpoint上触发受控topology事件；合成梯度仅用于事件测试，非在线质量实验。
#include "include/gaussian_model.h"
#include "improvements/selector/gumbel_network.h"
#include "include/mapping_cuda_optimizer.h"
#include <cuda_runtime_api.h>
#include <nvToolsExt.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>
using T=torch::Tensor;namespace fs=std::filesystem;using Net=improvements::selector::GumbelNetwork;
struct DiagnosticModel:GaussianModel {
  using GaussianModel::GaussianModel;
  void restoreWindows(){auto a=explicit_anchors_->size(),n=xyz_.size(0);auto options=xyz_.options().requires_grad(false);
    anchor_opacity_accum_=torch::zeros({a},options.dtype(torch::kFloat64));anchor_visits_=torch::zeros({a},options.dtype(torch::kLong));
    anchor_slot_opacity_accum_=torch::zeros({n},options.dtype(torch::kFloat64));anchor_slot_below_prune_=torch::zeros({n},options.dtype(torch::kLong));anchor_slot_below_cutoff_=torch::zeros_like(anchor_slot_below_prune_);
    anchor_growth_gradient_accum_=torch::zeros({n,1},options);anchor_growth_denom_=torch::ones({n,1},options);anchor_growth_threshold_=.5;
  }
  void growth(){anchor_growth_gradient_accum_.slice(0,0,6).fill_(1);}
  void pruneFirst(){anchor_visits_[0].fill_(401);}
};
int main(int argc,char** argv){try{
  TORCH_CHECK(argc==3,"topology_checkpoint_benchmark checkpoint output");torch::set_num_threads(2);torch::manual_seed(2026);fs::path cp=argv[1],out=argv[2];fs::create_directories(out);
  std::ofstream csv(out/"topology_events.csv");csv<<"scenario,repeat,fused,N_before,A_before,N_after,A_after,host_ms,cuda_ms\n";
  fs::path cloud;for(auto& p:fs::directory_iterator(cp/"point_cloud"))if(p.is_directory())cloud=p.path();
  for(const std::string scenario:{"noop_gaussian","gaussian_density","anchor_growth","anchor_prune"})for(int rep=0;rep<12;++rep)for(int order=0;order<2;++order){bool fused=(rep+order)%2==0;
    Net net(.1,true,false,true);net->to(torch::kCUDA);torch::serialize::InputArchive ar;ar.load_from((cp/"selector.pt").string(),torch::kCUDA);net->load(ar);auto bank=net->anchors();bank->setFusedFullStats(fused);
    torch::optim::Adam auxiliary(net->parameters(),torch::optim::AdamOptions(.001).eps(1e-15));GaussianModelParams model_options;model_options.sh_degree_=3;model_options.data_device_="cuda";
    DiagnosticModel g(model_options);g.enableExplicitAnchors(bank,&auxiliary);g.enablePrivateCompensation(net->privateCompensation());g.loadPly(cloud/"point_cloud.ply");g.loadSelectorMetadataPly(cloud/"selector_metadata.ply");g.restoreWindows();
    for(auto p:{g.xyz_,g.features_dc_,g.features_rest_,g.opacity_,g.scaling_,g.rotation_})p.requires_grad_(true);
    g.spatial_lr_scale_=1;GaussianOptimizationParams options;g.trainingSetup(options);g.optimizer_->add_param_group(torch::optim::OptimizerParamGroup(net->parameters()));
    const auto n=g.xyz_.size(0),a=bank->size();
    // Inference PLY has no birth clock; this controlled event fixture uses an
    // explicit identical synthetic clock, rather than claiming full resume.
    g.exist_since_iter_=torch::zeros({n},g.xyz_.options().dtype(torch::kLong));
    auto loss=g.xyz_.sum()+g.features_dc_.sum()+g.features_rest_.sum()+g.opacity_.sum()+g.scaling_.sum()+g.rotation_.sum()+bank->features().sum()+net->privateCompensation()->forward(torch::arange(n,bank->rows().options()),.05).sum();loss.backward();
    g.optimizer_->step();auxiliary.step();g.optimizer_->zero_grad(true);auxiliary.zero_grad(true);
    g.xyz_gradient_accum_=torch::zeros({n,1},g.xyz_.options());g.denom_=torch::ones_like(g.xyz_gradient_accum_);g.max_radii2D_=torch::zeros({n},g.xyz_.options());g.setAnchorTopologyPolicy(true,false);
    {torch::NoGradGuard guard;bank->topologyRound().fill_(scenario=="anchor_growth"||scenario=="anchor_prune"?1:0);
      if(scenario=="gaussian_density")g.xyz_gradient_accum_.slice(0,0,16).fill_(1);if(scenario=="anchor_growth")g.growth();if(scenario=="anchor_prune")g.pruneFirst();}
    torch::manual_seed(2026+rep);cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);torch::cuda::synchronize();cudaEventRecord(begin);auto start=std::chrono::steady_clock::now();
    nvtxRangePushA((scenario+(fused?"/fused":"/legacy")).c_str());g.densifyAndPrune(.0002,.02,2,0,500);nvtxRangePop();
    auto host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();cudaEventRecord(end);cudaEventSynchronize(end);float ms;cudaEventElapsedTime(&ms,begin,end);
    csv<<scenario<<','<<rep<<','<<fused<<','<<n<<','<<a<<','<<g.xyz_.size(0)<<','<<bank->size()<<','<<host<<','<<ms<<'\n';cudaEventDestroy(begin);cudaEventDestroy(end);
    bank->assertBlocks(g.xyz_.size(0));TORCH_CHECK(net->privateCompensation()->size()==g.xyz_.size(0),"R3 population after event");
    if(rep==0){torch::serialize::OutputArchive fixture;auto put=[&](std::string key,const T& value){fixture.write(key,value.detach().cpu());};
      put("owner",bank->rows());put("gaussian_uid",net->privateCompensation()->uids());put("counts",bank->memberCounts());put("levels",bank->levels());put("keys",bank->named_buffers()["keys"]);put("anchor_uid",bank->blockUIDs());
      std::vector<T> params{g.xyz_,g.features_dc_,g.features_rest_,g.opacity_,g.scaling_,g.rotation_,bank->features()};auto tables=net->privateCompensation()->tables();params.insert(params.end(),tables.begin(),tables.end());
      for(size_t i=0;i<params.size();++i){put("parameter_"+std::to_string(i),params[i]);for(size_t j=0;j<2;++j){auto& optimizer=j?auxiliary:*g.optimizer_;auto found=optimizer.state().find(params[i].unsafeGetTensorImpl());if(found==optimizer.state().end())continue;auto& state=static_cast<torch::optim::AdamParamState&>(*found->second);auto name="optimizer_"+std::to_string(j)+"_"+std::to_string(i);put(name+"_m",state.exp_avg());put(name+"_v",state.exp_avg_sq());put(name+"_step",torch::tensor(state.step(),torch::kLong));}}
      fixture.save_to((out/(scenario+(fused?"_fused.pt":"_legacy.pt"))).string());}
  }std::cout<<"PASS real checkpoint controlled topology events and R3 row alignment\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
